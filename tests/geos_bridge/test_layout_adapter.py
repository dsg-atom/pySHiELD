"""Fortran-vs-C array-order adapter for the radiation CFFI bridge (Phase 2).

Pure numpy + cffi: NO ndsl, pyRTE, or gt4py import, so this runs ANYWHERE
(including a laptop) and on Discover inside the fork-a venv:

    pytest tests/geos_bridge/test_layout_adapter.py -q

It proves the one Phase-2 piece that needs no GEOS fork: ``RadiationR8Conversion``
turning a flat Fortran column-major (``ncol`` fastest) ``double*`` buffer -- the
layout ``GEOS_IrradGridComp`` / ``GEOS_SolarGridComp`` allocate -- into the
C-contiguous ``(ncol, nlev)`` array the rest of the bridge expects, and writing an
output array back into an identical Fortran-order byte buffer.

The known field encodes BOTH indices (``value = col*1000 + lev``) with ncol != nlev,
so a column/level transposition or a C-vs-F mix-up is unambiguously caught.
"""

import cffi
import numpy as np
import pytest

from pyshield.geos_bridge.cffi.radiation_f_py_conversion import RadiationR8Conversion


# ncol = nx*ny = 6, layer nlev = nz = 4, level nlev = nz+1 = 5. All distinct so a
# transposition cannot accidentally pass.
NX, NY, NZ = 2, 3, 4
NCOL = NX * NY


def _known_field(ncol, nlev):
    """C-contiguous (ncol, nlev) with value[c, l] = c*1000 + l (both indices)."""
    c = np.arange(ncol, dtype=np.float64).reshape(ncol, 1)
    l = np.arange(nlev, dtype=np.float64).reshape(1, nlev)
    return np.ascontiguousarray(c * 1000.0 + l)


def _ptr(conv, flat):
    """A ``double*`` CData onto ``flat``'s memory (same FFI as the adapter)."""
    return conv._ffi.from_buffer("double[]", flat)


# --- inputs: flat Fortran/C buffer -> C-contiguous (ncol, nlev) ---------------


@pytest.mark.parametrize("kind,nlev", [("layer", NZ), ("level", NZ + 1)])
def test_input_fortran_order_folds_to_c_contiguous(kind, nlev):
    conv = RadiationR8Conversion(NX, NY, NZ, order="F")
    truth = _known_field(NCOL, nlev)

    # Simulate GEOS memory: column-major flat bytes (ncol varies fastest).
    fortran_flat = np.ascontiguousarray(truth.flatten(order="F"))
    # Sanity on the test's own setup: element (c, l) sits at index c + l*ncol.
    for c in range(NCOL):
        for l in range(nlev):
            assert fortran_flat[c + l * NCOL] == truth[c, l]

    out = conv._to_numpy(_ptr(conv, fortran_flat), kind)

    assert out.shape == (NCOL, nlev)
    assert out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(out, truth)


@pytest.mark.parametrize("kind,nlev", [("layer", NZ), ("level", NZ + 1)])
def test_input_c_order_is_identity_fold(kind, nlev):
    conv = RadiationR8Conversion(NX, NY, NZ, order="C")
    truth = _known_field(NCOL, nlev)

    c_flat = np.ascontiguousarray(truth.flatten(order="C"))
    out = conv._to_numpy(_ptr(conv, c_flat), kind)

    assert out.shape == (NCOL, nlev)
    np.testing.assert_array_equal(out, truth)


def test_fortran_and_c_differ_for_3d_field():
    """The two orders must actually produce different folds (guards a no-op bug)."""
    conv_f = RadiationR8Conversion(NX, NY, NZ, order="F")
    conv_c = RadiationR8Conversion(NX, NY, NZ, order="C")
    truth = _known_field(NCOL, NZ)
    flat = np.ascontiguousarray(truth.flatten(order="F"))

    as_f = conv_f._to_numpy(_ptr(conv_f, flat), "layer")
    as_c = conv_c._to_numpy(_ptr(conv_c, flat), "layer")

    np.testing.assert_array_equal(as_f, truth)      # F reads it correctly
    assert not np.array_equal(as_c, truth)          # C mis-reads the same bytes


def test_col_field_order_invariant():
    """A 1-D (ncol,) field is identical under C and F order."""
    truth = np.arange(NCOL, dtype=np.float64) * 7.0 + 3.0
    for order in ("C", "F"):
        conv = RadiationR8Conversion(NX, NY, NZ, order=order)
        flat = np.ascontiguousarray(truth.copy())
        out = conv._to_numpy(_ptr(conv, flat), "col")
        assert out.shape == (NCOL,)
        np.testing.assert_array_equal(out, truth)


# --- outputs: C-contiguous (ncol, nlev) -> flat Fortran/C buffer --------------


@pytest.mark.parametrize("nlev", [NZ, NZ + 1])
def test_output_written_back_in_fortran_order(nlev):
    conv = RadiationR8Conversion(NX, NY, NZ, order="F")
    result = _known_field(NCOL, nlev)

    dest = np.zeros(NCOL * nlev, dtype=np.float64)
    conv.copy_outputs({"x": result}, {"x": _ptr(conv, dest)})

    # The flat buffer must hold the column-major byte order GEOS expects.
    np.testing.assert_array_equal(dest, result.flatten(order="F"))


@pytest.mark.parametrize("nlev", [NZ, NZ + 1])
def test_output_written_back_in_c_order(nlev):
    conv = RadiationR8Conversion(NX, NY, NZ, order="C")
    result = _known_field(NCOL, nlev)

    dest = np.zeros(NCOL * nlev, dtype=np.float64)
    conv.copy_outputs({"x": result}, {"x": _ptr(conv, dest)})

    np.testing.assert_array_equal(dest, result.flatten(order="C"))


# --- full round-trip: Fortran bytes in -> adapter -> Fortran bytes out --------


def test_fortran_roundtrip_is_byte_identical():
    conv = RadiationR8Conversion(NX, NY, NZ, order="F")
    truth = _known_field(NCOL, NZ)
    fortran_in = np.ascontiguousarray(truth.flatten(order="F"))

    folded = conv._to_numpy(_ptr(conv, fortran_in), "layer")  # F -> C-contiguous

    fortran_out = np.zeros(NCOL * NZ, dtype=np.float64)
    conv.copy_outputs({"x": folded}, {"x": _ptr(conv, fortran_out)})

    np.testing.assert_array_equal(fortran_out, fortran_in)


# --- flag handling ------------------------------------------------------------


def test_env_default_is_fortran(monkeypatch):
    monkeypatch.delenv("GEOS_RRTMGP_ARRAY_ORDER", raising=False)
    assert RadiationR8Conversion(NX, NY, NZ).order == "F"


def test_env_selects_c_order(monkeypatch):
    monkeypatch.setenv("GEOS_RRTMGP_ARRAY_ORDER", "C")
    assert RadiationR8Conversion(NX, NY, NZ).order == "C"


def test_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("GEOS_RRTMGP_ARRAY_ORDER", "F")
    assert RadiationR8Conversion(NX, NY, NZ, order="C").order == "C"


def test_bad_order_rejected():
    with pytest.raises(ValueError):
        RadiationR8Conversion(NX, NY, NZ, order="Z")

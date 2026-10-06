"""Marshalling round-trip for the GEOS radiation bridge (Phase 0).

Runs STANDALONE on Discover (numpy backend) with no staged data and no
coefficient download: it builds the real RTE_RRTMGPState / SurfaceState with a
single-tile quantity_factory, fills synthetic whole-tile ``(ncol, nz)`` numpy
inputs, marshals them IN, then reads the state's compute interior back OUT and
asserts the trip is lossless -- including the vertical flip (top_at_1) and the
``ncol = nx*ny`` row-major fold.

This proves the Phase-0 deliverable (the marshalling contract) without the
driver, pyRTE coefficients, or any restart. Import of pyshield.radiation.state
pulls in pyRTE as a module (installed on Discover) but triggers NO coefficient
download -- nothing here constructs GasOptics.

Run on Discover inside the fork-a venv:
    pytest tests/geos_bridge/test_marshal_roundtrip.py -q
"""

import numpy as np
import pytest

from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import backend_python
from ndsl.dsl.typing import Float

from pyshield.geos_bridge import marshal
from pyshield.radiation.state import RTE_RRTMGPState
from pyshield.stencils.surface import SurfaceState


NX, NY, NZ, NHALO = 3, 4, 10, 3
NCOL = NX * NY


@pytest.fixture
def states():
    _, qf = get_factories_single_tile(nx=NX, ny=NY, nz=NZ, nhalo=NHALO, backend=backend_python)
    return RTE_RRTMGPState.init_zeros(qf), SurfaceState.init_zeros(qf)


def _ramp(shape):
    """Distinct deterministic value per (column, level): c*1000 + k."""
    arr = np.zeros(shape, dtype=Float)
    if arr.ndim == 1:
        arr[:] = np.arange(shape[0]) * 1000.0
    else:
        for c in range(shape[0]):
            arr[c, :] = c * 1000.0 + np.arange(shape[1])
    return arr


@pytest.mark.parametrize("top_at_1", [True, False])
def test_roundtrip_lossless(states, top_at_1):
    state, sfc = states

    inputs = {
        "prsi": _ramp((NCOL, NZ + 1)),  # 3-D interface (level)
        "prsl": _ramp((NCOL, NZ)),      # 3-D layer
        "tlyr": _ramp((NCOL, NZ)),
        "tsfc": _ramp((NCOL,)),         # 2-D column
    }
    sfc_inputs = {
        "tsfc": _ramp((NCOL,)),
        "islmsk": _ramp((NCOL,)),       # 2-D int-backed field
        "snowd": _ramp((NCOL,)),
    }

    marshal.marshal_inputs(state, sfc, inputs, sfc_inputs, NX, NY, NZ, top_at_1)

    # Read every field back out of the state's compute interior and compare.
    for name, arr in inputs.items():
        back = marshal.marshal_field_out(
            getattr(state, name), marshal._dims_of(state, name), NX, NY, NZ, top_at_1
        )
        np.testing.assert_array_equal(back, arr, err_msg=f"state.{name} round-trip")
    for name, arr in sfc_inputs.items():
        back = marshal.marshal_field_out(
            getattr(sfc, name), marshal._dims_of(sfc, name), NX, NY, NZ, top_at_1
        )
        np.testing.assert_array_equal(back, arr, err_msg=f"sfc.{name} round-trip")


def test_fold_is_row_major(states):
    """column c = i*ny + j maps to view[i, j]."""
    state, _ = states
    arr = _ramp((NCOL, NZ))
    marshal.marshal_field_in(
        state.prsl, marshal._dims_of(state, "prsl"), arr, NX, NY, NZ, top_at_1=False
    )
    view = np.asarray(state.prsl.view[:])
    for i in range(NX):
        for j in range(NY):
            c = i * NY + j
            np.testing.assert_array_equal(view[i, j, :], arr[c, :])


def test_vertical_flip_applied(states):
    """top_at_1=True must flip K so the state's K=0 is the caller's TOA bottom."""
    state, _ = states
    arr = _ramp((NCOL, NZ))  # arr[c, 0] = TOA (top-down), arr[c, -1] = surface
    marshal.marshal_field_in(
        state.prsl, marshal._dims_of(state, "prsl"), arr, NX, NY, NZ, top_at_1=True
    )
    view = np.asarray(state.prsl.view[:])
    # State is bottom-up: state K=0 must hold the caller's LAST (surface) level.
    for i in range(NX):
        for j in range(NY):
            c = i * NY + j
            assert view[i, j, 0] == arr[c, NZ - 1]
            assert view[i, j, NZ - 1] == arr[c, 0]


def test_no_flip_when_top_at_1_false(states):
    state, _ = states
    arr = _ramp((NCOL, NZ))
    marshal.marshal_field_in(
        state.prsl, marshal._dims_of(state, "prsl"), arr, NX, NY, NZ, top_at_1=False
    )
    view = np.asarray(state.prsl.view[:])
    for i in range(NX):
        for j in range(NY):
            c = i * NY + j
            assert view[i, j, 0] == arr[c, 0]


def test_halo_untouched(states):
    """Marshalling writes only the compute interior; the halo stays zero."""
    state, _ = states
    arr = _ramp((NCOL, NZ)) + 1.0  # strictly non-zero everywhere
    marshal.marshal_field_in(
        state.prsl, marshal._dims_of(state, "prsl"), arr, NX, NY, NZ, top_at_1=False
    )
    full = np.asarray(state.prsl.field)
    # Horizontal halo ring (first/last nhalo rows+cols) must still be zero.
    assert np.all(full[:NHALO, :, :] == 0.0)
    assert np.all(full[:, :NHALO, :] == 0.0)


def test_bad_shape_raises(states):
    state, _ = states
    with pytest.raises(ValueError):
        marshal.marshal_field_in(
            state.prsl, marshal._dims_of(state, "prsl"),
            np.zeros((NCOL, NZ + 5), dtype=Float), NX, NY, NZ, top_at_1=False,
        )

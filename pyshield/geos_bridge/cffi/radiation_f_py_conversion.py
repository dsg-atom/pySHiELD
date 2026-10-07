"""R8 (float64) whole-tile Fortran <-> numpy marshalling for the radiation CFFI bridge.

Radiation-specific analogue of the gtFV3 bridge's ``f_py_conversion.py``
(``gpu-analysis/FVdycoreCubed_GridComp/geos-gtfv3/f_py_conversion.py``). It is a
SEPARATE file on purpose: gtFV3 marshals float32 and we must NOT edit it.

Why a radiation-specific copy exists
-------------------------------------
RRTMGP is double precision (R8). Every field the bridge passes is ``c_double``,
so every buffer read and every copy-back must use an 8-byte element size.

gtFV3's ``f_py_conversion.py`` hardcodes a 4-byte (sizeof float32) stride in its
copy-back ``memmove`` (``f_py_conversion.py:325``:
``self._ffi.memmove(fptr + ptr_offset, numpy_array, 4 * numpy_array.size)``).
For R8 data that under-copies HALF the bytes and corrupts the returned fluxes.
This module uses ``ffi.sizeof("double")`` (== 8) as the element stride instead;
see :meth:`RadiationR8Conversion.copy_outputs`.

Layout contract and the Phase-2 array-order adapter
----------------------------------------------------
The Fortran side hands each field as a flat ``real(c_double) dimension(*)``
buffer whose logical shape is ``(ncol, nlev)`` (or ``(ncol,)`` for a 2-D field),
``ncol = nx*ny`` row-major with ``column c = i*ny + j``. How those flat elements
are ordered in memory depends on who allocated the buffer:

* **C order** (``GEOS_RRTMGP_ARRAY_ORDER="C"``) -- the Phase-1 standalone driver
  (``driver/geos_rrtmgp_driver.f90``) fills ``idx = (c-1)*nlev + l``, i.e. level
  varies fastest, so the flat buffer is already C-contiguous ``(ncol, nlev)`` and
  a plain ``np.frombuffer(...).reshape(ncol, nlev)`` is the identity fold.

* **F order** (``GEOS_RRTMGP_ARRAY_ORDER="F"``, the default -- what real GEOS
  needs) -- ``GEOS_IrradGridComp`` / ``GEOS_SolarGridComp`` allocate whole-tile
  ``(ncol, LM)`` / ``(ncol, LM+1)`` arrays in Fortran column-major order, so the
  COLUMN index varies fastest in memory (element ``(c, l)`` lives at flat index
  ``c + l*ncol``). To present the C-contiguous ``(ncol, nlev)`` array the rest of
  the bridge expects, the flat buffer is read with
  ``np.frombuffer(...).reshape(shape, order="F")`` and then made contiguous with
  ``np.ascontiguousarray``. Outputs are the reverse: the C-contiguous result is
  written back with ``arr.flatten(order="F")`` so the Fortran buffer regains its
  column-major byte order. This is the radiation analogue of gtFV3's
  ``_transform_from_fortran_layout`` (reshape-reversed + transpose) in
  ``gpu-analysis/FVdycoreCubed_GridComp/geos-gtfv3/f_py_conversion.py``, done in R8.

The ``order`` is selectable per instance (constructor arg) and falls back to the
``GEOS_RRTMGP_ARRAY_ORDER`` env var, default ``"F"``. The Phase-1 driver must set
``GEOS_RRTMGP_ARRAY_ORDER=C`` (see ``driver/build_and_run.sh``). For a 1-D
``"col"`` field C order and F order are identical, so the flag only matters for
3-D fields.

The field -> vertical-kind maps below mirror ``marshal._kind``:
``"level"`` -> ``nz+1`` (K_INTERFACE), ``"layer"`` -> ``nz`` (K), ``"col"`` -> 2-D.
"""

import os

import cffi
import numpy as np


# RTE_RRTMGPState base inputs the bridge marshals in (subset of
# marshal.RAD_INPUT_FIELDS: the real inputs; tlvl/mu0/albedo/sfc_emis are
# recomputed by the driver so they are not passed).
RAD_INPUT_KIND = {
    "prsi": "level",
    "prsl": "layer",
    "tlyr": "layer",
    "tsfc": "col",
    "qvapor": "layer",
    "qo3mr": "layer",
    "co2": "layer",
    "qliquid": "layer",
    "qice": "layer",
    "qcld": "layer",
}

# SurfaceState inputs. Under the chosen flags (ialbflg=-1 constant albedo,
# iemsflg=0 fixed emissivity) set_albedo / set_sfcemis take fixed-value paths and
# read only tsfc + islmsk; additional SFC_INPUT_FIELDS are added here (and in the
# f90/c/py ABI) when a flag that reads them is enabled.
SFC_INPUT_KIND = {
    "tsfc": "col",
    "islmsk": "col",
}

# RTE_RRTMGPState outputs (marshal.RAD_OUTPUT_FIELDS).
RAD_OUTPUT_KIND = {
    "flwu": "level",
    "flwd": "level",
    "fswu": "level",
    "fswd": "level",
    "fswn": "col",
    "flwu_clr": "level",
    "flwd_clr": "level",
    "fswu_clr": "level",
    "fswd_clr": "level",
    "hrtlw": "layer",
    "hrtsw": "layer",
    "hrtlw_clr": "layer",
    "hrtsw_clr": "layer",
}


class RadiationR8Conversion:
    """Marshal flat ``c_double`` Fortran buffers <-> whole-tile float64 numpy.

    Built once at bridge init (holds the tile dims) and reused each run, like
    gtFV3's ``FortranPythonConversion``. Carries its own ``cffi.FFI()`` purely for
    the primitive ``buffer`` / ``memmove`` / ``sizeof`` operations on the
    ``double*`` CData the embedding layer hands in (the same pattern gtFV3 uses;
    primitive C types are shared across FFI instances).
    """

    def __init__(self, nx: int, ny: int, nz: int, order: str = None):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.ncol = nx * ny
        self._ffi = cffi.FFI()
        # Element stride for every copy: R8 => 8 bytes. NOT 4 (the float32 bug).
        self._itemsize = self._ffi.sizeof("double")
        assert self._itemsize == 8, "radiation bridge requires 8-byte (R8) doubles"
        # Memory order of the incoming/outgoing flat Fortran buffers. "F" (default)
        # = real GEOS column-major (ncol varies fastest); "C" = the Phase-1 driver's
        # C-contiguous (ncol, nlev). Falls back to GEOS_RRTMGP_ARRAY_ORDER.
        order = order or os.environ.get("GEOS_RRTMGP_ARRAY_ORDER", "F")
        self.order = str(order).upper()
        if self.order not in ("C", "F"):
            raise ValueError(
                f"GEOS_RRTMGP_ARRAY_ORDER must be 'C' or 'F', got {order!r}"
            )

    def _nlev(self, kind: str) -> int:
        return self.nz + 1 if kind == "level" else self.nz

    def _shape(self, kind: str):
        if kind == "col":
            return (self.ncol,)
        return (self.ncol, self._nlev(kind))

    def _to_numpy(self, fptr, kind: str) -> np.ndarray:
        """Read a flat ``double*`` buffer as a C-contiguous (ncol[, nlev]) array.

        For ``order="C"`` the flat buffer is already C-contiguous ``(ncol, nlev)``
        and ``reshape`` is a zero-copy view (marshal only READS inputs). For
        ``order="F"`` the flat buffer is Fortran column-major, so it is reshaped
        with ``order="F"`` (interpreting ncol as the fastest-varying axis) and then
        made C-contiguous for the downstream marshal/state code.
        """
        shape = self._shape(kind)
        nelem = int(np.prod(shape))
        buf = self._ffi.buffer(fptr, nelem * self._itemsize)
        flat = np.frombuffer(buf, dtype=np.float64)
        if self.order == "F":
            # reshape(order="F") folds the column-major flat buffer into the
            # logical (ncol, nlev); ascontiguousarray gives the C layout the rest
            # of the bridge expects. (For a 1-D "col" field this is a no-op fold.)
            return np.ascontiguousarray(flat.reshape(shape, order="F"))
        return flat.reshape(shape)

    def inputs_from_fortran(self, ptrs: dict) -> dict:
        """``{name: double* CData}`` -> ``{name: (ncol[, nlev]) float64}`` for state inputs."""
        return {name: self._to_numpy(ptrs[name], RAD_INPUT_KIND[name]) for name in RAD_INPUT_KIND}

    def sfc_from_fortran(self, ptrs: dict) -> dict:
        """``{name: double* CData}`` -> ``{name: (ncol,) float64}`` for surface inputs."""
        return {name: self._to_numpy(ptrs[name], SFC_INPUT_KIND[name]) for name in SFC_INPUT_KIND}

    def copy_outputs(self, out_dict: dict, out_ptrs: dict) -> None:
        """Copy whole-tile output arrays back into the Fortran ``double*`` buffers.

        ``out_dict`` is what ``geos_rrtmgp_run`` returns (contiguous float64,
        shaped (ncol[, nlev])). Each is memmove'd into its matching Fortran buffer.
        """
        for name, fptr in out_ptrs.items():
            arr = np.asarray(out_dict[name], dtype=np.float64)
            if self.order == "F":
                # Reverse of the input fold: ravel in Fortran order so the flat
                # bytes regain GEOS's column-major (ncol fastest) layout. flatten
                # returns a C-contiguous copy, which memmove requires.
                flat = arr.flatten(order="F")
            else:
                flat = np.ascontiguousarray(arr).ravel(order="C")
            # R8 COPY-BACK: element stride is sizeof(double) == 8 bytes, so the
            # byte count is 8 * size. gtFV3's f_py_conversion.py:325 hardcodes
            # 4 * size (sizeof float32); for R8 that copies only half the bytes
            # and corrupts the fluxes. ffi.sizeof("double") makes it explicit.
            self._ffi.memmove(fptr, flat, self._itemsize * flat.size)

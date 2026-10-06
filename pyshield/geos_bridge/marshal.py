"""Whole-tile numpy <-> NDSL Quantity marshalling for the GEOS radiation bridge.

This is the unit-testable core of Phase 0 of the GEOS CFFI radiation bridge. It
maps the whole-tile numpy arrays a future CFFI bridge will hand us -- shaped
``(ncol, nz)`` / ``(ncol, nz+1)`` / ``(ncol,)`` with ``ncol = nx*ny`` -- into (and
back out of) the halo'd ``(nx, ny, nz)`` NDSL ``Quantity`` fields of
``RTE_RRTMGPState`` / ``SurfaceState``.

It is deliberately free of any pyRTE / rte_rrtmgp import so the marshalling
round-trip test can run standalone on Discover (numpy backend) with only ndsl
installed -- no coefficient download, no driver construction, no staged data.

Three concerns are handled explicitly:

1. Column fold. ``ncol = nx*ny`` is row-major: column ``c = i*ny + j`` (x varies
   slowest). This is the SAME fold ``RTE_RRTMGPState.to_rterrtmgp_xr`` applies
   (``.reshape(-1, nz)`` on the (nx, ny, nz) compute interior), so a value written
   to column ``c`` lands in the exact pyRTE column the driver computes on.

2. Halos. We write into / read from ``quantity.view[:]`` -- the compute interior,
   with the NDSL halo stripped. For a layer field ``view[:]`` is ``(nx, ny, nz)``
   and for an interface field ``(nx, ny, nz+1)``; these match the ``3:-4`` (interior)
   / ``:-1`` (layer) / ``:`` (level) slices ``to_rterrtmgp_xr`` takes from the full
   buffer. The halo ring is never touched (zero-halo approach), which is correct for
   the single-tile standalone path -- there is no neighbor to exchange with.

3. Vertical orientation. GEOS is top-down (index 0 = top of atmosphere); the
   PySHiELD solvers are compiled ``top_at_1 = False`` (bottom-up, surface at K=0,
   see RTE_RRTMGPDriver._solver_top_at_1). When the caller's arrays are top-down
   (``top_at_1=True``) we flip K on the way IN so the state is bottom-up, and flip
   back on the way OUT. The flip is parameterized because GEOS computes its
   orientation at runtime; a caller already supplying bottom-up data passes
   ``top_at_1=False`` and no flip happens.
"""

import numpy as np

import ndsl.dsl.gt4py_utils as gt_utils
from ndsl.constants import K_DIM, K_INTERFACE_DIM


# ---------------------------------------------------------------------------
# Field groups. These name the RTE_RRTMGPState / SurfaceState fields the bridge
# marshals. Only BASE fields are listed: fields the driver computes internally
# (clwp/cip/clwr/cir from progcld; the recomputed tlvl/mu0/albedo/sfc_emis) are
# NOT required as inputs -- step_radiation fills them from the base state.
# ---------------------------------------------------------------------------

#: RTE_RRTMGPState fields read as radiation inputs. prsi/prsl/tlyr/tsfc/qvapor/
#: qliquid/qice/qo3mr/qcld/co2 are true inputs; tlvl/mu0/albedo/sfc_emis are
#: listed (they are intent "inout") but are OVERWRITTEN by the driver
#: (calc_tlvl_gfs / coszmn / set_albedo / set_sfcemis), so supplying them is
#: optional and harmless.
RAD_INPUT_FIELDS = (
    "prsi",
    "prsl",
    "tlyr",
    "tlvl",
    "tsfc",
    "mu0",
    "albedo",
    "sfc_emis",
    "qvapor",
    "qliquid",
    "qice",
    "qo3mr",
    "qcld",
    "co2",
)

#: The minimum base set the driver actually requires to be non-zero for a
#: physically meaningful solve (everything else is derived internally).
RAD_REQUIRED_INPUT_FIELDS = (
    "prsi",
    "prsl",
    "tlyr",
    "tsfc",
    "qvapor",
    "qo3mr",
    "qcld",
    "qliquid",
    "co2",
)

#: RTE_RRTMGPState output fields (intent "out") the bridge extracts.
RAD_OUTPUT_FIELDS = (
    "flwu",
    "flwd",
    "fswu",
    "fswd",
    "fswn",
    "flwu_clr",
    "flwd_clr",
    "fswu_clr",
    "fswd_clr",
    "hrtlw",
    "hrtsw",
    "hrtlw_clr",
    "hrtsw_clr",
)

#: SurfaceState fields read by set_albedo / set_sfcemis / progcld (GFS physics).
SFC_INPUT_FIELDS = (
    "tsfc",
    "islmsk",
    "snowd",
    "sncovr",
    "snoalb",
    "zorl",
    "hprim",
    "alvsf",
    "alnsf",
    "alvwf",
    "alnwf",
    "facsf",
    "facwf",
    "fice",
    "tisfc",
)


def _dims_of(state_obj, name):
    """Return the ndsl dim list declared in a dataclass field's metadata."""
    return state_obj.__dataclass_fields__[name].metadata["dims"]


def _kind(dims):
    """Classify a field's vertical structure from its ndsl dims.

    Returns "level" (K_INTERFACE_DIM, nz+1), "layer" (K_DIM, nz), or "col" (2-D,
    horizontal only).
    """
    if K_INTERFACE_DIM in dims:
        return "level"
    if K_DIM in dims:
        return "layer"
    return "col"


def _nlev(kind, nz):
    return nz + 1 if kind == "level" else nz


def marshal_field_in(quantity, dims, arr, nx, ny, nz, top_at_1):
    """Write a whole-tile numpy array into a Quantity's compute interior.

    Args:
        quantity: the NDSL ``Quantity`` to fill.
        dims: that field's ndsl dim list (from its dataclass metadata).
        arr: whole-tile numpy array, ``(ncol,)`` for a 2-D field or
            ``(ncol, nlev)`` for a 3-D field, ``ncol = nx*ny`` row-major.
        nx, ny, nz: single-tile compute dims.
        top_at_1: if True the array is top-down and K is flipped to the state's
            bottom-up ordering; if False no flip.
    """
    kind = _kind(dims)
    view = quantity.view[:]
    arr = np.asarray(arr)
    if kind == "col":
        expected = (nx * ny,)
        if arr.shape != expected:
            raise ValueError(
                f"2-D field expected shape {expected}, got {arr.shape}"
            )
        view[:] = arr.reshape(nx, ny)
        return
    nlev = _nlev(kind, nz)
    expected = (nx * ny, nlev)
    if arr.shape != expected:
        raise ValueError(f"3-D field expected shape {expected}, got {arr.shape}")
    src = arr[:, ::-1] if top_at_1 else arr
    view[:] = src.reshape(nx, ny, nlev)


def marshal_field_out(quantity, dims, nx, ny, nz, top_at_1):
    """Read a Quantity's compute interior back into a whole-tile numpy array.

    Inverse of :func:`marshal_field_in`: folds (nx, ny) -> ncol row-major and,
    if ``top_at_1``, flips K back to the caller's top-down ordering. Returns a
    contiguous host numpy array (``(ncol,)`` or ``(ncol, nlev)``).
    """
    kind = _kind(dims)
    view = gt_utils.asarray(quantity.view[:])
    if kind == "col":
        return np.ascontiguousarray(view.reshape(nx * ny))
    nlev = _nlev(kind, nz)
    out = view.reshape(nx * ny, nlev)
    if top_at_1:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)


def marshal_inputs(
    state,
    sfc_state,
    inputs,
    sfc_inputs,
    nx,
    ny,
    nz,
    top_at_1,
):
    """Marshal whole-tile input dicts into the state + surface-state Quantities.

    Args:
        state: RTE_RRTMGPState.
        sfc_state: SurfaceState.
        inputs: ``{field_name: whole_tile_array}`` for RTE_RRTMGPState fields.
        sfc_inputs: ``{field_name: whole_tile_array}`` for SurfaceState fields.
        nx, ny, nz, top_at_1: see marshal_field_in.
    """
    for name, arr in inputs.items():
        quantity = getattr(state, name)
        marshal_field_in(quantity, _dims_of(state, name), arr, nx, ny, nz, top_at_1)
    for name, arr in sfc_inputs.items():
        quantity = getattr(sfc_state, name)
        marshal_field_in(
            quantity, _dims_of(sfc_state, name), arr, nx, ny, nz, top_at_1
        )


def extract_outputs(state, nx, ny, nz, top_at_1, fields=RAD_OUTPUT_FIELDS):
    """Extract the driver's output Quantities into a dict of whole-tile arrays."""
    out = {}
    for name in fields:
        quantity = getattr(state, name)
        out[name] = marshal_field_out(
            quantity, _dims_of(state, name), nx, ny, nz, top_at_1
        )
    return out

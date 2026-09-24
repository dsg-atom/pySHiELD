"""GT4Py port of RRTMGP gas optics.

This module reimplements, as NDSL/GT4Py stencils, the RRTMGP gas-optics
kernels that pyRTE-RRTMGP currently runs on the CPU (the four
`GasOptics.compute` calls in `rte_rrtmgp.py`). The goal is a GPU-resident
gas optics so the radiation solve stays on-device with the rest of the
resident block (dynamics + moisture).

Port order follows the Fortran kernel
`mo_gas_optics_rrtmgp_kernels.F90`:
  1. interpolation      -> T/p indices+fractions, eta, fmajor/fminor  (this file, staged)
  2. compute_tau_absorption -> tau (major + minor gas gathers)
  3. compute_Planck_source  -> layer/level/surface Planck sources

Each stage is validated against pyRTE's output (`tests/gas_optics/`), on the
numpy backend first (correctness) then on cupy (A100 timing).

Stage 1: the temperature/pressure part of `interpolation`
(Fortran lines 107-122) -- per-cell index and fraction in the temperature
and pressure table axes, and the troposphere flag. Pure float math so the
GT4Py harness is proven before table gathers (GlobalTable) are introduced.

Stage 2: the eta / binary-species part of `interpolation` for one
flavor -- gathers the reference volume-mixing-ratio table `vmr_ref` by the
runtime temperature/atmosphere indices to form the mixing ratio, then the
eta interpolation index `jeta` and fraction `feta`. This is the first
`GlobalTable` gather in the port: `table.A[i, j, k]` read by runtime integer
indices. The gas indices for the flavor are compile-time externals (the
flavor->gas bookkeeping is host-side numpy setup, not a stencil).

Stage 3: the major-gas 8-point k-table interpolation
(`interpolate3D_byflav`, kernels lines 760-789), for one g-point. First
4-axis `GlobalTable` gather (kmajor) and first integer index arithmetic
(jtemp+1, jpress+1, jeta+1) used as gather indices. The g-point runs as a
compile-time external.

Stage 3b (here): the same 8-point interpolation selecting the g-point by a
runtime index field, so one compiled stencil fills every g-point when driven
in a Python loop over g-points. This is how the port represents the
g-point axis: the columns and layers are the framework's (i, j, k), and the
g-points are a host loop that writes each result into the g-point data axis
of a `tau` field. A single in-stencil compile-time-unrolled g-point loop is
not available here -- this gt4py.cartesian frontend has no `for ... in
range()` (only `while`), and it unrolls data-dim accesses at compile time, so
a runtime data-dim write index is not supported. The loop-driven form reuses
only already-proven pieces (the Stage-2 runtime-index `.A` gather) and keeps
`tau` a single resident data-dimension Quantity.

Stage 4: the major/minor interpolation weights (`fmajor`, `fminor`) from the
temperature/pressure/eta fractions (tail of the Fortran `interpolation`
routine, kernels lines 158-166). Pure float math, no gather. This completes
the `interpolation` routine: it turns Stage-1 `ftemp`/`fpress` and Stage-2
`feta` into the eight `fmajor` weights the Stage-3 kmajor interpolation
consumes and the four `fminor` weights the minor-gas 2-D interpolation
consumes.
"""

from ndsl.dsl.gt4py import GlobalTable, PARALLEL, computation, floor, interval, log
from ndsl.dsl.typing import Float, FloatField, IntField

# Number of longwave g-points (LW_G256). tau carries these as a data
# dimension ("gpt"); the caller allocates it via
# quantity_factory.add_data_dimensions({"gpt": NGPT}) and fills one g-point
# slice per stencil call.
NGPT = 256

# vmr_ref reference table, Fortran layout (atmos_layer=2, absorber_ext=20,
# temperature=14) for the LW_G256 coefficient file. GlobalTable has no spatial
# axes -- only these data dims -- and is gathered read-only with `.A[i, j, k]`.
# The shape is fixed per coefficient file; assert it in the caller.
VmrRef = GlobalTable[(Float, (2, 20, 14))]

# kmajor absorption-coefficient table, Fortran layout
# (temperature=14, eta=9, pressure=60, gpt=256) for LW_G256. netCDF stores it
# (temperature, pressure, eta, gpt); the caller transposes to this order.
KMajor = GlobalTable[(Float, (14, 9, 60, 256))]


def interp_tp(
    play: FloatField,
    tlay: FloatField,
    jtemp: FloatField,
    ftemp: FloatField,
    jpress: FloatField,
    fpress: FloatField,
    tropo: FloatField,
):
    """Temperature/pressure interpolation indices and fractions.

    Mirrors the first double loop of the Fortran `interpolation` routine.
    Indices are 1-based (as in Fortran) and stored as floats for now; they
    become integer gathers when the k-table lookups are added. `tropo` is
    1.0 in the lower atmosphere (troposphere table), 0.0 in the upper.

    Externals (all floats): temp_ref_min, temp_ref_delta, ntemp,
    press_ref_log_1 (= log(press_ref[0])), press_ref_log_delta, npres,
    press_ref_trop_log.
    """
    from __externals__ import (
        npres,
        press_ref_log_1,
        press_ref_log_delta,
        press_ref_trop_log,
        ntemp,
        temp_ref_delta,
        temp_ref_min,
    )

    with computation(PARALLEL), interval(...):
        # temperature index and fraction
        jt = floor((tlay - (temp_ref_min - temp_ref_delta)) / temp_ref_delta)
        jt = min(ntemp - 1.0, max(1.0, jt))
        # temp_ref is uniform, so temp_ref[jt] == temp_ref_min + (jt-1)*delta
        ftemp = (tlay - (temp_ref_min + (jt - 1.0) * temp_ref_delta)) / temp_ref_delta
        jtemp = jt

        # pressure index and fraction (uniform in log-pressure)
        locpress = 1.0 + (log(play) - press_ref_log_1) / press_ref_log_delta
        jp = min(npres - 1.0, max(1.0, floor(locpress)))
        fpress = locpress - jp
        jpress = jp

        # lower vs upper atmosphere
        tropo = 0.0
        if log(play) > press_ref_trop_log:
            tropo = 1.0


def interp_eta_1flavor(
    col_gas1: FloatField,
    col_gas2: FloatField,
    itropo: IntField,
    jt0: IntField,
    jt1: IntField,
    vmr_ref: VmrRef,
    col_mix0: FloatField,
    col_mix1: FloatField,
    jeta0: FloatField,
    jeta1: FloatField,
    feta0: FloatField,
    feta1: FloatField,
):
    """Binary-species (eta) interpolation for one flavor.

    Mirrors the itemp=1,2 body of the Fortran `interpolation` flavor loop.
    For each of the two bracketing reference temperatures it forms the
    reference mixing ratio from `vmr_ref`, the total mixed column `col_mix`,
    the binary species parameter eta, and eta's table index `jeta` (1-based)
    and fraction `feta`.

    Inputs carry the integer table indices already computed on the host:
      `itropo` -- 0=lower / 1=upper atmosphere (vmr_ref axis 0),
      `jt0`, `jt1` -- 0-based temperature indices of the two brackets
                      (jt1 = jt0 + 1), into vmr_ref axis 2.
    `col_gas1`, `col_gas2` are the flavor's two gas column amounts.

    Outputs are floats; `jeta` holds a 1-based integer-valued index (matching
    the Fortran) as a float, deferring the float->int cast to a later stage.

    Externals: igas1, igas2 (compile-time absorber indices into vmr_ref axis 1),
    neta_m1 (= neta - 1, float), eta_half_thresh (float; the col_mix floor
    below which eta defaults to 0.5).
    """
    from __externals__ import eta_half_thresh, igas1, igas2, neta_m1

    with computation(PARALLEL), interval(...):
        # bracket 0 (Fortran itemp=1): reference temperature index jt0
        ratio0 = vmr_ref.A[itropo, igas1, jt0] / vmr_ref.A[itropo, igas2, jt0]
        col_mix0 = col_gas1 + ratio0 * col_gas2
        eta0 = 0.5
        if col_mix0 > eta_half_thresh:
            eta0 = col_gas1 / col_mix0
        loceta0 = eta0 * neta_m1
        jeta0 = min(floor(loceta0) + 1.0, neta_m1)
        feta0 = loceta0 - floor(loceta0)

        # bracket 1 (Fortran itemp=2): reference temperature index jt1 = jt0 + 1
        ratio1 = vmr_ref.A[itropo, igas1, jt1] / vmr_ref.A[itropo, igas2, jt1]
        col_mix1 = col_gas1 + ratio1 * col_gas2
        eta1 = 0.5
        if col_mix1 > eta_half_thresh:
            eta1 = col_gas1 / col_mix1
        loceta1 = eta1 * neta_m1
        jeta1 = min(floor(loceta1) + 1.0, neta_m1)
        feta1 = loceta1 - floor(loceta1)


def interp_weights(
    ftemp: FloatField,
    fpress: FloatField,
    feta1: FloatField,
    feta2: FloatField,
    fmn11: FloatField,
    fmn21: FloatField,
    fmn12: FloatField,
    fmn22: FloatField,
    f111: FloatField,
    f211: FloatField,
    f121: FloatField,
    f221: FloatField,
    f112: FloatField,
    f212: FloatField,
    f122: FloatField,
    f222: FloatField,
):
    """Major/minor interpolation weights from the T/p/eta fractions.

    Port of the weight construction at the tail of the Fortran `interpolation`
    routine (kernels lines 158-166), per cell and per flavor. Pure float math;
    no table gather. Completes the `interpolation` routine.

    Inputs (per cell):
      ftemp  -- temperature fraction (Stage 1, interp_tp),
      fpress -- pressure fraction (Stage 1, interp_tp),
      feta1  -- eta fraction of temperature bracket 1 (Stage 2, itemp=1),
      feta2  -- eta fraction of temperature bracket 2 (Stage 2, itemp=2).

    The temperature-bracket term is `(1-ftemp)` for bracket 1 and `ftemp` for
    bracket 2 (Fortran `(2-itemp) + (2*itemp-3)*ftemp`).

    Outputs:
      fminor fmn<e><t> = fminor(eta-level e, temp-level t):
        fmn11 = (1-feta1)*(1-ftemp), fmn21 = feta1*(1-ftemp),
        fmn12 = (1-feta2)*ftemp,     fmn22 = feta2*ftemp;
      fmajor f<e><p><t> = fmajor(eta-level e, press-level p, temp-level t):
        f<e>1<t> = (1-fpress)*fmn<e><t>,  f<e>2<t> = fpress*fmn<e><t>.
    The eight fmajor feed interp3d_major_1gpt/_gpt; the four fminor feed the
    minor-gas 2-D interpolation. Each of the eight fmajor and the four fminor
    sums to 1 over a cell (partition of unity).
    """
    with computation(PARALLEL), interval(...):
        ft1 = 1.0 - ftemp
        fp0 = 1.0 - fpress

        # fminor (eta-level, temp-level), via the per-bracket temperature term
        m11 = (1.0 - feta1) * ft1
        m21 = feta1 * ft1
        m12 = (1.0 - feta2) * ftemp
        m22 = feta2 * ftemp
        fmn11 = m11
        fmn21 = m21
        fmn12 = m12
        fmn22 = m22

        # fmajor = pressure weight * fminor
        f111 = fp0 * m11
        f211 = fp0 * m21
        f121 = fpress * m11
        f221 = fpress * m21
        f112 = fp0 * m12
        f212 = fp0 * m22
        f122 = fpress * m12
        f222 = fpress * m22


def interp3d_major_1gpt(
    scaling1: FloatField,
    scaling2: FloatField,
    f111: FloatField,
    f211: FloatField,
    f121: FloatField,
    f221: FloatField,
    f112: FloatField,
    f212: FloatField,
    f122: FloatField,
    f222: FloatField,
    jtemp: IntField,
    jpress: IntField,
    jeta1: IntField,
    jeta2: IntField,
    kmajor: KMajor,
    res: FloatField,
):
    """Major-gas 8-point interpolation of the k table, for one g-point.

    Faithful port of the Fortran `interpolate3D_byflav` inner expression
    (kernels lines 777-787), evaluated at a single fixed g-point `igpt`
    (a compile-time external). This is the first 4-axis `GlobalTable` gather
    and the first use of integer index arithmetic (jtemp+1, jpress+1, jeta+1)
    as gather indices; how the 256 g-points become a field dimension is a
    separate stage.

    Weights (per cell), matching Fortran fmajor(eta-level, press-level,
    temp-level): f<a><b><c> = fmajor(a, b, c), a,b,c in {1,2}.
    `scaling1`, `scaling2` are the two col_mix brackets.

    Integer inputs are 0-based table indices of the LOWER bracket:
      `jtemp`  -- temperature   (upper bracket = jtemp+1),
      `jpress` -- pressure      (Fortran jpress-1; upper bracket = jpress+1),
      `jeta1`  -- eta for temperature bracket 1 (upper = jeta1+1),
      `jeta2`  -- eta for temperature bracket 2 (upper = jeta2+1).

    External: igpt (compile-time g-point index into kmajor axis 3).
    """
    from __externals__ import igpt

    with computation(PARALLEL), interval(...):
        jtp = jtemp + 1
        jpp = jpress + 1
        je1p = jeta1 + 1
        je2p = jeta2 + 1
        res = scaling1 * (
            f111 * kmajor.A[jtemp, jeta1, jpress, igpt]
            + f211 * kmajor.A[jtemp, je1p, jpress, igpt]
            + f121 * kmajor.A[jtemp, jeta1, jpp, igpt]
            + f221 * kmajor.A[jtemp, je1p, jpp, igpt]
        ) + scaling2 * (
            f112 * kmajor.A[jtp, jeta2, jpress, igpt]
            + f212 * kmajor.A[jtp, je2p, jpress, igpt]
            + f122 * kmajor.A[jtp, jeta2, jpp, igpt]
            + f222 * kmajor.A[jtp, je2p, jpp, igpt]
        )


def interp3d_major_gpt(
    scaling1: FloatField,
    scaling2: FloatField,
    f111: FloatField,
    f211: FloatField,
    f121: FloatField,
    f221: FloatField,
    f112: FloatField,
    f212: FloatField,
    f122: FloatField,
    f222: FloatField,
    jtemp: IntField,
    jpress: IntField,
    jeta1: IntField,
    jeta2: IntField,
    igpt: IntField,
    kmajor: KMajor,
    res: FloatField,
):
    """Major-gas 8-point interpolation of the k table, g-point by index field.

    Same expression as `interp3d_major_1gpt`, but the g-point is a runtime
    index field `igpt` (every cell holds the same g-point per call) instead of
    a compile-time external. The stencil therefore compiles once and is driven
    in a Python loop over g-points -- the caller sets `igpt` to g and copies
    `res` into the g-point data axis of `tau`, filling all NGPT.

    A single in-stencil g-point loop is not available in this gt4py.cartesian
    frontend (no `for ... in range()`; data-dim accesses are unrolled at
    compile time, so a runtime data-dim write index is unsupported). The
    loop-driven form reuses only the Stage-2 runtime-index `.A` gather, which
    is already validated.

    Integer inputs are the 0-based lower-bracket table indices (see
    `interp3d_major_1gpt`). `igpt` is the 0-based g-point index into kmajor's
    4th axis.
    """
    with computation(PARALLEL), interval(...):
        jtp = jtemp + 1
        jpp = jpress + 1
        je1p = jeta1 + 1
        je2p = jeta2 + 1
        res = scaling1 * (
            f111 * kmajor.A[jtemp, jeta1, jpress, igpt]
            + f211 * kmajor.A[jtemp, je1p, jpress, igpt]
            + f121 * kmajor.A[jtemp, jeta1, jpp, igpt]
            + f221 * kmajor.A[jtemp, je1p, jpp, igpt]
        ) + scaling2 * (
            f112 * kmajor.A[jtp, jeta2, jpress, igpt]
            + f212 * kmajor.A[jtp, je2p, jpress, igpt]
            + f122 * kmajor.A[jtp, jeta2, jpp, igpt]
            + f222 * kmajor.A[jtp, je2p, jpp, igpt]
        )

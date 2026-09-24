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

Stage 2 (here): the eta / binary-species part of `interpolation` for one
flavor -- gathers the reference volume-mixing-ratio table `vmr_ref` by the
runtime temperature/atmosphere indices to form the mixing ratio, then the
eta interpolation index `jeta` and fraction `feta`. This is the first
`GlobalTable` gather in the port: `table.A[i, j, k]` read by runtime integer
indices. The gas indices for the flavor are compile-time externals (the
flavor->gas bookkeeping is host-side numpy setup, not a stencil).
"""

from ndsl.dsl.gt4py import GlobalTable, PARALLEL, computation, floor, interval, log
from ndsl.dsl.typing import Float, FloatField, IntField

# vmr_ref reference table, Fortran layout (atmos_layer=2, absorber_ext=20,
# temperature=14) for the LW_G256 coefficient file. GlobalTable has no spatial
# axes -- only these data dims -- and is gathered read-only with `.A[i, j, k]`.
# The shape is fixed per coefficient file; assert it in the caller.
VmrRef = GlobalTable[(Float, (2, 20, 14))]


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

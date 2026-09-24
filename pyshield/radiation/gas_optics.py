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

Stage 1 (here): the temperature/pressure part of `interpolation`
(Fortran lines 107-122) -- per-cell index and fraction in the temperature
and pressure table axes, and the troposphere flag. Pure float math so the
GT4Py harness is proven before table gathers (GlobalTable) are introduced.
"""

from ndsl.dsl.gt4py import PARALLEL, computation, floor, interval, log
from ndsl.dsl.typing import FloatField


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

"""GT4Py port of the RRTMGP shortwave two-stream RTE solver.

This module reimplements, as NDSL/GT4Py stencils, the Fortran
`sw_solver_2stream` (and its helpers `sw_dif_and_source` and `adding`) from
`rte-kernels/mo_rte_solver_kernels.F90`. It is the shortwave counterpart of the
longwave no-scattering solver in `lw_solver.py`: pyRTE still runs the RTE solve
on the CPU (`sw_optics.rte.solve` in `rte_rrtmgp.py`, problem_type
`TWO_STREAM`), and this is the shortwave half of that solve.

Scope -- the "simple path" of `sw_solver_2stream`:
    do_broadband = TRUE   (spectrally integrated flux only)
    clear-sky             (tau/ssa/g are the gas-optics two-stream properties)
    has_dif_bc  = FALSE   (no diffuse incident BC; inc_flux_dif = 0)
Unlike the longwave clear-sky path, the shortwave clear-sky path genuinely
exercises the two-stream math: gas-only asymmetry g == 0, but the Rayleigh
single-scattering albedo ssa is nonzero, so Rdif/Tdif and the `adding` transport
all do real work.

Vertical layout (identical convention to `lw_solver.py`): every field lives on
ONE common K axis of length nlev = nlay + 1. Layer fields (`tau`, `ssa`, `g`,
`mu0`, and the per-layer cell properties `rdif`/`tdif`/`rdir`/`tdir`/`tnoscat`
and the direct-beam diffuse sources `src_up`/`src_dn`) occupy K = 0 .. nlay-1;
level/working fields (`flux_dir`, `albedo`, `src`, `denom`, `flux_up`,
`flux_dn`) span all of K = 0 .. nlay.

The solver is three sequential computations expressed as GT4Py `computation`
blocks within one stencil per orientation:

  1. cell properties (PARALLEL, pointwise per layer) -- the Meador-Weaver /
     Zdunkowski PIFM `sw_dif_and_source` coefficients: gamma1..4, k, Rdif, Tdif,
     Rdir, Tdir, Tnoscat, with the mu0 clamp (min_mu0) and the Rdir/Tdir
     energy-conservation clamps.
  2. direct-beam K-sweep -- transmit the collimated beam down the column
     (flux_dir), forming the diffuse sources src_up = Rdir*dir_inc,
     src_dn = Tdir*dir_inc, with the nighttime mask (mu0 <= 0 -> zero source).
  3. `adding` (Shonk-Hogan 2008) -- two DEPENDENT K-sweeps sharing the
     persistent working arrays albedo/src/denom: first from the surface upward
     building albedo(k)/src(k)/denom(k), then from the top downward building the
     diffuse flux_dn(k)/flux_up(k). This two-sweep, shared-working-array index
     pattern was proven bit-for-bit against a NumPy transcription of the Fortran
     `adding` (both orientations) before these stencils were written.

Which sweep is FORWARD and which is BACKWARD is fixed by `top_at_1` (is layer 1
the top of the atmosphere?). As in the longwave solver, that run-time constant
is handled with two stencils -- `sw_2stream_gpoint_top_at_1` and
`sw_2stream_gpoint_top_at_n` -- rather than a run-time branch, because a stencil
cannot reorder its FORWARD/BACKWARD computation blocks.

The 224 shortwave g-points are driven by a Python (host) loop around a
compile-once stencil (the gas-optics / longwave pattern): each call solves one
g-point and accumulates its up/down/direct flux into the broadband `inout`
fields. There is NO pi*weight intensity->flux scaling and NO Gauss-angle loop
(those are longwave only); broadband accumulation per g-point is
    broadband_up  += flux_up
    broadband_dn  += flux_dn + flux_dir   (adding() returns diffuse flux_dn only)
    broadband_dir += flux_dir
All cell math is kept in float64 (the Fortran working precision).
"""

from ndsl.dsl.gt4py import (
    BACKWARD,
    FORWARD,
    PARALLEL,
    computation,
    exp,
    interval,
    sqrt,
)
from ndsl.dsl.typing import FloatField


def sw_2stream_gpoint_top_at_1(
    tau: FloatField,
    ssa: FloatField,
    g: FloatField,
    mu0: FloatField,
    sfc_alb_dir: FloatField,
    sfc_alb_dif: FloatField,
    inc_flux_dir: FloatField,
    rdif: FloatField,
    tdif: FloatField,
    rdir: FloatField,
    tdir: FloatField,
    tnoscat: FloatField,
    src_up: FloatField,
    src_dn: FloatField,
    flux_dir: FloatField,
    albedo: FloatField,
    src: FloatField,
    denom: FloatField,
    flux_up: FloatField,
    flux_dn: FloatField,
    broadband_up: FloatField,
    broadband_dn: FloatField,
    broadband_dir: FloatField,
):
    """One g-point of `sw_solver_2stream`, top-of-atmosphere at K = 0.

    top_at_1 = TRUE: top level is K = 0, surface level is K = nlay. The direct
    beam and the `adding` flux (phase B) sweep FORWARD (K increasing); the
    `adding` albedo/src/denom build (phase A) sweeps BACKWARD from the surface.

    Externals: min_mu0 (= sqrt(epsilon)), min_k (= 1e4*epsilon), eps
    (= epsilon(float64)).
    """
    from __externals__ import eps, min_k, min_mu0

    # --- 1. cell properties, per layer (interval(0, -1)) ----------------------
    with computation(PARALLEL), interval(0, -1):
        mu0c = max(min_mu0, mu0)
        gamma1 = (8.0 - ssa * (5.0 + 3.0 * g)) * 0.25
        gamma2 = 3.0 * (ssa * (1.0 - g)) * 0.25
        k = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), min_k))
        exp_minusktau = exp(-tau * k)
        exp_minus2ktau = exp_minusktau * exp_minusktau
        rt_term = 1.0 / (
            k * (1.0 + exp_minus2ktau) + gamma1 * (1.0 - exp_minus2ktau)
        )
        rdif = rt_term * gamma2 * (1.0 - exp_minus2ktau)
        tdif = rt_term * 2.0 * k * exp_minusktau

        k_mu = k * mu0c
        omk = 1.0 - k_mu * k_mu
        abs_omk = omk if omk >= 0.0 else -omk
        rt2 = ssa * rt_term / omk if abs_omk >= eps else ssa * rt_term / eps

        gamma3 = (2.0 - 3.0 * mu0c * g) * 0.25
        gamma4 = 1.0 - gamma3
        alpha1 = gamma1 * gamma4 + gamma2 * gamma3
        alpha2 = gamma1 * gamma3 + gamma2 * gamma4
        k_gamma3 = k * gamma3
        k_gamma4 = k * gamma4
        tnoscat = exp(-tau / mu0c)

        rdir_u = rt2 * (
            (1.0 - k_mu) * (alpha2 + k_gamma3)
            - (1.0 + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau
            - 2.0 * (k_gamma3 - alpha2 * k_mu) * exp_minusktau * tnoscat
        )
        tdir_u = -rt2 * (
            (1.0 + k_mu) * (alpha1 + k_gamma4) * tnoscat
            - (1.0 - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * tnoscat
            - 2.0 * (k_gamma4 + alpha1 * k_mu) * exp_minusktau
        )
        rdir = max(0.0, min(rdir_u, 1.0 - tnoscat))
        tdir = max(0.0, min(tdir_u, 1.0 - tnoscat - rdir))

    # --- 2. direct beam: FORWARD, boundary at top level K = 0 -----------------
    with computation(FORWARD):
        with interval(0, 1):
            flux_dir = inc_flux_dir * mu0
            if mu0 > 0.0:
                src_up = rdir * flux_dir
                src_dn = tdir * flux_dir
            else:
                src_up = 0.0
                src_dn = 0.0
        with interval(1, None):
            flux_dir = tnoscat[0, 0, -1] * flux_dir[0, 0, -1]
            if mu0 > 0.0:
                src_up = rdir * flux_dir
                src_dn = tdir * flux_dir
            else:
                src_up = 0.0
                src_dn = 0.0

    # --- 3a. adding phase A (albedo/src/denom): BACKWARD from surface K = nlay -
    with computation(BACKWARD):
        with interval(-1, None):
            albedo = sfc_alb_dif
            src = flux_dir * sfc_alb_dir if mu0 > 0.0 else 0.0
        with interval(0, -1):
            denom = 1.0 / (1.0 - rdif * albedo[0, 0, 1])
            albedo = rdif + tdif * tdif * albedo[0, 0, 1] * denom
            src = src_up + tdif * denom * (
                src[0, 0, 1] + albedo[0, 0, 1] * src_dn
            )

    # --- 3b. adding phase B (fluxes): FORWARD from top K = 0 ------------------
    with computation(FORWARD):
        with interval(0, 1):
            flux_dn = 0.0
            flux_up = flux_dn * albedo + src
        with interval(1, None):
            flux_dn = (
                tdif[0, 0, -1] * flux_dn[0, 0, -1]
                + rdif[0, 0, -1] * src
                + src_dn[0, 0, -1]
            ) * denom[0, 0, -1]
            flux_up = flux_dn * albedo + src

    # --- 4. total + broadband accumulate --------------------------------------
    with computation(PARALLEL), interval(...):
        broadband_up = broadband_up + flux_up
        broadband_dn = broadband_dn + flux_dn + flux_dir
        broadband_dir = broadband_dir + flux_dir


def sw_2stream_gpoint_top_at_n(
    tau: FloatField,
    ssa: FloatField,
    g: FloatField,
    mu0: FloatField,
    sfc_alb_dir: FloatField,
    sfc_alb_dif: FloatField,
    inc_flux_dir: FloatField,
    rdif: FloatField,
    tdif: FloatField,
    rdir: FloatField,
    tdir: FloatField,
    tnoscat: FloatField,
    src_up: FloatField,
    src_dn: FloatField,
    flux_dir: FloatField,
    albedo: FloatField,
    src: FloatField,
    denom: FloatField,
    flux_up: FloatField,
    flux_dn: FloatField,
    broadband_up: FloatField,
    broadband_dn: FloatField,
    broadband_dir: FloatField,
):
    """One g-point of `sw_solver_2stream`, top-of-atmosphere at K = nlay.

    top_at_1 = FALSE: top level is K = nlay, surface level is K = 0. The direct
    beam and the `adding` flux (phase B) sweep BACKWARD (K decreasing); the
    `adding` albedo/src/denom build (phase A) sweeps FORWARD from the surface.
    `denom` is stored at the level index it produces (K = fortran-layer + 1), so
    phase B reads it at [0, 0, +1]; this is the orientation-specific `denom`
    convention proven in the NumPy prototype.

    See `sw_2stream_gpoint_top_at_1` for the externals contract.
    """
    from __externals__ import eps, min_k, min_mu0

    # --- 1. cell properties, per layer (interval(0, -1)) ----------------------
    with computation(PARALLEL), interval(0, -1):
        mu0c = max(min_mu0, mu0)
        gamma1 = (8.0 - ssa * (5.0 + 3.0 * g)) * 0.25
        gamma2 = 3.0 * (ssa * (1.0 - g)) * 0.25
        k = sqrt(max((gamma1 - gamma2) * (gamma1 + gamma2), min_k))
        exp_minusktau = exp(-tau * k)
        exp_minus2ktau = exp_minusktau * exp_minusktau
        rt_term = 1.0 / (
            k * (1.0 + exp_minus2ktau) + gamma1 * (1.0 - exp_minus2ktau)
        )
        rdif = rt_term * gamma2 * (1.0 - exp_minus2ktau)
        tdif = rt_term * 2.0 * k * exp_minusktau

        k_mu = k * mu0c
        omk = 1.0 - k_mu * k_mu
        abs_omk = omk if omk >= 0.0 else -omk
        rt2 = ssa * rt_term / omk if abs_omk >= eps else ssa * rt_term / eps

        gamma3 = (2.0 - 3.0 * mu0c * g) * 0.25
        gamma4 = 1.0 - gamma3
        alpha1 = gamma1 * gamma4 + gamma2 * gamma3
        alpha2 = gamma1 * gamma3 + gamma2 * gamma4
        k_gamma3 = k * gamma3
        k_gamma4 = k * gamma4
        tnoscat = exp(-tau / mu0c)

        rdir_u = rt2 * (
            (1.0 - k_mu) * (alpha2 + k_gamma3)
            - (1.0 + k_mu) * (alpha2 - k_gamma3) * exp_minus2ktau
            - 2.0 * (k_gamma3 - alpha2 * k_mu) * exp_minusktau * tnoscat
        )
        tdir_u = -rt2 * (
            (1.0 + k_mu) * (alpha1 + k_gamma4) * tnoscat
            - (1.0 - k_mu) * (alpha1 - k_gamma4) * exp_minus2ktau * tnoscat
            - 2.0 * (k_gamma4 + alpha1 * k_mu) * exp_minusktau
        )
        rdir = max(0.0, min(rdir_u, 1.0 - tnoscat))
        tdir = max(0.0, min(tdir_u, 1.0 - tnoscat - rdir))

    # --- 2. direct beam: BACKWARD, boundary at top level K = nlay -------------
    with computation(BACKWARD):
        with interval(-1, None):
            # top_layer = nlay-1: its mu0 is the layer just below this level.
            flux_dir = inc_flux_dir * mu0[0, 0, -1]
        with interval(0, -1):
            flux_dir = tnoscat * flux_dir[0, 0, 1]
            if mu0 > 0.0:
                src_up = rdir * flux_dir[0, 0, 1]
                src_dn = tdir * flux_dir[0, 0, 1]
            else:
                src_up = 0.0
                src_dn = 0.0

    # --- 3a. adding phase A (albedo/src/denom): FORWARD from surface K = 0 -----
    with computation(FORWARD):
        with interval(0, 1):
            albedo = sfc_alb_dif
            src = flux_dir * sfc_alb_dir if mu0 > 0.0 else 0.0
        with interval(1, None):
            denom = 1.0 / (1.0 - rdif[0, 0, -1] * albedo[0, 0, -1])
            albedo = rdif[0, 0, -1] + tdif[0, 0, -1] * tdif[0, 0, -1] * albedo[
                0, 0, -1
            ] * denom
            src = src_up[0, 0, -1] + tdif[0, 0, -1] * denom * (
                src[0, 0, -1] + albedo[0, 0, -1] * src_dn[0, 0, -1]
            )

    # --- 3b. adding phase B (fluxes): BACKWARD from top K = nlay ---------------
    with computation(BACKWARD):
        with interval(-1, None):
            flux_dn = 0.0
            flux_up = flux_dn * albedo + src
        with interval(0, -1):
            flux_dn = (
                tdif * flux_dn[0, 0, 1] + rdif * src + src_dn
            ) * denom[0, 0, 1]
            flux_up = flux_dn * albedo + src

    # --- 4. total + broadband accumulate --------------------------------------
    with computation(PARALLEL), interval(...):
        broadband_up = broadband_up + flux_up
        broadband_dn = broadband_dn + flux_dn + flux_dir
        broadband_dir = broadband_dir + flux_dir

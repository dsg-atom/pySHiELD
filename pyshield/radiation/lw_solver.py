"""GT4Py port of the RRTMGP longwave clear-sky no-scattering RTE solver.

This module reimplements, as NDSL/GT4Py stencils, the Fortran
`lw_solver_noscat` (and its helpers `lw_source_noscat`,
`lw_transport_noscat_dn`, `lw_transport_noscat_up`) from
`rte-kernels/mo_rte_solver_kernels.F90`. It is the next port target after the
gas-optics gather core (`gas_optics.py`): pyRTE still runs the RTE solve on the
CPU (`lw_optics.rte.solve` in `rte_rrtmgp.py`), and this is the longwave half of
that solve.

Scope -- the "simple path" of `lw_solver_noscat_oneangle`:
    do_broadband = TRUE   (spectrally integrated flux only)
    do_Jacobians = FALSE  (no surface-temperature Jacobian)
    do_rescaling = FALSE  (no Tang scattering rescaling)
    nmus = 1              (single Gauss-Jacobi-5 quadrature angle)
    clear-sky            (tau is the gas-optics absorption optical depth)
The harder paths (multi-angle quadrature, Jacobians, the two-stream solver, the
shortwave solvers) attach later; the field/stencil layout here is chosen so they
bolt on without rework.

The one non-trivial GT4Py mechanic is the staggered level<->layer K-recurrence.
`flux_up`/`flux_dn` live on the nlay+1 interfaces; `tau`/`trans`/`source` live on
the nlay layers; the sweeps read a layer field at a vertical offset from a level
computation. The port represents every field on ONE common K axis of length
nlev = nlay + 1: layer fields occupy K = 0 .. nlay-1 (the last interface slot is
unused for them), level fields occupy all of K = 0 .. nlay. The downward and
upward sweeps are then plain `computation(FORWARD)` / `computation(BACKWARD)`
recurrences that read the neighbor interface at `[0, 0, -1]` or `[0, 0, +1]` and
the layer source/trans one slot back. This index arithmetic was proven bit-for-
bit against a NumPy translation of the Fortran before the stencils were written.

Which sweep is FORWARD and which is BACKWARD is fixed by `top_at_1` (is layer 1
the top of the atmosphere?). That orientation is a run-time constant, so rather
than branch inside a stencil -- which cannot reorder its FORWARD/BACKWARD
computation blocks -- there are two stencils, `lw_noscat_gpoint_top_at_1` and
`lw_noscat_gpoint_top_at_n`, and the host compiles the one matching the run.

The 256 longwave g-points are driven by a Python (host) loop around a
compile-once stencil (the same pattern the gas-optics tau gather uses): each call
solves one g-point and accumulates its flux into the broadband `inout` fields.
A final `scale_broadband` stencil applies the pi * weight intensity->flux factor
once. All source-function math is kept in float64 (the Fortran working
precision), including the `tau_thresh` series-expansion branch.
"""

from ndsl.dsl.gt4py import BACKWARD, FORWARD, PARALLEL, computation, exp, interval
from ndsl.dsl.typing import FloatField


def lw_noscat_gpoint_top_at_1(
    tau: FloatField,
    lay_source: FloatField,
    lev_source: FloatField,
    sfc_emis: FloatField,
    sfc_src: FloatField,
    inc_flux: FloatField,
    trans: FloatField,
    source_dn: FloatField,
    source_up: FloatField,
    flux_dn: FloatField,
    flux_up: FloatField,
    broadband_up: FloatField,
    broadband_dn: FloatField,
):
    """One g-point of `lw_solver_noscat_oneangle`, top-of-atmosphere at K = 0.

    Orientation top_at_1 = TRUE: top interface is K = 0, surface is K = nlay.
    Downward transport sweeps FORWARD (K increasing), upward sweeps BACKWARD.

    Fields are on the common K axis of length nlev = nlay + 1. Layer inputs
    (`tau`, `lay_source`) are valid on K = 0 .. nlay-1; `lev_source` and the
    flux fields on all K. `sfc_emis`, `sfc_src`, `inc_flux` are per-column
    constants broadcast across K (only read at the boundary interface).
    `broadband_up` / `broadband_dn` are `inout` accumulators (zeroed by the host
    before the g-point loop, scaled by `scale_broadband` after it).

    Externals: D (quadrature secant), weight (quadrature weight), pi,
    tau_thresh (= sqrt(sqrt(epsilon(float64))), the series-expansion cutoff).
    """
    from __externals__ import D, pi, tau_thresh, weight

    # --- source function + transmissivity, per layer (interval(0, -1)) --------
    with computation(PARALLEL), interval(0, -1):
        tau_loc = tau * D
        trans = exp(-tau_loc)
        if tau_loc > tau_thresh:
            fact = (1.0 - trans) / tau_loc - trans
        else:
            fact = tau_loc * (0.5 + tau_loc * (-1.0 / 3.0 + tau_loc * (1.0 / 8.0)))
        # top_at_1: source_inc => source_dn (uses lev_source[+1]),
        #           source_dec => source_up (uses lev_source[0]).
        source_dn = (1.0 - trans) * lev_source[0, 0, 1] + 2.0 * fact * (
            lay_source - lev_source[0, 0, 1]
        )
        source_up = (1.0 - trans) * lev_source + 2.0 * fact * (lay_source - lev_source)

    # --- downward transport: FORWARD, boundary at the top interface K = 0 -----
    with computation(FORWARD):
        with interval(0, 1):
            flux_dn = inc_flux / (pi * weight)
        with interval(1, None):
            flux_dn = trans[0, 0, -1] * flux_dn[0, 0, -1] + source_dn[0, 0, -1]

    # --- surface + upward transport: BACKWARD, boundary at surface K = nlay ---
    with computation(BACKWARD):
        with interval(-1, None):
            flux_up = flux_dn * (1.0 - sfc_emis) + sfc_emis * sfc_src
        with interval(0, -1):
            flux_up = trans * flux_up[0, 0, 1] + source_up

    # --- accumulate intensity into the broadband fields -----------------------
    with computation(PARALLEL), interval(...):
        broadband_up = broadband_up + flux_up
        broadband_dn = broadband_dn + flux_dn


def lw_noscat_gpoint_top_at_n(
    tau: FloatField,
    lay_source: FloatField,
    lev_source: FloatField,
    sfc_emis: FloatField,
    sfc_src: FloatField,
    inc_flux: FloatField,
    trans: FloatField,
    source_dn: FloatField,
    source_up: FloatField,
    flux_dn: FloatField,
    flux_up: FloatField,
    broadband_up: FloatField,
    broadband_dn: FloatField,
):
    """One g-point of `lw_solver_noscat_oneangle`, top-of-atmosphere at K = nlay.

    Orientation top_at_1 = FALSE: top interface is K = nlay, surface is K = 0.
    Downward transport sweeps BACKWARD (K decreasing), upward sweeps FORWARD. The
    BACKWARD downward block must run before the FORWARD upward block because the
    surface boundary of the up sweep reads the just-computed surface `flux_dn`;
    the block order here encodes that (which is why the two orientations are
    separate stencils rather than one with a run-time branch).

    See `lw_noscat_gpoint_top_at_1` for the field/externals contract.
    """
    from __externals__ import D, pi, tau_thresh, weight

    # --- source function + transmissivity, per layer (interval(0, -1)) --------
    with computation(PARALLEL), interval(0, -1):
        tau_loc = tau * D
        trans = exp(-tau_loc)
        if tau_loc > tau_thresh:
            fact = (1.0 - trans) / tau_loc - trans
        else:
            fact = tau_loc * (0.5 + tau_loc * (-1.0 / 3.0 + tau_loc * (1.0 / 8.0)))
        # top_at_1 FALSE: source_inc => source_up (uses lev_source[+1]),
        #                 source_dec => source_dn (uses lev_source[0]).
        source_up = (1.0 - trans) * lev_source[0, 0, 1] + 2.0 * fact * (
            lay_source - lev_source[0, 0, 1]
        )
        source_dn = (1.0 - trans) * lev_source + 2.0 * fact * (lay_source - lev_source)

    # --- downward transport: BACKWARD, boundary at the top interface K = nlay -
    with computation(BACKWARD):
        with interval(-1, None):
            flux_dn = inc_flux / (pi * weight)
        with interval(0, -1):
            flux_dn = trans * flux_dn[0, 0, 1] + source_dn

    # --- surface + upward transport: FORWARD, boundary at surface K = 0 -------
    with computation(FORWARD):
        with interval(0, 1):
            flux_up = flux_dn * (1.0 - sfc_emis) + sfc_emis * sfc_src
        with interval(1, None):
            flux_up = trans[0, 0, -1] * flux_up[0, 0, -1] + source_up[0, 0, -1]

    # --- accumulate intensity into the broadband fields -----------------------
    with computation(PARALLEL), interval(...):
        broadband_up = broadband_up + flux_up
        broadband_dn = broadband_dn + flux_dn


def scale_broadband(
    broadband_up: FloatField,
    broadband_dn: FloatField,
):
    """Convert accumulated intensity to flux: multiply by pi * weight once.

    Mirrors the final `broadband_* = pi * weight * broadband_*` of
    `lw_solver_noscat_oneangle` (the g-point loop accumulates intensity; this
    applies the azimuthal-isotropy + quadrature-weight factor a single time).

    Externals: pi, weight.
    """
    from __externals__ import pi, weight

    with computation(PARALLEL), interval(...):
        broadband_up = broadband_up * (pi * weight)
        broadband_dn = broadband_dn * (pi * weight)

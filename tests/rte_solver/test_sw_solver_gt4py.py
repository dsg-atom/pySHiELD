"""Validate SWTwoStreamSolverGT4Py vs pyRTE's shortwave TWO_STREAM solve.

The solver port (`pyshield/radiation/sw_solver.py` + `sw_solver_gt4py.py`)
reimplements `sw_solver_2stream` (do_broadband, clear-sky, has_dif_bc=FALSE) as
GT4Py stencils -- the shortwave counterpart of `test_lw_solver_gt4py.py`. This
test runs it against pyRTE's own two-stream RTE solver on IDENTICAL inputs, so
the comparison isolates the solver: both consume the SAME pyRTE gas-optics
output (tau, ssa, g, toa_source) and the SAME mu0 / surface albedo.

Procedure (mirrors tests/rte_solver/test_lw_solver_gt4py.py):
  1. pyRTE GasOptics(SW_G224).compute on a 4x4 RFMIP subset, problem_type
     TWO_STREAM -> `sw_optics` (tau, ssa, g, toa_source).
  2. Set the two-stream boundary conditions on `sw_optics` the way the driver
     (`rte_rrtmgp.py` step_radiation) does: `sw_optics["mu0"]` (cosine solar
     zenith) and `sw_optics["surface_albedo"]`. These field names are THE
     Discover confirmation point -- read the installed pyRTE two-stream solve
     source to confirm it reads `mu0`, `toa_source`, and `surface_albedo`
     (and whether direct/diffuse albedo are separate names). We set the
     direct/diffuse variants too, harmlessly, in case they are.
  3. Reference = sw_optics.rte.solve(add_to_input=False).sw_flux_{up,down}
     (and sw_flux_dir if the driver exposes it).
  4. Fold tau/ssa/g/mu0/albedo/toa_source into the (nx, ny) stencil tile, run
     SWTwoStreamSolverGT4Py.solve, compare broadband flux to rtol 1e-10.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/rte_solver/test_sw_solver_gt4py.py -q
"""

import os

import numpy as np
import pytest

from ndsl.config import backend_python

from pyshield.radiation.sw_solver_gt4py import SWTwoStreamSolverGT4Py

SITES = [0, 25, 50, 75]
EXPTS = [0, 6, 12, 17]


def _sw_optics_and_atm():
    if not os.environ.get("XDG_CACHE_HOME"):
        pytest.skip("XDG_CACHE_HOME not set; pyRTE cannot cache the coeff file")
    try:
        from pyrte_rrtmgp import rte
        from pyrte_rrtmgp.examples import RFMIP_FILES, load_example_file
        from pyrte_rrtmgp.rrtmgp import (
            DEFAULT_GAS_MAPPING,
            GasOptics,
            GasOpticsFiles,
            create_default_mapping,
        )
        from pyrte_rrtmgp.tests.test_rfmip_clear_sky import RFMIP_GAS_MAPPING
    except ImportError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"pyRTE not importable: {exc}")

    atm = load_example_file(RFMIP_FILES.ATMOSPHERE).isel(site=SITES, expt=EXPTS)
    gm = {g: RFMIP_GAS_MAPPING[g] for g in DEFAULT_GAS_MAPPING if g in RFMIP_GAS_MAPPING}
    sw_optics = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224).compute(
        atm.copy(deep=True),
        problem_type=rte.OpticsTypes.TWO_STREAM,
        gas_name_map=gm,
        variable_mapping=create_default_mapping(),
        add_to_input=False,
    )
    return sw_optics, atm


def _tile(da, noncore, rest, nx, ny):
    """DataArray -> (nx, ny, *rest), folding the non-core column dims first."""
    arr = da.transpose(*noncore, *rest).values
    rest_shape = tuple(da.sizes[r] for r in rest)
    return np.ascontiguousarray(arr).reshape(nx, ny, *rest_shape)


def _col(da, noncore, nx, ny):
    """Per-column DataArray -> (nx, ny)."""
    arr = da.transpose(*noncore).values
    return np.ascontiguousarray(arr).reshape(nx, ny)


def test_sw_solver_gt4py():
    sw, atm = _sw_optics_and_atm()

    layer_dim = sw.mapping.get_dim("layer")
    level_dim = sw.mapping.get_dim("level")
    top_at_1 = bool(sw.attrs["top_at_1"])

    noncore = [d for d in sw["tau"].dims if d not in (layer_dim, level_dim, "gpt")]
    nz = int(sw.sizes[layer_dim])
    ngpt = int(sw.sizes["gpt"])
    ncol = int(np.prod([sw.sizes[d] for d in noncore]))
    nx, ny = len(SITES), len(EXPTS)
    assert ncol == nx * ny, (ncol, nx, ny)

    # --- two-stream boundary conditions, driver convention -------------------
    # mu0 (cosine solar zenith). RFMIP ships solar_zenith_angle in degrees.
    import xarray as xr

    sza = atm["solar_zenith_angle"]
    mu0_da = np.cos(np.deg2rad(sza))
    # broadcast mu0 over the gas-optics non-core column dims
    mu0_da = mu0_da.broadcast_like(sw[noncore[0]] if noncore else sw["tau"])
    for d in noncore:
        if d not in mu0_da.dims:
            mu0_da = mu0_da.expand_dims({d: sw[d]})
    mu0_da = mu0_da.transpose(*noncore)

    # surface albedo (direct == diffuse here, driver sets a single value)
    if "surface_albedo" in atm:
        alb_da = atm["surface_albedo"]
    else:
        alb_da = xr.full_like(sza, 0.06)
    for d in noncore:
        if d not in alb_da.dims:
            alb_da = alb_da.expand_dims({d: sw[d]})
    alb_da = alb_da.transpose(*noncore)

    sw["mu0"] = mu0_da
    sw["surface_albedo"] = alb_da
    # Set the direct/diffuse variants too in case the solve reads those names.
    sw["surface_albedo_direct"] = alb_da
    sw["surface_albedo_diffuse"] = alb_da

    # --- pyRTE reference solve ----------------------------------------------
    fluxes = sw.rte.solve(add_to_input=False)
    flux_lev = [d for d in fluxes["sw_flux_up"].dims if d not in noncore]
    assert len(flux_lev) == 1, fluxes["sw_flux_up"].dims
    ref_up = _tile(fluxes["sw_flux_up"], noncore, flux_lev, nx, ny)
    ref_dn = _tile(fluxes["sw_flux_down"], noncore, flux_lev, nx, ny)
    ref_dir = None
    for name in ("sw_flux_dir", "sw_flux_direct", "sw_flux_down_direct"):
        if name in fluxes:
            ref_dir = _tile(fluxes[name], noncore, flux_lev, nx, ny)
            break

    # --- GT4Py solver inputs (same tau/ssa/g/mu0/albedo/toa_source) ----------
    tau = _tile(sw["tau"], noncore, [layer_dim, "gpt"], nx, ny)
    ssa = _tile(sw["ssa"], noncore, [layer_dim, "gpt"], nx, ny)
    gg = _tile(sw["g"], noncore, [layer_dim, "gpt"], nx, ny)
    # toa_source follows total_solar_irradiance's dims (RFMIP: site-only), so it
    # can lack some non-core dims (e.g. expt). Broadcast it to the full noncore set
    # before tiling -- the same way pyRTE's own solve broadcasts it against the
    # atmosphere -- using the expand_dims pattern used for mu0/albedo above.
    toa_da = sw["toa_source"]
    for d in noncore:
        if d not in toa_da.dims:
            toa_da = toa_da.expand_dims({d: sw[d]})
    inc_dir = _tile(toa_da, noncore, ["gpt"], nx, ny)
    mu0 = _col(sw["mu0"], noncore, nx, ny)
    alb = _col(sw["surface_albedo"], noncore, nx, ny)
    alb_gpt = np.repeat(alb[:, :, None], ngpt, axis=2)

    solver = SWTwoStreamSolverGT4Py(
        nx=nx, ny=ny, nz=nz, ngpt=ngpt, backend=backend_python, top_at_1=top_at_1
    )
    bb_up, bb_dn, bb_dir = solver.solve(
        tau=tau,
        ssa=ssa,
        g=gg,
        mu0=mu0,
        sfc_alb_dir=alb_gpt,
        sfc_alb_dif=alb_gpt,
        inc_flux_dir=inc_dir,
    )

    np.testing.assert_allclose(bb_up, ref_up, rtol=1e-10, atol=1e-12, err_msg="sw_flux_up")
    np.testing.assert_allclose(bb_dn, ref_dn, rtol=1e-10, atol=1e-12, err_msg="sw_flux_down")
    if ref_dir is not None:
        np.testing.assert_allclose(
            bb_dir, ref_dir, rtol=1e-10, atol=1e-12, err_msg="sw_flux_dir"
        )

    # sanity: finite everywhere.
    assert np.all(np.isfinite(bb_up)) and np.all(np.isfinite(bb_dn))
    # Non-negativity holds only for DAYTIME columns. RFMIP's global sites include
    # nighttime (solar_zenith_angle > 90 deg => mu0 < 0), where RTE-RRTMGP sets the
    # top direct beam to inc_flux_dir*mu0 < 0 by design and propagates it downward;
    # pyRTE produces the identical negative, so the rtol-1e-10 allclose above
    # already confirms the port matches it. Check physical non-negativity (within
    # round-off) only where mu0 > 0.
    day = np.broadcast_to((mu0 > 0.0)[:, :, None], bb_up.shape)
    assert np.all(bb_up[day] >= -1e-6), bb_up[day].min()
    assert np.all(bb_dn[day] >= -1e-6), bb_dn[day].min()
    # downward total >= direct, and both >= 0
    assert np.all(bb_dn + 1e-9 >= bb_dir)

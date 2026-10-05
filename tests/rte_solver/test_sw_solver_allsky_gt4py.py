"""Validate SWTwoStreamSolverGT4Py vs pyRTE on ALL-SKY (gas+cloud) optics.

This is the all-sky companion to `test_sw_solver_gt4py.py`. That clear-sky test
drove the solver on gas-only optics, where the asymmetry parameter g is ~0. This
test closes that gap: it builds COMBINED gas+cloud optics so g != 0 (real cloud
asymmetry) and proves the SAME, already-validated `SWTwoStreamSolverGT4Py`
reproduces pyRTE's two-stream solve on those combined optics. There is NO new
solver -- the all-sky SW solve is the clear-sky solver fed cloud-laden optics.

Isolation discipline (same as the other ports): the COMBINED optics are built
entirely with pyRTE's own code -- pyRTE `GasOptics(SW_G224).compute`, pyRTE
`CloudOptics(SW_BND).compute`, and pyRTE's `cloud_props.rte.add_to(sw_optics)`
(NOT `CloudOpticsGT4Py`) -- so this test isolates the SOLVER from any
cloud-optics-port error. `CloudOpticsGT4Py` is validated separately in
`test_cloud_optics_gt4py.py`.

Procedure (mirrors test_sw_solver_gt4py.py + the cloud-state synthesis of
test_cloud_optics_gt4py.py):
  1. RFMIP 4x4 subset; synthesize a cloud state (water paths 10 g/m2, radii at
     the LUT midpoints) the way the RRTMGP all-sky example does.
  2. pyRTE GasOptics(SW_G224).compute(TWO_STREAM) -> sw_optics (gas tau/ssa/g,
     g ~ 0).
  3. pyRTE CloudOptics(SW_BND).compute(TWO_STREAM) -> by-band cloud optics;
     cloud_props.rte.add_to(sw_optics) mutates sw_optics in place -> COMBINED
     optics with real cloud asymmetry g != 0.
  4. Set mu0 / surface albedo / toa_source exactly as test_sw_solver_gt4py.py.
  5. Reference = sw_optics.rte.solve(add_to_input=False).sw_flux_{up,down}.
  6. Feed the SAME combined tau/ssa/g (+mu0/albedo/toa_source) into
     SWTwoStreamSolverGT4Py.solve; compare broadband up/down (and dir if
     exposed) at rtol 1e-10, atol 1e-12.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/rte_solver/test_sw_solver_allsky_gt4py.py -q
"""

import os

import numpy as np
import pytest

from ndsl.config import backend_python

from pyshield.radiation.sw_solver_gt4py import SWTwoStreamSolverGT4Py

SITES = [0, 25, 50, 75]
EXPTS = [0, 6, 12, 17]


def _env():
    if not os.environ.get("XDG_CACHE_HOME"):
        pytest.skip("XDG_CACHE_HOME not set; pyRTE cannot cache the coeff file")
    try:
        from pyrte_rrtmgp import rte
        from pyrte_rrtmgp.examples import RFMIP_FILES, load_example_file
        from pyrte_rrtmgp.rrtmgp import (
            DEFAULT_GAS_MAPPING,
            CloudOptics,
            GasOptics,
            create_default_mapping,
        )
        from pyrte_rrtmgp.rrtmgp_data_files import CloudOpticsFiles, GasOpticsFiles
        from pyrte_rrtmgp.tests.test_rfmip_clear_sky import RFMIP_GAS_MAPPING
    except ImportError as exc:  # pragma: no cover - environment guard
        pytest.skip(f"pyRTE not importable: {exc}")

    atm = load_example_file(RFMIP_FILES.ATMOSPHERE).isel(site=SITES, expt=EXPTS)
    gm = {g: RFMIP_GAS_MAPPING[g] for g in DEFAULT_GAS_MAPPING if g in RFMIP_GAS_MAPPING}
    return dict(
        atm=atm, gm=gm, rte=rte,
        CloudOptics=CloudOptics, GasOptics=GasOptics,
        CloudOpticsFiles=CloudOpticsFiles, GasOpticsFiles=GasOpticsFiles,
        make_mapping=create_default_mapping,
    )


def _coeff_bounds(CloudOptics, cloud_file_e):
    """LUT radius bounds straight from pyRTE's CloudOptics coeff dataset.

    Read directly from pyRTE (not CloudOpticsGT4Py) so cloud-state synthesis
    carries no dependency on the cloud-optics port -- this test isolates the
    SOLVER. The ice LUT is indexed by effective diameter (diamice_*); liquid by
    radius (radliq_*).
    """
    co = CloudOptics(cloud_optics_file=cloud_file_e)
    ds = None
    for attr in ("_ds", "_dataset", "_cloud_optics", "dataset"):
        cand = getattr(co, attr, None)
        if cand is not None and hasattr(cand, "sizes"):
            ds = cand
            break
    assert ds is not None, "could not find pyRTE CloudOptics coeff dataset"

    def _v(*names):
        for n in names:
            if n in ds:
                return float(ds[n].values)
        raise KeyError(names)

    return dict(
        radliq_lwr=_v("radliq_lwr"),
        radliq_upr=_v("radliq_upr"),
        radice_lwr=_v("diamice_lwr", "radice_lwr"),
        radice_upr=_v("diamice_upr", "radice_upr"),
    )


def _add_clouds(atm, mapping, bounds):
    """Synthesize a cloud state (lwp, iwp, rel, rei); mirrors the all-sky example.

    Cloud present in the middle third of layers, water paths 10 g/m2, radii at
    the LUT midpoint (safely in bounds). Identical to test_cloud_optics_gt4py.py.
    """
    import xarray as xr

    layer_dim = mapping.get_dim("layer")
    pres = atm[mapping.get_var("pres_layer")]
    nlay = pres.sizes[layer_dim]
    k = xr.DataArray(np.arange(nlay), dims=[layer_dim])
    cloud = (k >= nlay // 3) & (k < 2 * nlay // 3)

    rel_val = 0.5 * (bounds["radliq_lwr"] + bounds["radliq_upr"])
    rei_val = 0.5 * (bounds["radice_lwr"] + bounds["radice_upr"])
    zeros = xr.zeros_like(pres)
    atm["lwp"] = zeros + xr.where(cloud, 10.0, 0.0)
    atm["iwp"] = zeros + xr.where(cloud, 10.0, 0.0)
    atm["rel"] = zeros + xr.where(cloud, rel_val, 0.0)
    atm["rei"] = zeros + xr.where(cloud, rei_val, 0.0)
    return atm


def _tile(da, noncore, rest, nx, ny):
    """DataArray -> (nx, ny, *rest), folding the non-core column dims first."""
    arr = da.transpose(*noncore, *rest).values
    rest_shape = tuple(da.sizes[r] for r in rest)
    return np.ascontiguousarray(arr).reshape(nx, ny, *rest_shape)


def _col(da, noncore, nx, ny):
    """Per-column DataArray -> (nx, ny)."""
    arr = da.transpose(*noncore).values
    return np.ascontiguousarray(arr).reshape(nx, ny)


def test_sw_solver_allsky_gt4py():
    env = _env()
    atm, gm, rte = env["atm"], env["gm"], env["rte"]
    nx, ny = len(SITES), len(EXPTS)

    cloud_file_e = getattr(env["CloudOpticsFiles"], "SW_BND")
    gas_file_e = getattr(env["GasOpticsFiles"], "SW_G224")
    bounds = _coeff_bounds(env["CloudOptics"], cloud_file_e)

    # cloud-laden atmosphere (clouds do not change gas optics; they feed cloud
    # optics). mapping must resolve the pyRTE canonical cloud names.
    atm_full = atm.copy(deep=True)
    atm_full.mapping.set_mapping(env["make_mapping"]())
    _add_clouds(atm_full, atm_full.mapping, bounds)

    pt = rte.OpticsTypes.TWO_STREAM

    # --- gas optics (g ~ 0) --------------------------------------------------
    sw = env["GasOptics"](gas_optics_file=gas_file_e).compute(
        atm_full.copy(deep=True),
        problem_type=pt,
        gas_name_map=gm,
        variable_mapping=env["make_mapping"](),
        add_to_input=False,
    )

    layer_dim = sw.mapping.get_dim("layer")
    level_dim = sw.mapping.get_dim("level")
    top_at_1 = bool(sw.attrs["top_at_1"])
    noncore = [d for d in sw["tau"].dims if d not in (layer_dim, level_dim, "gpt")]
    nz = int(sw.sizes[layer_dim])
    ngpt = int(sw.sizes["gpt"])
    ncol = int(np.prod([sw.sizes[d] for d in noncore]))
    assert ncol == nx * ny, (ncol, nx, ny)

    # record gas-only g to confirm the cloud add_to really introduces asymmetry.
    g_gas_max = float(np.abs(sw["g"].values).max())

    # --- cloud optics + add_to -> COMBINED optics (g != 0) -------------------
    cloud_props = env["CloudOptics"](cloud_optics_file=cloud_file_e).compute(
        atm_full.copy(deep=True),
        problem_type=pt,
        variable_mapping=env["make_mapping"](),
        add_to_input=False,
    )
    cloud_props.rte.add_to(sw)  # sw now holds COMBINED gas+cloud optics in place

    g_all_max = float(np.abs(sw["g"].values).max())
    # the whole point of this test: real cloud asymmetry present now.
    assert g_all_max > 1e-3, (g_gas_max, g_all_max)

    # --- two-stream boundary conditions, driver convention -------------------
    import xarray as xr

    sza = atm["solar_zenith_angle"]
    mu0_da = np.cos(np.deg2rad(sza))
    mu0_da = mu0_da.broadcast_like(sw[noncore[0]] if noncore else sw["tau"])
    for d in noncore:
        if d not in mu0_da.dims:
            mu0_da = mu0_da.expand_dims({d: sw[d]})
    mu0_da = mu0_da.transpose(*noncore)

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
    sw["surface_albedo_direct"] = alb_da
    sw["surface_albedo_diffuse"] = alb_da

    # --- pyRTE reference solve on the COMBINED optics ------------------------
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

    # --- GT4Py solver inputs (SAME combined tau/ssa/g/mu0/albedo/toa) --------
    tau = _tile(sw["tau"], noncore, [layer_dim, "gpt"], nx, ny)
    ssa = _tile(sw["ssa"], noncore, [layer_dim, "gpt"], nx, ny)
    gg = _tile(sw["g"], noncore, [layer_dim, "gpt"], nx, ny)
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
    # Non-negativity holds only for DAYTIME columns (RFMIP has nighttime sites
    # with mu0 < 0, where RTE-RRTMGP by design propagates a negative top beam;
    # pyRTE produces the identical negative, so the allclose above already
    # confirms the match). Check physical non-negativity only where mu0 > 0.
    day = np.broadcast_to((mu0 > 0.0)[:, :, None], bb_up.shape)
    assert np.all(bb_up[day] >= -1e-6), bb_up[day].min()
    assert np.all(bb_dn[day] >= -1e-6), bb_dn[day].min()
    assert np.all(bb_dn + 1e-9 >= bb_dir)

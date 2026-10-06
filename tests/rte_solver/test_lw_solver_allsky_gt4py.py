"""Validate LWNoScatSolverGT4Py vs pyRTE on ALL-SKY (gas+cloud) LW optics.

This is the all-sky companion to `test_lw_solver_gt4py.py`. That clear-sky test
drove the no-scattering solver on gas-only absorption tau. This test closes the
gap to the cloudy case: it builds COMBINED gas+cloud ABSORPTION optics and proves
the SAME, already-validated `LWNoScatSolverGT4Py` reproduces pyRTE's longwave
solve on those combined optics. There is NO new solver and NO rescaling: the LW
path runs in ABSORPTION mode (1scl), so both gas optics and cloud optics produce
TAU ONLY (no ssa/g). pyRTE's cloud `add_to` increments tau only and leaves the
Planck sources untouched, so the all-sky LW solve is pure no-scattering on
cloud-absorption-augmented tau plus the unchanged gas Planck sources -- exactly
the simple path the clear-sky solver already implements. This mirrors the all-sky
SW task.

Isolation discipline (same as the other ports): the COMBINED optics are built
entirely with pyRTE's own code -- pyRTE `GasOptics(LW_G256).compute`, pyRTE
`CloudOptics(LW_BND).compute`, and pyRTE's `cloud_props.rte.add_to(lw_optics)`
(NOT `CloudOpticsGT4Py`) -- so this test isolates the SOLVER from any
cloud-optics-port error. `CloudOpticsGT4Py` is validated separately in
`test_cloud_optics_gt4py.py`.

Procedure (mirrors test_lw_solver_gt4py.py + the cloud-state synthesis of
test_sw_solver_allsky_gt4py.py):
  1. RFMIP 4x4 subset; synthesize a cloud state (water paths 10 g/m2, radii at
     the LUT midpoints) the way the RRTMGP all-sky example does.
  2. pyRTE GasOptics(LW_G256).compute(ABSORPTION) -> lw_optics (gas tau + Planck
     layer/level/surface sources). Set surface_emissivity = 1.0 (iemsflg=0).
  3. pyRTE CloudOptics(LW_BND).compute(ABSORPTION) -> by-band cloud TAU;
     cloud_props.rte.add_to(lw_optics) mutates lw_optics["tau"] in place ->
     COMBINED absorption tau. Sources are left unchanged (asserted).
  4. Reference = lw_optics.rte.solve(add_to_input=False).lw_flux_{up,down} on the
     combined optics.
  5. Feed the SAME combined tau + (unchanged) layer/level/surface sources +
     surface emissivity into LWNoScatSolverGT4Py.solve; compare broadband up/down
     at rtol 1e-10, atol 1e-12.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/rte_solver/test_lw_solver_allsky_gt4py.py -q
"""

import os

import numpy as np
import pytest

from ndsl.config import backend_python, backend_gpu

# GPU runs: set RTE_TEST_BACKEND=dace:gpu (or gt:gpu) on an A100 node to compile the
# stencils for the GPU; unset = the default CPU backend used for correctness.
backend_python = backend_gpu if os.environ.get("RTE_TEST_BACKEND") else backend_python

from pyshield.radiation.lw_solver_gt4py import (
    D_DEFAULT,
    WEIGHT_DEFAULT,
    LWNoScatSolverGT4Py,
)

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
    the LUT midpoint (safely in bounds). Identical to the SW all-sky test.
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


def test_lw_solver_allsky_gt4py():
    env = _env()
    atm, gm, rte = env["atm"], env["gm"], env["rte"]
    nx, ny = len(SITES), len(EXPTS)

    cloud_file_e = getattr(env["CloudOpticsFiles"], "LW_BND")
    gas_file_e = getattr(env["GasOpticsFiles"], "LW_G256")
    bounds = _coeff_bounds(env["CloudOptics"], cloud_file_e)

    # cloud-laden atmosphere (clouds do not change gas optics; they feed cloud
    # optics). mapping must resolve the pyRTE canonical cloud names.
    atm_full = atm.copy(deep=True)
    atm_full.mapping.set_mapping(env["make_mapping"]())
    _add_clouds(atm_full, atm_full.mapping, bounds)

    pt = rte.OpticsTypes.ABSORPTION

    # --- gas optics (tau + Planck sources) -----------------------------------
    lw = env["GasOptics"](gas_optics_file=gas_file_e).compute(
        atm_full.copy(deep=True),
        problem_type=pt,
        gas_name_map=gm,
        variable_mapping=env["make_mapping"](),
        add_to_input=False,
    )
    # iemsflg=0 : black surface everywhere (as the clear-sky LW test does).
    lw["surface_emissivity"] = lw["surface_source"] * 0.0 + 1.0

    layer_dim = lw.mapping.get_dim("layer")
    level_dim = lw.mapping.get_dim("level")
    top_at_1 = bool(lw.attrs["top_at_1"])
    noncore = [d for d in lw["tau"].dims if d not in (layer_dim, level_dim, "gpt")]
    nz = int(lw.sizes[layer_dim])
    ngpt = int(lw.sizes["gpt"])
    ncol = int(np.prod([lw.sizes[d] for d in noncore]))
    assert ncol == nx * ny, (ncol, nx, ny)

    # snapshot gas-only tau + sources to prove (a) clouds really add absorption,
    # (b) the cloud add_to leaves the Planck sources untouched (ABSORPTION =
    # tau-only, so NO rescaling / NO source perturbation).
    tau_gas = lw["tau"].values.copy()
    lay_before = lw["layer_source"].values.copy()
    lev_before = lw["level_source"].values.copy()
    sfc_before = lw["surface_source"].values.copy()

    # --- cloud optics + add_to -> COMBINED absorption tau --------------------
    cloud_props = env["CloudOptics"](cloud_optics_file=cloud_file_e).compute(
        atm_full.copy(deep=True),
        problem_type=pt,
        variable_mapping=env["make_mapping"](),
        add_to_input=False,
    )
    cloud_props.rte.add_to(lw)  # lw["tau"] now holds COMBINED gas+cloud tau

    tau_all = lw["tau"].values
    # the whole point of this test: clouds actually add absorption somewhere.
    assert np.any(tau_all > tau_gas + 1e-12), (
        float(tau_gas.max()), float(tau_all.max())
    )
    assert np.all(tau_all >= tau_gas - 1e-12)  # absorption only ever increases
    # ABSORPTION cloud add_to is tau-only: Planck sources are NOT perturbed.
    np.testing.assert_array_equal(lw["layer_source"].values, lay_before)
    np.testing.assert_array_equal(lw["level_source"].values, lev_before)
    np.testing.assert_array_equal(lw["surface_source"].values, sfc_before)

    # --- GT4Py solver inputs (SAME combined tau + unchanged sources) ---------
    tau = _tile(lw["tau"], noncore, [layer_dim, "gpt"], nx, ny)
    lay = _tile(lw["layer_source"], noncore, [layer_dim, "gpt"], nx, ny)
    lev = _tile(lw["level_source"], noncore, [level_dim, "gpt"], nx, ny)
    sfc = _tile(lw["surface_source"], noncore, ["gpt"], nx, ny)
    emis = _tile(lw["surface_emissivity"], noncore, ["gpt"], nx, ny)

    solver = LWNoScatSolverGT4Py(
        nx=nx, ny=ny, nz=nz, ngpt=ngpt, backend=backend_python, top_at_1=top_at_1,
        D=D_DEFAULT, weight=WEIGHT_DEFAULT,
    )
    bb_up, bb_dn = solver.solve(
        tau=tau, lay_source=lay, lev_source=lev, sfc_src=sfc, sfc_emis=emis,
    )

    # --- pyRTE reference solve on the COMBINED optics ------------------------
    fluxes = lw.rte.solve(add_to_input=False)
    flux_lev = [d for d in fluxes["lw_flux_up"].dims if d not in noncore]
    assert len(flux_lev) == 1, fluxes["lw_flux_up"].dims
    ref_up = _tile(fluxes["lw_flux_up"], noncore, flux_lev, nx, ny)
    ref_dn = _tile(fluxes["lw_flux_down"], noncore, flux_lev, nx, ny)

    np.testing.assert_allclose(bb_up, ref_up, rtol=1e-10, atol=1e-12, err_msg="lw_flux_up")
    np.testing.assert_allclose(bb_dn, ref_dn, rtol=1e-10, atol=1e-12, err_msg="lw_flux_down")

    # sanity: real, physical longwave flux.
    assert np.all(np.isfinite(bb_up)) and np.all(np.isfinite(bb_dn))
    assert np.all(bb_up >= 0.0) and np.all(bb_dn >= 0.0)

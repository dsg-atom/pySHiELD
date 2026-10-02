"""Validate CloudOpticsGT4Py vs pyRTE's CloudOptics.compute + rte.add_to.

The cloud-optics port (`pyshield/radiation/cloud_optics.py` +
`cloud_optics_gt4py.py`) reimplements pyRTE's own cloud optics -- the LUT branch
of the Fortran `ty_cloud_optics_rrtmgp%cloud_optics` -- as GT4Py stencils, plus
the by-band -> by-g-point optical-props increment (`rte.add_to`). This test runs
it against pyRTE on IDENTICAL cloud inputs so the comparison isolates the port:

  1. Build an RFMIP atmosphere subset (4 sites x 4 experiments -> a 4x4 tile) and
     synthesize a cloud state the way the RRTMGP all-sky example does: cloud in a
     band of layers, liquid/ice water paths of 10 g/m2, effective radii at the
     midpoint of each LUT range (so they are safely in bounds).
  2. pyRTE CloudOptics(SW_BND / LW_BND).compute -> reference by-band cloud optics
     (SW: tau/ssa/g; LW ABSORPTION: tau).
  3. pyRTE GasOptics(SW_G224 / LW_G256).compute -> gas optics; then
     cloud_props.rte.add_to(gas_optics) -> reference combined all-sky optics.
  4. CloudOpticsGT4Py.compute (compare per-band cloud optics) and .add_to
     (compare combined optics) to the pyRTE references, rtol ~1e-10.

DISCOVER CONFIRMATION POINTS (read the installed pyRTE to confirm; the port has
defensive fallbacks but these are the places a wrong guess would show up):
  * the variable names CloudOptics.compute reads for the cloud water/ice paths
    and effective radii (here: lwp, iwp, rel, rei via the atmosphere mapping);
  * the attribute holding the pyRTE CloudOptics coefficient dataset
    (CloudOpticsGT4Py._coeff_dataset tries _dataset / _cloud_optics / dataset);
  * the by-band -> by-g-point map source (here: the gas-optics file's
    bnd_limits_gpt, 1-based (pair, bnd)).

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/rte_solver/test_cloud_optics_gt4py.py -q
"""

import os

import numpy as np
import pytest

from ndsl.config import backend_python

from pyshield.radiation.cloud_optics_gt4py import CloudOpticsGT4Py

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


def _add_clouds(atm, mapping, bounds):
    """Add a synthetic cloud state (lwp, iwp, rel, rei) to the atmosphere.

    Mirrors the RRTMGP all-sky example: cloud present in the middle third of
    layers, water paths 10 g/m2, radii at the LUT midpoint (in bounds). The
    variable names are the pyRTE canonical cloud names; `mapping` must resolve
    them (see the module docstring's confirmation points).
    """
    import xarray as xr

    layer_dim = mapping.get_dim("layer")
    # a template per (col..., layer) field: reuse pres_layer's shape/coords.
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


def _bounds(env, cloud_file):
    co = CloudOpticsGT4Py(
        cloud_optics_file=cloud_file, nx=len(SITES), ny=len(EXPTS),
        nz=1, backend=backend_python,
    )
    return co, dict(
        radliq_lwr=co.radliq_lwr, radliq_upr=co.radliq_upr,
        radice_lwr=co.radice_lwr, radice_upr=co.radice_upr,
    )


def _tile(da, noncore, rest, nx, ny):
    arr = da.transpose(*noncore, *rest).values
    rest_shape = tuple(da.sizes[r] for r in rest)
    return np.ascontiguousarray(arr).reshape(nx, ny, *rest_shape)


def _gpt2band(gas_file, GasOptics):
    ds = GasOptics(gas_optics_file=gas_file)._dataset
    lims = ds["bnd_limits_gpt"].transpose("pair", "bnd").values
    return CloudOpticsGT4Py.gpt2band_from_limits(lims)


def _run(problem, cloud_file, gas_file):
    env = _env()
    atm, gm, rte = env["atm"], env["gm"], env["rte"]
    nx, ny = len(SITES), len(EXPTS)
    mapping = env["make_mapping"]()

    # probe nz / bounds, build the port, synthesize clouds.
    probe = atm.copy(deep=False)
    probe.mapping.set_mapping(env["make_mapping"]())
    layer_dim = probe.mapping.get_dim("layer")
    nz = int(probe.sizes[layer_dim])

    cloud_file_e = getattr(env["CloudOpticsFiles"], cloud_file)
    gas_file_e = getattr(env["GasOpticsFiles"], gas_file)
    co, bounds = _bounds(env, cloud_file_e)
    co = CloudOpticsGT4Py(
        cloud_optics_file=cloud_file_e, nx=nx, ny=ny, nz=nz, backend=backend_python
    )

    atm_full = atm.copy(deep=True)
    atm_full.mapping.set_mapping(env["make_mapping"]())
    _add_clouds(atm_full, atm_full.mapping, bounds)

    pt = (rte.OpticsTypes.TWO_STREAM if problem == "sw" else rte.OpticsTypes.ABSORPTION)

    # ---- reference: pyRTE cloud optics + add_to -----------------------------
    ref_cloud = env["CloudOptics"](cloud_optics_file=cloud_file_e).compute(
        atm_full.copy(deep=True), problem_type=pt,
        variable_mapping=env["make_mapping"](), add_to_input=False,
    )
    gas_ref = env["GasOptics"](gas_optics_file=gas_file_e).compute(
        atm_full.copy(deep=True), problem_type=pt,
        gas_name_map=gm, variable_mapping=env["make_mapping"](), add_to_input=False,
    )
    cloud_ref_for_add = env["CloudOptics"](cloud_optics_file=cloud_file_e).compute(
        atm_full.copy(deep=True), problem_type=pt,
        variable_mapping=env["make_mapping"](), add_to_input=False,
    )
    cloud_ref_for_add.rte.add_to(gas_ref)  # gas_ref now holds combined optics

    # ---- port: CloudOpticsGT4Py.compute + add_to ----------------------------
    mine_cloud = co.compute(
        atm_full.copy(deep=True), problem_type=pt,
        variable_mapping=env["make_mapping"](), add_to_input=False,
    )
    gas_mine = env["GasOptics"](gas_optics_file=gas_file_e).compute(
        atm_full.copy(deep=True), problem_type=pt,
        gas_name_map=gm, variable_mapping=env["make_mapping"](), add_to_input=False,
    )
    gpt2band = _gpt2band(gas_file_e, env["GasOptics"])
    co.add_to(gas_mine, mine_cloud, gpt2band)

    return mine_cloud, ref_cloud, gas_mine, gas_ref


def _assert_var(mine, ref, var, nx, ny, rtol=1e-10, atol=1e-14):
    layer = mine.mapping.get_dim("layer")
    noncore = [d for d in mine[var].dims if d != layer and mine.sizes[d] != 0]
    # align dims of ref to mine
    r = ref[var].transpose(*mine[var].dims)
    np.testing.assert_allclose(
        mine[var].values, r.values, rtol=rtol, atol=atol, err_msg=var
    )


def test_cloud_optics_gt4py_sw():
    nx, ny = len(SITES), len(EXPTS)
    mine_cloud, ref_cloud, gas_mine, gas_ref = _run("sw", "SW_BND", "SW_G224")

    # per-band cloud optics
    _assert_var(mine_cloud, ref_cloud, "tau", nx, ny)
    _assert_var(mine_cloud, ref_cloud, "ssa", nx, ny)
    _assert_var(mine_cloud, ref_cloud, "g", nx, ny)
    # combined gas+cloud optics after add_to
    _assert_var(gas_mine, gas_ref, "tau", nx, ny)
    _assert_var(gas_mine, gas_ref, "ssa", nx, ny)
    _assert_var(gas_mine, gas_ref, "g", nx, ny)

    assert np.all(np.isfinite(mine_cloud["tau"].values))
    assert np.all(mine_cloud["tau"].values >= 0.0)
    assert np.all((mine_cloud["ssa"].values >= 0.0) & (mine_cloud["ssa"].values <= 1.0 + 1e-12))
    assert np.any(mine_cloud["tau"].values > 0.0)  # clouds really present


def test_cloud_optics_gt4py_lw():
    nx, ny = len(SITES), len(EXPTS)
    mine_cloud, ref_cloud, gas_mine, gas_ref = _run("lw", "LW_BND", "LW_G256")

    # per-band cloud absorption optical depth
    _assert_var(mine_cloud, ref_cloud, "tau", nx, ny)
    # combined gas+cloud absorption after add_to
    _assert_var(gas_mine, gas_ref, "tau", nx, ny)

    assert np.all(np.isfinite(mine_cloud["tau"].values))
    assert np.all(mine_cloud["tau"].values >= 0.0)
    assert np.any(mine_cloud["tau"].values > 0.0)

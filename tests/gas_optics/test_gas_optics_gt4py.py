"""Validate GasOpticsGT4Py.compute vs pyRTE GasOptics.compute (class level).

The per-kernel gather tests (test_tau_full{,_sw}.py, test_sw.py, test_planck.py)
check each stencil in isolation against a compiled-Fortran reference. This test
checks the assembled `GasOpticsGT4Py` driver interface: it runs the class's
`.compute(...)` and pyRTE's `GasOptics.compute(...)` on the same RFMIP
atmosphere and asserts every overwritten output var matches.

`GasOpticsGT4Py` is overwrite-first, so its output is pyRTE's own dataset with
the gather vars replaced by the GT4Py stencils. The comparison therefore also
confirms the plumbing -- interpolate re-run, non-core column dims folded to the
stencil tile and back, vars re-inserted in the container's dim order -- not just
the numerics, which the kernel tests already pin.

Both shortwave (SW_G224, TWO_STREAM) and longwave (LW_G256, ABSORPTION) are
exercised. `toa_source` is now reproduced by the class from the solar-source
coefficient tables (mirroring pyRTE `compute_sources`), so it too is compared.
A 4x4 RFMIP subset (4 sites x 4 experiments) folds to a (4, 4) tile.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/gas_optics/test_gas_optics_gt4py.py -q
"""

import os

import numpy as np
import pytest

from ndsl.config import backend_python

from pyshield.radiation.gas_optics_gt4py import GasOpticsGT4Py

SITES = [0, 25, 50, 75]
EXPTS = [0, 6, 12, 17]


def _rfmip():
    if not os.environ.get("XDG_CACHE_HOME"):
        pytest.skip("XDG_CACHE_HOME not set; pyRTE cannot cache the coeff file")
    try:
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

    from pyrte_rrtmgp import rte

    atm = load_example_file(RFMIP_FILES.ATMOSPHERE).isel(site=SITES, expt=EXPTS)
    gm = {g: RFMIP_GAS_MAPPING[g] for g in DEFAULT_GAS_MAPPING if g in RFMIP_GAS_MAPPING}
    return dict(
        atm=atm, gm=gm, rte=rte,
        GasOptics=GasOptics, GasOpticsFiles=GasOpticsFiles,
        make_mapping=create_default_mapping,
    )


def _nz(atm, make_mapping):
    """Layer count for the subset (mapping-agnostic; restored after probe)."""
    probe = atm.copy(deep=False)
    probe.mapping.set_mapping(make_mapping())
    return int(probe.sizes[probe.mapping.get_dim("layer")])


# per-var tolerances, matched to the isolated gather tests
_TOL = {
    "tau": dict(rtol=1e-9, atol=1e-22),
    "ssa": dict(rtol=1e-8, atol=1e-25),
    "g": dict(rtol=0, atol=0),
    "toa_source": dict(rtol=1e-12, atol=1e-30),  # class-reproduced solar source
    "surface_source": dict(rtol=1e-10, atol=1e-22),
    "layer_source": dict(rtol=1e-10, atol=1e-22),
    "level_source": dict(rtol=1e-10, atol=1e-22),
    "surface_source_jacobian": dict(rtol=1e-9, atol=1e-22),
}


def _compare(mine, ref, vars):
    for var in vars:
        assert var in mine, f"missing {var} in GasOpticsGT4Py output"
        m = mine[var]
        r = ref[var].transpose(*m.dims)
        tol = _TOL.get(var, dict(rtol=1e-9, atol=1e-22))
        np.testing.assert_allclose(
            m.values, r.values, err_msg=f"var={var}", **tol
        )


def test_gas_optics_gt4py_sw():
    env = _rfmip()
    atm, gm, rte = env["atm"], env["gm"], env["rte"]
    nz = _nz(atm, env["make_mapping"])
    file = env["GasOpticsFiles"].SW_G224

    go = GasOpticsGT4Py(
        gas_optics_file=file, nx=len(SITES), ny=len(EXPTS), nz=nz, backend=backend_python
    )
    assert go.is_sw
    mine = go.compute(
        atm.copy(deep=True),
        problem_type=rte.OpticsTypes.TWO_STREAM,
        gas_name_map=gm,
        variable_mapping=env["make_mapping"](),
        add_to_input=False,
    )

    ref = env["GasOptics"](gas_optics_file=file).compute(
        atm.copy(deep=True),
        problem_type=rte.OpticsTypes.TWO_STREAM,
        gas_name_map=gm,
        variable_mapping=env["make_mapping"](),
        add_to_input=False,
    )

    _compare(mine, ref, ["tau", "ssa", "g", "toa_source"])

    # sanity: real, physical shortwave optics
    assert np.all(np.isfinite(mine["tau"].values))
    assert np.all(mine["tau"].values >= 0.0)
    assert np.all((mine["ssa"].values >= 0.0) & (mine["ssa"].values <= 1.0))
    assert np.all(mine["g"].values == 0.0)


def test_gas_optics_gt4py_lw():
    env = _rfmip()
    atm, gm, rte = env["atm"], env["gm"], env["rte"]
    nz = _nz(atm, env["make_mapping"])
    file = env["GasOpticsFiles"].LW_G256

    go = GasOpticsGT4Py(
        gas_optics_file=file, nx=len(SITES), ny=len(EXPTS), nz=nz, backend=backend_python
    )
    assert not go.is_sw
    mine = go.compute(
        atm.copy(deep=True),
        problem_type=rte.OpticsTypes.ABSORPTION,
        gas_name_map=gm,
        variable_mapping=env["make_mapping"](),
        add_to_input=False,
    )

    ref = env["GasOptics"](gas_optics_file=file).compute(
        atm.copy(deep=True),
        problem_type=rte.OpticsTypes.ABSORPTION,
        gas_name_map=gm,
        variable_mapping=env["make_mapping"](),
        add_to_input=False,
    )

    _compare(
        mine,
        ref,
        [
            "tau",
            "surface_source",
            "layer_source",
            "level_source",
            "surface_source_jacobian",
        ],
    )

    # sanity: real, positive longwave sources
    assert np.all(np.isfinite(mine["tau"].values))
    assert np.all(mine["tau"].values >= 0.0)
    assert np.any(mine["layer_source"].values > 0.0)
    assert np.any(mine["surface_source"].values > 0.0)

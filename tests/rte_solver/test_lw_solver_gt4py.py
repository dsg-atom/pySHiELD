"""Validate LWNoScatSolverGT4Py vs pyRTE's longwave ABSORPTION solve.

The solver port (`pyshield/radiation/lw_solver.py` +
`lw_solver_gt4py.py`) reimplements `lw_solver_noscat` (do_broadband, nmus=1, no
Jacobians, no rescaling, clear-sky) as GT4Py stencils. This test runs it against
pyRTE's own RTE solver on IDENTICAL inputs, so the comparison isolates the
solver: both consume the SAME pyRTE gas-optics output (tau + Planck sources), so
no gas-optics error is folded in.

Procedure (mirrors tests/gas_optics/test_gas_optics_gt4py.py):
  1. pyRTE GasOptics(LW_G256).compute on a 4x4 RFMIP subset -> `lw_optics`
     (tau, layer_source, level_source, surface_source, ...). Set
     surface_emissivity = 1.0 (iemsflg=0), incident flux = 0 (longwave).
  2. Fold lw_optics' per-column fields into the (nx, ny) stencil tile and run
     LWNoScatSolverGT4Py.solve -> broadband up/down flux.
  3. Reference = lw_optics.rte.solve(...).lw_flux_{up,down} (pyRTE solving its
     own gas optics).
  4. Compare broadband flux to rtol 1e-10.

Quadrature: nmus=1 defaults (D = 1/0.6096748751, weight = 1.0) and `top_at_1`
are read where possible from the pyRTE dataset / asserted against the solver's
defaults. If a future pyRTE changes the default LW quadrature, this test is where
the mismatch surfaces -- confirm the actual secant/weight from a pyRTE LW solve
on Discover.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/rte_solver/test_lw_solver_gt4py.py -q
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


def _lw_optics():
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
    lw_optics = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256).compute(
        atm.copy(deep=True),
        problem_type=rte.OpticsTypes.ABSORPTION,
        gas_name_map=gm,
        variable_mapping=create_default_mapping(),
        add_to_input=False,
    )
    # iemsflg=0 : black surface everywhere.
    lw_optics["surface_emissivity"] = lw_optics["surface_source"] * 0.0 + 1.0
    return lw_optics


def _tile(da, noncore, rest, nx, ny):
    """DataArray -> (nx, ny, *rest), folding the non-core column dims first."""
    arr = da.transpose(*noncore, *rest).values
    rest_shape = tuple(da.sizes[r] for r in rest)
    return np.ascontiguousarray(arr).reshape(nx, ny, *rest_shape)


def test_lw_solver_gt4py():
    lw = _lw_optics()

    layer_dim = lw.mapping.get_dim("layer")
    level_dim = lw.mapping.get_dim("level")
    top_at_1 = bool(lw.attrs["top_at_1"])

    noncore = [d for d in lw["tau"].dims if d not in (layer_dim, level_dim, "gpt")]
    nz = int(lw.sizes[layer_dim])
    ngpt = int(lw.sizes["gpt"])
    ncol = int(np.prod([lw.sizes[d] for d in noncore]))
    nx, ny = len(SITES), len(EXPTS)
    assert ncol == nx * ny, (ncol, nx, ny)

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

    # pyRTE reference: solve its own gas optics. Detect the flux level dim from
    # the output rather than assuming it matches lw.mapping's level name.
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

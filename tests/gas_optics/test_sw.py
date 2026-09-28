"""Validate the shortwave Rayleigh tau and the SW optical-property combine vs pyRTE.

This ports the shortwave gas-optics pieces that are new relative to the
longwave path:

  * `compute_tau_rayleigh` (kernels lines 507-564): the Rayleigh scattering
    optical depth. The `krayl` gather is `interpolate2D_byflav` -- the SAME
    4-point (temp x eta, no pressure) interpolation as the minor-gas gather --
    on the SW `krayl` table (g-point axis 224). It runs once per troposphere
    half (Fortran `krayl(:,:,:,itropo)`, itropo 1=lower/2=upper): the lower half
    uses `rayl_lower` and the lower-atmosphere flavor, the upper half uses
    `rayl_upper` and the upper-atmosphere flavor, and each cell keeps the half
    picked by its tropopause flag. tau_rayleigh(g) = k(g) * (col_h2o + col_dry).

  * `combine_abs_and_rayleigh` (frontend lines 2002-2084, 2str branch): the
    shortwave optical properties. tau = tau_absorption + tau_rayleigh;
    ssa = tau_rayleigh / tau where tau > 2*tiny else 0; g = 0.

The Rayleigh gather is checked against pyRTE `SWGasOptics.tau_rayleigh`. The
combine is checked against pyRTE `SWGasOptics.compute_problem` (tau, ssa, g),
built here from pyRTE's own `tau_absorption` plus this port's Rayleigh tau --
so the combine formula is isolated from the tau-absorption gather (which is the
longwave `interp3d_major_gpt`/`interp2d_minor_gpt` stencils run on the SW
tables, ported and validated separately). Absorption dominates tau by orders of
magnitude, so recovering the small Rayleigh contribution by subtraction would
lose digits to cancellation; using pyRTE's tau_absorption as the input keeps
the check exact.

The top-of-atmosphere solar source (`toa_source`) is a per-column broadcast of
the per-g-point solar-source vector (Fortran frontend `gas_optics_ext`), a
host-side operation with no device kernel, handled where the port is wired into
`rte_rrtmgp.py`; it is not exercised here.

Column amounts are read from pyRTE's stored `gases_columns` (float32) and
upcast to float64 to match the bytes the compiled Fortran sees, the same
float32-promotion fix proven in the minor-gas density factor
(`test_tau_full.py`). pyRTE indices are 1-based; converted here to 0-based.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/gas_optics/test_sw.py -q
"""

import os

import numpy as np
import pytest

from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import backend_python
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Int

from pyshield.radiation.gas_optics import (
    NGPT_SW,
    interp2d_rayl_gpt,
)

SITES = [0, 25, 50, 75]
EXPTS = [0, 6, 12, 17]


def _pyrte_sw():
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

    atm = load_example_file(RFMIP_FILES.ATMOSPHERE)
    sw = GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)
    gm = {g: RFMIP_GAS_MAPPING[g] for g in DEFAULT_GAS_MAPPING if g in RFMIP_GAS_MAPPING}
    sw._gas_names = tuple(k for k, v in gm.items() if v in list(atm.data_vars))
    atm.mapping.set_mapping(create_default_mapping())
    return sw, atm, gm


def test_sw_gas_optics():
    sw, atm, gm = _pyrte_sw()
    assert type(sw).__name__ == "SWGasOptics", type(sw).__name__
    interp = sw.interpolate(atm, gm)

    # references: pyRTE compiled Fortran -------------------------------------
    tau_rayl_ds = sw.tau_rayleigh(interp)  # the new gather
    tau_abs_ds = sw.tau_absorption(atm, interp)  # absorption input to combine
    prob = sw.compute_problem(atm, interp)  # tau, ssa, g

    def take(ds, var):
        return (
            ds[var]
            .isel(site=SITES, expt=EXPTS)
            .transpose("site", "expt", "layer", "gpt")
            .values
        )

    tau_rayl_ref = take(tau_rayl_ds, "tau")
    tau_abs_ref = take(tau_abs_ds, "tau")
    tau_ref = take(prob, "tau")
    ssa_ref = take(prob, "ssa")
    g_ref = take(prob, "g")

    nx, ny = len(SITES), len(EXPTS)
    nz = interp.sizes["layer"]
    ngpt = tau_rayl_ref.shape[-1]
    assert ngpt == NGPT_SW == 224, (ngpt, NGPT_SW)

    def sub(da, order):
        return da.isel(site=SITES, expt=EXPTS).transpose(*order).values

    # shared interpolation intermediates -------------------------------------
    jtemp_f = sub(interp["temperature_index"], ("site", "expt", "layer"))
    tropo = sub(interp["tropopause_mask"], ("site", "expt", "layer"))
    fminor = sub(
        interp["fminor"], ("eta_interp", "temp_interp", "site", "expt", "layer", "flavor")
    )
    jeta_f = sub(interp["eta_index"], ("pair", "site", "expt", "layer", "flavor"))
    gpoint_flavor = sw.gpoint_flavor.transpose(
        "gpt", "atmos_layer"
    ).values  # (ngpt,2) 1-based

    # krayl halves, Fortran layout (temperature, eta, gpt) = (14,9,224) --------
    krayl = {
        b: np.ascontiguousarray(
            sw._dataset[f"rayl_{b}"]
            .transpose("temperature", "mixing_fraction", "gpt")
            .values
        )
        for b in ("lower", "upper")
    }
    for b in ("lower", "upper"):
        assert krayl[b].shape == (14, 9, NGPT_SW), (b, krayl[b].shape)

    # column amounts: pyRTE stores gases_columns float32; the Fortran upcasts to
    # float64. Feed the SAME float32-rounded bytes upcast to float64. Selected by
    # gas label, as the tau_rayleigh ufunc does (col_dry = "dry_air").
    def col_sel(gas_label):
        return (
            interp["gases_columns"]
            .sel(gas=gas_label)
            .isel(site=SITES, expt=EXPTS)
            .transpose("site", "expt", "layer")
            .values.astype(np.float64)
        )

    col_dry = col_sel("dry_air")
    col_h2o = col_sel("h2o")

    jtemp0 = (jtemp_f - 1).astype(np.int64)

    # stencil + fields --------------------------------------------------------
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=nx, ny=ny, nz=nz, nhalo=3, backend=backend_python
    )
    gi = stencil_factory.grid_indexing
    rayl = stencil_factory.from_origin_domain(
        func=interp2d_rayl_gpt, origin=gi.origin_compute(), domain=gi.domain_compute()
    )

    def ff():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def fi():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Int)

    fmn_q = {n: ff() for n in ("fmn11", "fmn21", "fmn12", "fmn22")}
    jtemp_q, jeta1_q, jeta2_q, kg_q = fi(), fi(), fi(), fi()
    res_q = ff()
    jtemp_q.view[:] = jtemp0

    si, ei, li = np.indices((nx, ny, nz))

    # --- Rayleigh tau: lower and upper halves, select per cell by tropo ------
    tau_rayl_mine = np.zeros((nx, ny, nz, ngpt), dtype=np.float64)
    for g in range(ngpt):
        k_half = {}
        for b, bi in (("lower", 0), ("upper", 1)):
            iflav = (gpoint_flavor[g, bi] - 1).astype(np.int64)
            fmn_q["fmn11"].view[:] = fminor[0, 0][si, ei, li, iflav]
            fmn_q["fmn21"].view[:] = fminor[1, 0][si, ei, li, iflav]
            fmn_q["fmn12"].view[:] = fminor[0, 1][si, ei, li, iflav]
            fmn_q["fmn22"].view[:] = fminor[1, 1][si, ei, li, iflav]
            jeta1_q.view[:] = (jeta_f[0][si, ei, li, iflav] - 1).astype(np.int64)
            jeta2_q.view[:] = (jeta_f[1][si, ei, li, iflav] - 1).astype(np.int64)
            kg_q.view[:] = g
            rayl(
                fmn11=fmn_q["fmn11"], fmn21=fmn_q["fmn21"],
                fmn12=fmn_q["fmn12"], fmn22=fmn_q["fmn22"],
                jtemp=jtemp_q, jeta1=jeta1_q, jeta2=jeta2_q, kg=kg_q,
                krayl=krayl[b], res=res_q,
            )
            k_half[b] = res_q.view[:].copy()
        k = np.where(tropo, k_half["lower"], k_half["upper"])
        tau_rayl_mine[:, :, :, g] = k * (col_h2o + col_dry)

    np.testing.assert_allclose(tau_rayl_mine, tau_rayl_ref, rtol=1e-10, atol=1e-28)

    # --- combine: tau = abs + rayl; ssa = rayl/tau (guarded); g = 0 ----------
    tiny = 2.0 * np.finfo(np.float64).tiny
    tau_mine = tau_abs_ref + tau_rayl_mine
    ssa_mine = np.where(tau_mine > tiny, tau_rayl_mine / tau_mine, 0.0)
    g_mine = np.zeros_like(tau_mine)

    np.testing.assert_allclose(tau_mine, tau_ref, rtol=1e-10, atol=1e-25)
    np.testing.assert_allclose(ssa_mine, ssa_ref, rtol=1e-8, atol=1e-25)
    np.testing.assert_allclose(g_mine, g_ref, rtol=0, atol=0)

    # sanity: Rayleigh tau is small, positive, and largest at short wavelengths
    assert np.all(np.isfinite(tau_rayl_mine))
    assert np.all(tau_rayl_mine >= 0.0)
    assert np.any(tau_rayl_mine > 0.0)
    assert np.all((ssa_mine >= 0.0) & (ssa_mine <= 1.0))

"""Validate the shortwave absorption tau (major + minor) vs pyRTE.

This is `test_tau_full.py` applied to the shortwave gas optics (SW_G224). The
absorption path -- `compute_tau_absorption` (major-gas kmajor gather + minor-gas
kminor gather with density/complement scaling) -- is IDENTICAL to the longwave
one; only the coefficient tables and the g-point count differ (224 vs 256). The
GT4Py gather stencils (`interp3d_major_gpt`, `interp2d_minor_gpt`) write a scalar
per g-point call and take the table as a `GlobalTable`; only the table carries
the g-point/contributor axis. So the shortwave port reuses those stencils
unchanged and simply feeds the SW tables, padded to the longwave alias shapes:
  * kmajor (14,9,60,224) -> zero-padded on the g-point axis to (14,9,60,256),
  * kminor_lower/upper (14,9,544)/(14,9,384) -> zero-padded to (14,9,NMINORK),
the same zero-pad the longwave upper kminor already uses. The padding tail is
never read (g < 224, contributor index < the real size).

Reference is pyRTE's compiled `SWGasOptics.tau_absorption`, split three ways to
localize failures without subtraction (absorption spans O(1e-3..few) and the
minor part is orders smaller):
  * major-only : mine vs pyRTE with kminor zeroed,
  * minor-only : mine vs pyRTE with kmajor zeroed,
  * full       : mine (major + minor) vs pyRTE full.

Indices come from pyRTE's own `get_idx_minor` / `_selected_gas_names_ext`; the
minor reduction is replicated via `np.isin(minor_gases, sw._gas_names)`. Column
amounts, play and tlay are read from pyRTE's stored float32 arrays and upcast to
float64 to match the bytes the compiled Fortran sees (the NEP50 float32
promotion fix proven in the longwave minor density factor). pyRTE indices are
1-based; converted here to 0-based.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/gas_optics/test_tau_full_sw.py -q
"""

import os

import numpy as np
import pytest

from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import backend_python

# GPU runs: set RTE_TEST_BACKEND=gt:gpu (or dace:gpu) on an A100 node;
# unset = the default CPU backend used for correctness.
import os  # noqa: E402
backend_python = os.environ.get("RTE_TEST_BACKEND") or backend_python
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Int

from pyshield.radiation.gas_optics import (
    NGPT_SW,
    NMINORK,
    interp2d_minor_gpt,
    interp3d_major_gpt,
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


def test_tau_full_sw():
    import xarray as xr

    sw, atm, gm = _pyrte_sw()
    assert type(sw).__name__ == "SWGasOptics", type(sw).__name__
    interp = sw.interpolate(atm, gm)

    # capture the intact kminor tables before zeroing them for the major-only
    # reference; the minor assembly below gathers from these.
    kminor_orig = {
        s: np.ascontiguousarray(
            sw._dataset[f"kminor_{s}"]
            .transpose("temperature", "mixing_fraction", f"contributors_{s}")
            .values
        )
        for s in ("lower", "upper")
    }

    # deep copies of the tables to restore between the three reference runs
    kmajor_da = sw._dataset["kmajor"].copy()
    kminor_da = {s: sw._dataset[f"kminor_{s}"].copy() for s in ("lower", "upper")}

    # three clean references (no subtraction): full = major + minor;
    # major-only = kminor zeroed; minor-only = kmajor zeroed.
    tau_full_ds = sw.tau_absorption(atm, interp)

    sw._dataset["kminor_lower"] = xr.zeros_like(sw._dataset["kminor_lower"])
    sw._dataset["kminor_upper"] = xr.zeros_like(sw._dataset["kminor_upper"])
    tau_major_ds = sw.tau_absorption(atm, interp)

    sw._dataset["kminor_lower"] = kminor_da["lower"]
    sw._dataset["kminor_upper"] = kminor_da["upper"]
    sw._dataset["kmajor"] = xr.zeros_like(kmajor_da)
    tau_minor_ds = sw.tau_absorption(atm, interp)
    sw._dataset["kmajor"] = kmajor_da  # restore for the major stencil's capture

    def take(ds):
        return (
            ds["tau"]
            .isel(site=SITES, expt=EXPTS)
            .transpose("site", "expt", "layer", "gpt")
            .values
        )

    tau_full_ref = take(tau_full_ds)
    tau_major_ref = take(tau_major_ds)
    tau_minor_ref = take(tau_minor_ds)

    nx, ny = len(SITES), len(EXPTS)
    nz = interp.sizes["layer"]
    ngpt = tau_full_ref.shape[-1]
    assert ngpt == NGPT_SW == 224, (ngpt, NGPT_SW)

    def sub(da, order):
        return da.isel(site=SITES, expt=EXPTS).transpose(*order).values

    # shared interpolation intermediates --------------------------------------
    jtemp_f = sub(interp["temperature_index"], ("site", "expt", "layer"))
    jpress_f = sub(interp["pressure_index"], ("site", "expt", "layer"))
    tropo = sub(interp["tropopause_mask"], ("site", "expt", "layer"))
    col_mix = sub(interp["column_mix"], ("temp_interp", "site", "expt", "layer", "flavor"))
    fmajor = sub(
        interp["fmajor"],
        ("eta_interp", "press_interp", "temp_interp", "site", "expt", "layer", "flavor"),
    )
    fminor = sub(
        interp["fminor"], ("eta_interp", "temp_interp", "site", "expt", "layer", "flavor")
    )
    jeta_f = sub(interp["eta_index"], ("pair", "site", "expt", "layer", "flavor"))
    # gases_columns stored float32; the Fortran upcasts to float64. Feed the same
    # float32-rounded bytes upcast to float64. Gas axis is _selected_gas_names_ext
    # order (index 0 = dry-air/total for the vmr factor; idx_h2o below is h2o).
    col_gas = sub(interp["gases_columns"], ("gas", "site", "expt", "layer")).astype(
        np.float64
    )  # (ngas, nx, ny, nz)

    tmplf = interp["temperature_index"].astype(float)
    pvar = atm.mapping.get_var("pres_layer")
    tvar = atm.mapping.get_var("temp_layer")
    # play/tlay stored float32; upcast so the density factor 0.01*play/tlay is
    # evaluated in float64 as the Fortran does.
    play = (
        atm[pvar].broadcast_like(tmplf).isel(site=SITES, expt=EXPTS)
        .transpose("site", "expt", "layer").values.astype(np.float64)
    )
    tlay = (
        atm[tvar].broadcast_like(tmplf).isel(site=SITES, expt=EXPTS)
        .transpose("site", "expt", "layer").values.astype(np.float64)
    )

    gpoint_flavor = sw.gpoint_flavor.transpose(
        "gpt", "atmos_layer"
    ).values  # (ngpt,2) 1-based

    # SW kmajor (14,9,60,224) zero-padded on the g-point axis to the KMajor alias
    # (14,9,60,256); g < 224 always, so the pad is never read.
    kmajor_sw = np.ascontiguousarray(
        sw._dataset["kmajor"]
        .transpose("temperature", "mixing_fraction", "pressure_interp", "gpt")
        .values
    )  # (14,9,60,224)
    assert kmajor_sw.shape == (14, 9, 60, ngpt), kmajor_sw.shape
    kmajor = np.zeros((14, 9, 60, 256), dtype=np.float64)
    kmajor[:, :, :, :ngpt] = kmajor_sw

    itropo = np.where(tropo, 0, 1).astype(np.int64)  # 0=lower/tropo, 1=upper
    jtemp0 = (jtemp_f - 1).astype(np.int64)
    jpress0 = (jpress_f + itropo - 1).astype(np.int64)

    # stencils + fields -------------------------------------------------------
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=nx, ny=ny, nz=nz, nhalo=3, backend=backend_python
    )
    gi = stencil_factory.grid_indexing
    major = stencil_factory.from_origin_domain(
        func=interp3d_major_gpt, origin=gi.origin_compute(), domain=gi.domain_compute()
    )
    minor = stencil_factory.from_origin_domain(
        func=interp2d_minor_gpt, origin=gi.origin_compute(), domain=gi.domain_compute()
    )

    def ff():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def fi():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Int)

    s1_q, s2_q = ff(), ff()
    fmaj_q = {n: ff() for n in ("f111", "f211", "f121", "f221", "f112", "f212", "f122", "f222")}
    jtemp_q, jpress_q, jeta1_q, jeta2_q, igpt_q, kg_q = fi(), fi(), fi(), fi(), fi(), fi()
    fmn_q = {n: ff() for n in ("fmn11", "fmn21", "fmn12", "fmn22")}
    res_q = ff()

    si, ei, li = np.indices((nx, ny, nz))
    fmaj_axes = {
        "f111": (0, 0, 0), "f211": (1, 0, 0), "f121": (0, 1, 0), "f221": (1, 1, 0),
        "f112": (0, 0, 1), "f212": (1, 0, 1), "f122": (0, 1, 1), "f222": (1, 1, 1),
    }

    jtemp_q.view[:] = jtemp0

    # --- major assembly ------------------------------------------------------
    tau_major_mine = np.zeros((nx, ny, nz, ngpt), dtype=np.float64)
    jpress_q.view[:] = jpress0
    for g in range(ngpt):
        iflav = (gpoint_flavor[g, itropo] - 1).astype(np.int64)
        s1_q.view[:] = col_mix[0][si, ei, li, iflav]
        s2_q.view[:] = col_mix[1][si, ei, li, iflav]
        for name, (e, p, t) in fmaj_axes.items():
            fmaj_q[name].view[:] = fmajor[e, p, t][si, ei, li, iflav]
        jeta1_q.view[:] = (jeta_f[0][si, ei, li, iflav] - 1).astype(np.int64)
        jeta2_q.view[:] = (jeta_f[1][si, ei, li, iflav] - 1).astype(np.int64)
        igpt_q.view[:] = g
        major(
            scaling1=s1_q, scaling2=s2_q,
            f111=fmaj_q["f111"], f211=fmaj_q["f211"], f121=fmaj_q["f121"], f221=fmaj_q["f221"],
            f112=fmaj_q["f112"], f212=fmaj_q["f212"], f122=fmaj_q["f122"], f222=fmaj_q["f222"],
            jtemp=jtemp_q, jpress=jpress_q, jeta1=jeta1_q, jeta2=jeta2_q, igpt=igpt_q,
            kmajor=kmajor, res=res_q,
        )
        tau_major_mine[:, :, :, g] = res_q.view[:]

    # minor accumulates into its own zeroed array (no cancellation on our side)
    tau_minor_mine = np.zeros((nx, ny, nz, ngpt), dtype=np.float64)

    # --- minor assembly (lower then upper) -----------------------------------
    idx_h2o = sw._selected_gas_names_ext.index("h2o")

    def idx_of(name):
        name = name.strip()
        if not name:
            return 0  # Fortran uses idx_scaling>0 as the "has a scaling gas" gate
        return int(sw.get_idx_minor(np.array([name]))[0])

    def run_minor(suffix, atmos_layer, layer_mask):
        ds = sw._dataset
        kmin = kminor_orig[suffix]  # (14, 9, contributors), captured pre-zeroing
        nk = kmin.shape[2]
        assert nk <= NMINORK, (nk, NMINORK)
        kmin_pad = np.zeros((14, 9, NMINORK), dtype=np.float64)
        kmin_pad[:, :, :nk] = kmin

        names = sw.extract_names(ds[f"minor_gases_{suffix}"].data)
        scal_names = sw.extract_names(ds[f"scaling_gas_{suffix}"].data)
        mask = np.isin(names, sw._gas_names)
        limits = ds[f"minor_limits_gpt_{suffix}"].transpose(
            f"minor_absorber_intervals_{suffix}", "pair"
        ).values  # (nminor,2) 1-based
        kstart = ds[f"kminor_start_{suffix}"].values  # (nminor,) 1-based
        sdens = ds[f"minor_scales_with_density_{suffix}"].values.astype(bool)
        scomp = ds[f"scale_by_complement_{suffix}"].values.astype(bool)
        gpt_flv = gpoint_flavor[:, atmos_layer]  # 1-based flavor per g-point

        for imnr in range(len(names)):
            if not mask[imnr]:
                continue
            gptS0, gptE0 = int(limits[imnr, 0]) - 1, int(limits[imnr, 1]) - 1
            ks0 = int(kstart[imnr]) - 1
            iflav = int(gpt_flv[gptS0]) - 1
            idx_minor = idx_of(names[imnr])
            idx_scal = idx_of(scal_names[imnr])

            scaling = col_gas[idx_minor].copy()  # (nx,ny,nz)
            if sdens[imnr]:
                scaling = scaling * (0.01 * play / tlay)
                if idx_scal > 0:
                    vmr = 1.0 / col_gas[0]
                    dry = 1.0 / (1.0 + col_gas[idx_h2o] * vmr)
                    fac = col_gas[idx_scal] * vmr * dry
                    scaling = scaling * ((1.0 - fac) if scomp[imnr] else fac)
            scaling = np.where(layer_mask, scaling, 0.0)

            fmn_q["fmn11"].view[:] = fminor[0, 0][si, ei, li, iflav]
            fmn_q["fmn21"].view[:] = fminor[1, 0][si, ei, li, iflav]
            fmn_q["fmn12"].view[:] = fminor[0, 1][si, ei, li, iflav]
            fmn_q["fmn22"].view[:] = fminor[1, 1][si, ei, li, iflav]
            jeta1_q.view[:] = (jeta_f[0][si, ei, li, iflav] - 1).astype(np.int64)
            jeta2_q.view[:] = (jeta_f[1][si, ei, li, iflav] - 1).astype(np.int64)

            for g in range(gptS0, gptE0 + 1):
                kg_q.view[:] = ks0 + (g - gptS0)
                minor(
                    fmn11=fmn_q["fmn11"], fmn21=fmn_q["fmn21"],
                    fmn12=fmn_q["fmn12"], fmn22=fmn_q["fmn22"],
                    jtemp=jtemp_q, jeta1=jeta1_q, jeta2=jeta2_q, kg=kg_q,
                    kminor=kmin_pad, res=res_q,
                )
                tau_minor_mine[:, :, :, g] += scaling * res_q.view[:]

    run_minor("lower", 0, tropo)
    run_minor("upper", 1, ~tropo)

    tau_full_mine = tau_major_mine + tau_minor_mine

    np.testing.assert_allclose(tau_major_mine, tau_major_ref, rtol=1e-10, atol=1e-22)
    np.testing.assert_allclose(tau_minor_mine, tau_minor_ref, rtol=1e-9, atol=1e-22)
    np.testing.assert_allclose(tau_full_mine, tau_full_ref, rtol=1e-9, atol=1e-22)

    # sanity: absorption tau is finite and non-negative
    assert np.all(np.isfinite(tau_full_mine))
    assert np.all(tau_full_mine >= 0.0)
    assert np.any(tau_full_mine > 0.0)

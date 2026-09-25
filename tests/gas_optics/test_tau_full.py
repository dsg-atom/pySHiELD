"""Validate the full gas-optics absorption tau (major + minor) vs pyRTE.

This completes `compute_tau_absorption`: the major-gas assembly validated in
`test_tau_major.py` PLUS the minor-gas assembly (Fortran `gas_optical_depths_minor`,
kernels lines 404-502, called once for the lower and once for the upper
atmosphere). The minor path reuses the S4 `fminor` weights through the new
`interp2d_minor_gpt` stencil (4-point temp x eta gather of `kminor`, no pressure)
and adds the per-cell scaling: minor-gas column amount, an optional density
factor (0.01*P/T), and an optional complement/second-gas factor.

Reference is pyRTE's compiled RRTMGP Fortran `GasOptics.tau_absorption`. To
localize failures the test splits the check three ways, each against a clean
pyRTE reference that isolates one contribution by zeroing the OTHER table --
never by subtraction. The minor tau is O(1e-8) and sits inside a major tau of
O(1-40); recovering it as (full - major) would lose ~7 digits to float64
cancellation (noise floor eps*major/minor ~ 1e-7), so we zero kmajor to read
the minor contribution directly instead.
  * major-only : mine vs pyRTE with kminor zeroed,
  * minor-only : mine vs pyRTE with kmajor zeroed,
  * full       : mine (major + minor) vs pyRTE full.

The minor reduction (drop absorbers whose gas is not in `_gas_names`) is
replicated exactly via `np.isin(minor_gases, go._gas_names)`; kept absorbers use
the FULL kminor table and FULL 1-based `kminor_start`, since the coefficient
values are identical to pyRTE's reduced view. Indices come from pyRTE's own
`get_idx_minor` / `_selected_gas_names_ext`, so no index convention is guessed.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/gas_optics/test_tau_full.py -q
"""

import os

import numpy as np
import pytest

from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import backend_python
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Int

from pyshield.radiation.gas_optics import (
    NGPT,
    NMINORK,
    interp2d_minor_gpt,
    interp3d_major_gpt,
)

SITES = [0, 25, 50, 75]
EXPTS = [0, 6, 12, 17]


def _pyrte():
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
    go = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256)
    gm = {g: RFMIP_GAS_MAPPING[g] for g in DEFAULT_GAS_MAPPING if g in RFMIP_GAS_MAPPING}
    go._gas_names = tuple(k for k, v in gm.items() if v in list(atm.data_vars))
    atm.mapping.set_mapping(create_default_mapping())
    return go, atm, gm


def test_tau_full():
    import xarray as xr

    go, atm, gm = _pyrte()
    interp = go.interpolate(atm, gm)

    # capture the intact kminor tables before zeroing them for the major-only
    # reference; the minor assembly below gathers from these.
    kminor_orig = {
        s: np.ascontiguousarray(
            go._dataset[f"kminor_{s}"]
            .transpose("temperature", "mixing_fraction", f"contributors_{s}")
            .values
        )
        for s in ("lower", "upper")
    }

    # deep copies of the tables to restore between the three reference runs
    kmajor_da = go._dataset["kmajor"].copy()
    kminor_da = {s: go._dataset[f"kminor_{s}"].copy() for s in ("lower", "upper")}

    # three clean references (no subtraction): full = major + minor;
    # major-only = kminor zeroed; minor-only = kmajor zeroed.
    tau_full_ds = go.tau_absorption(atm, interp)

    go._dataset["kminor_lower"] = xr.zeros_like(go._dataset["kminor_lower"])
    go._dataset["kminor_upper"] = xr.zeros_like(go._dataset["kminor_upper"])
    tau_major_ds = go.tau_absorption(atm, interp)

    go._dataset["kminor_lower"] = kminor_da["lower"]
    go._dataset["kminor_upper"] = kminor_da["upper"]
    go._dataset["kmajor"] = xr.zeros_like(kmajor_da)
    tau_minor_ds = go.tau_absorption(atm, interp)
    go._dataset["kmajor"] = kmajor_da  # restore for the major stencil's capture

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
    assert ngpt == NGPT == 256

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
    # Fortran gas_optical_depths_minor scales by the gas COLUMN amounts. pyRTE's
    # tau_absorption does not read the stored interp["gases_columns"] (float32);
    # it recomputes them fresh in float64 via
    #   col_gas = self.get_gases_columns(atmosphere, gas_name_map)
    #                 .sel(gas=self._selected_gas_names_ext)
    # where gas_name_map is the same mapping passed to interpolate (our `gm`).
    # Reading the float32 copy leaves a ~1e-7 gap in the minor scaling; match the
    # float64 recompute (the major path already uses float64 column_mix, hence it
    # passed at 1e-10). The explicit .sel below fixes the gas order to
    # _selected_gas_names_ext: index 0 is the dry-air/total column used by the
    # vmr factor, index idx_h2o below is h2o.
    col_gas = (
        go.get_gases_columns(atm, gm)
        .sel(gas=go._selected_gas_names_ext)
        .isel(site=SITES, expt=EXPTS)
        .transpose("gas", "site", "expt", "layer")
        .values
    )  # (ngas, nx, ny, nz), float64

    tmplf = interp["temperature_index"].astype(float)
    pvar = atm.mapping.get_var("pres_layer")
    tvar = atm.mapping.get_var("temp_layer")
    play = (
        atm[pvar].broadcast_like(tmplf).isel(site=SITES, expt=EXPTS)
        .transpose("site", "expt", "layer").values
    )
    tlay = (
        atm[tvar].broadcast_like(tmplf).isel(site=SITES, expt=EXPTS)
        .transpose("site", "expt", "layer").values
    )

    gpoint_flavor = go.gpoint_flavor.transpose("gpt", "atmos_layer").values  # (ngpt,2) 1-based
    kmajor = np.ascontiguousarray(
        go._dataset["kmajor"]
        .transpose("temperature", "mixing_fraction", "pressure_interp", "gpt")
        .values
    )  # (14,9,60,256)

    itropo = np.where(tropo, 0, 1).astype(np.int64)  # 0=lower/tropo, 1=upper
    jtemp0 = (jtemp_f - 1).astype(np.int64)
    jpress0 = (jpress_f + itropo - 1).astype(np.int64)

    # stencils + fields -------------------------------------------------------
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=nx, ny=ny, nz=nz, nhalo=3, backend=backend_python
    )
    quantity_factory.add_data_dimensions({"gpt": NGPT})
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

    def tauq():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM, "gpt"], "", dtype=Float)

    si, ei, li = np.indices((nx, ny, nz))
    fmaj_axes = {
        "f111": (0, 0, 0), "f211": (1, 0, 0), "f121": (0, 1, 0), "f221": (1, 1, 0),
        "f112": (0, 0, 1), "f212": (1, 0, 1), "f122": (0, 1, 1), "f222": (1, 1, 1),
    }

    jtemp_q.view[:] = jtemp0

    # --- major assembly (validated in test_tau_major) ------------------------
    tau_major_q = tauq()
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
        tau_major_q.view[:, :, :, g] = res_q.view[:]

    # minor accumulates into its own zeroed field (no cancellation on our side)
    tau_minor_q = tauq()

    # --- minor assembly (lower then upper) -----------------------------------
    idx_h2o = go._selected_gas_names_ext.index("h2o")

    def idx_of(name):
        name = name.strip()
        if not name:
            return 0  # Fortran uses idx_scaling>0 as the "has a scaling gas" gate
        return int(go.get_idx_minor(np.array([name]))[0])

    def run_minor(suffix, atmos_layer, layer_mask):
        ds = go._dataset
        kmin = kminor_orig[suffix]  # (14, 9, contributors), captured pre-zeroing
        nk = kmin.shape[2]
        assert nk <= NMINORK, (nk, NMINORK)
        kmin_pad = np.zeros((14, 9, NMINORK), dtype=kmin.dtype)
        kmin_pad[:, :, :nk] = kmin

        names = go.extract_names(ds[f"minor_gases_{suffix}"].data)
        scal_names = go.extract_names(ds[f"scaling_gas_{suffix}"].data)
        mask = np.isin(names, go._gas_names)
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
                tau_minor_q.view[:, :, :, g] += scaling * res_q.view[:]

    run_minor("lower", 0, tropo)
    run_minor("upper", 1, ~tropo)

    tau_full_mine = tau_major_q.view[:] + tau_minor_q.view[:]

    np.testing.assert_allclose(tau_major_q.view[:], tau_major_ref, rtol=1e-10, atol=1e-22)
    np.testing.assert_allclose(tau_minor_q.view[:], tau_minor_ref, rtol=1e-9, atol=1e-22)
    np.testing.assert_allclose(tau_full_mine, tau_full_ref, rtol=1e-9, atol=1e-22)

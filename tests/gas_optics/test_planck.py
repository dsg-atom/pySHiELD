"""Validate the LW Planck sources (compute_Planck_source) vs pyRTE.

This ports the Fortran `compute_Planck_source` (kernels lines 567-710), which
produces the four longwave source terms the RTE solver needs:
  * surface_source            sfc_src      (ncol, ngpt)
  * layer_source              lay_src      (ncol, nlay,   ngpt)
  * level_source              lev_src      (ncol, nlay+1, ngpt)
  * surface_source_jacobian   sfc_src_jac  (ncol, ngpt)

Two device operations feed them:
  * `pfrac` -- the per-g-point fraction of each band's Planck function -- is
    `interpolate3D_byflav(one, fmajor, pfracin, ...)`, i.e. the SAME 8-point
    gather as the major-gas tau (`interp3d_major_gpt`) with the two col_mix
    scalings set to 1 and the `plank_fraction` table (same (14,9,60,256) layout
    as kmajor) in place of kmajor. Reused verbatim here.
  * `planck_interp1d` (new) -- the 1-D interpolation of `totplnk` in temperature
    to a per-band Planck value (Fortran `interpolate1D`). Evaluated at the
    surface (tsfc, and tsfc+1 for the Jacobian), each layer center (tlay), and
    each level/interface (tlev). The temperature index and fraction are prepared
    on the host (as jtemp/jpress were in the tau tests; the float part is the
    same floor+clamp interp_tp already runs on-device), so the stencil is the
    totplnk gather plus the linear blend.

The assembly is elementwise, matching the Fortran:
  sfc_src(g)     = pfrac(sfc_lay, g) * planck(tsfc, band(g))
  sfc_src_jac(g) = pfrac(sfc_lay, g) * (planck(tsfc+1, band) - planck(tsfc, band))
  lay_src(k, g)  = pfrac(k, g)       * planck(tlay(k), band(g))
  lev_src(0, g)      = pfrac(0, g)      * planck(tlev(0), band)
  lev_src(k, g)      = sqrt(pfrac(k-1,g)*pfrac(k,g)) * planck(tlev(k), band)  (interior)
  lev_src(nlay, g)   = pfrac(nlay-1, g) * planck(tlev(nlay), band)

Reference is pyRTE's compiled RRTMGP Fortran via `GasOptics.compute_planck`.
The interpolation intermediates come from pyRTE's own `GasOptics.interpolate`,
so this isolates the Planck assembly from the interpolation stencils.

pyRTE indices are 1-based (Fortran); converted here to 0-based lower brackets:
  jtemp0  = jtemp_fortran - 1
  jpress0 = jpress_fortran + itropo - 1   (itropo 0=lower/tropo, 1=upper)
  jeta0   = jeta_fortran - 1
  sfc_lay0 = nlay-1 if top_at_1 else 0    (Fortran sfc_lay = nlay or 1)

Temperatures are upcast to float64 before the index math: they are stored
float32 and the Fortran evaluates val0 = (T - temp_ref_min)/totplnk_delta in
float64 (real(wp)); keeping the arithmetic in float32 would drift ~1 ulp (the
same trap fixed in the minor-gas density factor).

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/gas_optics/test_planck.py -q
"""

import os

import numpy as np
import pytest

from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import backend_python
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Int

from pyshield.radiation.gas_optics import (
    NBND,
    NGPT,
    NPLANCKTEMP,
    interp3d_major_gpt,
    planck_interp1d,
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


def _idx_frac(temp, temp_ref_min, totplnk_delta, nplancktemp):
    """Host side of Fortran `interpolate1D`: temperature -> (idx0, frac).

    val0 = (T - temp_ref_min)/totplnk_delta; iv = int(val0) (toward zero, as
    Fortran `int()`, == floor for the in-range positive val0 here). The 1-based
    Fortran index is min(N-1, max(1, iv+1)); the 0-based lower index is
    clip(iv, 0, N-2). `frac` uses the un-clamped iv, matching the Fortran.
    """
    val0 = (temp - temp_ref_min) / totplnk_delta
    iv = val0.astype(np.int64)  # trunc toward zero, as Fortran int()
    frac = val0 - iv
    idx0 = np.clip(iv, 0, nplancktemp - 2).astype(np.int64)
    return idx0, frac


def test_planck():
    go, atm, gm = _pyrte()
    interp = go.interpolate(atm, gm)

    # reference: pyRTE's compiled compute_Planck_source ------------------------
    ref = go.compute_planck(atm, interp)

    layer_dim = atm.mapping.get_dim("layer")
    level_dim = atm.mapping.get_dim("level")
    nz = interp.sizes[layer_dim]
    nlev = atm.sizes[level_dim]
    assert nlev == nz + 1, (nlev, nz)
    nx, ny = len(SITES), len(EXPTS)

    def take(da, extra):
        return (
            da.isel(site=SITES, expt=EXPTS)
            .transpose("site", "expt", *extra)
            .values
        )

    sfc_ref = take(ref["surface_source"], ["gpt"])  # (nx,ny,ngpt)
    lay_ref = take(ref["layer_source"], [layer_dim, "gpt"])  # (nx,ny,nz,ngpt)
    lev_ref = take(ref["level_source"], [level_dim, "gpt"])  # (nx,ny,nz+1,ngpt)
    jac_ref = take(ref["surface_source_jacobian"], ["gpt"])  # (nx,ny,ngpt)

    ngpt = sfc_ref.shape[-1]
    assert ngpt == NGPT == 256, (ngpt, NGPT)

    def sub(da, order):
        return da.isel(site=SITES, expt=EXPTS).transpose(*order).values

    # shared interpolation intermediates (same as the tau tests) --------------
    jtemp_f = sub(interp["temperature_index"], ("site", "expt", "layer"))
    jpress_f = sub(interp["pressure_index"], ("site", "expt", "layer"))
    tropo = sub(interp["tropopause_mask"], ("site", "expt", "layer"))
    fmajor = sub(
        interp["fmajor"],
        ("eta_interp", "press_interp", "temp_interp", "site", "expt", "layer", "flavor"),
    )
    jeta_f = sub(interp["eta_index"], ("pair", "site", "expt", "layer", "flavor"))

    gpoint_flavor = go.gpoint_flavor.transpose("gpt", "atmos_layer").values  # (ngpt,2) 1-based
    band_lims = go._dataset["bnd_limits_gpt"].transpose("pair", "bnd").values  # (2,nbnd) 1-based
    nbnd = band_lims.shape[1]
    assert nbnd == NBND, (nbnd, NBND)

    # plank_fraction: same (temperature,eta,pressure,gpt) layout as kmajor
    pfracin = np.ascontiguousarray(
        go._dataset["plank_fraction"]
        .transpose("temperature", "mixing_fraction", "pressure_interp", "gpt")
        .values
    )
    assert pfracin.shape == (14, 9, 60, 256), pfracin.shape

    # totplnk table + interpolation constants ---------------------------------
    nplancktemp = go._dataset.sizes["temperature_Planck"]
    assert nplancktemp == NPLANCKTEMP, (nplancktemp, NPLANCKTEMP)
    totplnk = np.ascontiguousarray(
        go._dataset["totplnk"].transpose("temperature_Planck", "bnd").values
    )
    assert totplnk.shape == (NPLANCKTEMP, NBND), totplnk.shape
    temp_ref_min = float(go._dataset["temp_ref"].min())
    temp_ref_max = float(go._dataset["temp_ref"].max())
    totplnk_delta = (temp_ref_max - temp_ref_min) / (nplancktemp - 1)

    # temperatures (upcast to float64 to match the Fortran density-free interp)
    tmplf = interp["temperature_index"].astype(float)
    tlay_var = atm.mapping.get_var("temp_layer")
    tlev_var = atm.mapping.get_var("temp_level")
    tsfc_var = atm.mapping.get_var("surface_temperature")
    tlay = (
        atm[tlay_var].broadcast_like(tmplf).isel(site=SITES, expt=EXPTS)
        .transpose("site", "expt", "layer").values.astype(np.float64)
    )  # (nx,ny,nz)
    tlev = (
        atm[tlev_var].isel(site=SITES, expt=EXPTS)
        .transpose("site", "expt", level_dim).values.astype(np.float64)
    )  # (nx,ny,nz+1)
    tsfc = (
        atm[tsfc_var].isel(site=SITES, expt=EXPTS)
        .transpose("site", "expt").values.astype(np.float64)
    )  # (nx,ny)

    # top orientation -> surface layer index (Fortran sfc_lay, converted 0-based)
    pres_layer_var = atm.mapping.get_var("pres_layer")
    top_at_1 = bool(
        atm[pres_layer_var].values[0, 0] < atm[pres_layer_var].values[0, -1]
    )
    sfc_lay0 = (nz - 1) if top_at_1 else 0

    # index conversions -------------------------------------------------------
    itropo = np.where(tropo, 0, 1).astype(np.int64)  # 0=lower/tropo, 1=upper
    jtemp0 = (jtemp_f - 1).astype(np.int64)
    jpress0 = (jpress_f + itropo - 1).astype(np.int64)

    # ------------------------------------------------------------------------
    # pfrac: reuse the major-gas kmajor gather with scaling=1 and pfracin
    # ------------------------------------------------------------------------
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=nx, ny=ny, nz=nz, nhalo=3, backend=backend_python
    )
    gi = stencil_factory.grid_indexing
    major = stencil_factory.from_origin_domain(
        func=interp3d_major_gpt, origin=gi.origin_compute(), domain=gi.domain_compute()
    )
    planck_lay_sten = stencil_factory.from_origin_domain(
        func=planck_interp1d, origin=gi.origin_compute(), domain=gi.domain_compute()
    )

    def ff():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def fi():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Int)

    s1_q, s2_q = ff(), ff()
    s1_q.view[:] = 1.0
    s2_q.view[:] = 1.0
    fmaj_q = {n: ff() for n in ("f111", "f211", "f121", "f221", "f112", "f212", "f122", "f222")}
    jtemp_q, jpress_q, jeta1_q, jeta2_q, igpt_q = fi(), fi(), fi(), fi(), fi()
    res_q = ff()
    jtemp_q.view[:] = jtemp0
    jpress_q.view[:] = jpress0

    fmaj_axes = {
        "f111": (0, 0, 0), "f211": (1, 0, 0), "f121": (0, 1, 0), "f221": (1, 1, 0),
        "f112": (0, 0, 1), "f212": (1, 0, 1), "f122": (0, 1, 1), "f222": (1, 1, 1),
    }
    si, ei, li = np.indices((nx, ny, nz))

    pfrac = np.zeros((nx, ny, nz, ngpt), dtype=np.float64)
    for g in range(ngpt):
        iflav = (gpoint_flavor[g, itropo] - 1).astype(np.int64)
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
            kmajor=pfracin, res=res_q,
        )
        pfrac[:, :, :, g] = res_q.view[:]

    # ------------------------------------------------------------------------
    # planck values via the totplnk 1-D interp (host idx/frac, stencil gather)
    # ------------------------------------------------------------------------
    # layer- and surface-sized planck use the (nx,ny,nz) factory; tsfc is
    # broadcast over K and read back at k=0.
    tsfc_b = np.broadcast_to(tsfc[:, :, None], (nx, ny, nz))
    idx_lay, frac_lay = _idx_frac(tlay, temp_ref_min, totplnk_delta, nplancktemp)
    idx_sfc, frac_sfc = _idx_frac(tsfc_b, temp_ref_min, totplnk_delta, nplancktemp)
    idx_sfcd, frac_sfcd = _idx_frac(
        tsfc_b + 1.0, temp_ref_min, totplnk_delta, nplancktemp
    )

    frac_lay_q, idx_lay_q = ff(), fi()
    frac_sfc_q, idx_sfc_q = ff(), fi()
    frac_sfcd_q, idx_sfcd_q = ff(), fi()
    iband_q = fi()
    planck_lay_q, planck_sfc_q, planck_sfcd_q = ff(), ff(), ff()
    frac_lay_q.view[:] = frac_lay
    idx_lay_q.view[:] = idx_lay
    frac_sfc_q.view[:] = frac_sfc
    idx_sfc_q.view[:] = idx_sfc
    frac_sfcd_q.view[:] = frac_sfcd
    idx_sfcd_q.view[:] = idx_sfcd

    # level-sized planck uses a second (nx,ny,nz+1) factory
    sf_lev, qf_lev = get_factories_single_tile(
        nx=nx, ny=ny, nz=nlev, nhalo=3, backend=backend_python
    )
    gi_lev = sf_lev.grid_indexing
    planck_lev_sten = sf_lev.from_origin_domain(
        func=planck_interp1d, origin=gi_lev.origin_compute(), domain=gi_lev.domain_compute()
    )
    frac_lev_q = qf_lev.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)
    idx_lev_q = qf_lev.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Int)
    iband_lev_q = qf_lev.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Int)
    planck_lev_q = qf_lev.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)
    idx_lev, frac_lev = _idx_frac(tlev, temp_ref_min, totplnk_delta, nplancktemp)
    frac_lev_q.view[:] = frac_lev
    idx_lev_q.view[:] = idx_lev

    # ------------------------------------------------------------------------
    # assembly, per band then per g-point in the band
    # ------------------------------------------------------------------------
    sfc_src = np.zeros((nx, ny, ngpt), dtype=np.float64)
    jac_src = np.zeros((nx, ny, ngpt), dtype=np.float64)
    lay_src = np.zeros((nx, ny, nz, ngpt), dtype=np.float64)
    lev_src = np.zeros((nx, ny, nlev, ngpt), dtype=np.float64)

    for ibnd in range(nbnd):
        gptS0 = int(band_lims[0, ibnd]) - 1
        gptE0 = int(band_lims[1, ibnd]) - 1

        iband_q.view[:] = ibnd
        iband_lev_q.view[:] = ibnd
        planck_lay_sten(frac=frac_lay_q, idx=idx_lay_q, iband=iband_q, totplnk=totplnk, planck=planck_lay_q)
        planck_lay_sten(frac=frac_sfc_q, idx=idx_sfc_q, iband=iband_q, totplnk=totplnk, planck=planck_sfc_q)
        planck_lay_sten(frac=frac_sfcd_q, idx=idx_sfcd_q, iband=iband_q, totplnk=totplnk, planck=planck_sfcd_q)
        planck_lev_sten(frac=frac_lev_q, idx=idx_lev_q, iband=iband_lev_q, totplnk=totplnk, planck=planck_lev_q)

        pl_lay = planck_lay_q.view[:]  # (nx,ny,nz)
        pl_sfc = planck_sfc_q.view[:, :, 0]  # (nx,ny)
        pl_sfcd = planck_sfcd_q.view[:, :, 0]  # (nx,ny)
        pl_lev = planck_lev_q.view[:]  # (nx,ny,nz+1)

        for g in range(gptS0, gptE0 + 1):
            pf = pfrac[:, :, :, g]  # (nx,ny,nz)
            sfc_src[:, :, g] = pf[:, :, sfc_lay0] * pl_sfc
            jac_src[:, :, g] = pf[:, :, sfc_lay0] * (pl_sfcd - pl_sfc)
            lay_src[:, :, :, g] = pf * pl_lay
            lev_src[:, :, 0, g] = pf[:, :, 0] * pl_lev[:, :, 0]
            lev_src[:, :, nz, g] = pf[:, :, nz - 1] * pl_lev[:, :, nz]
            lev_src[:, :, 1:nz, g] = (
                np.sqrt(pf[:, :, 0 : nz - 1] * pf[:, :, 1:nz]) * pl_lev[:, :, 1:nz]
            )

    # ------------------------------------------------------------------------
    np.testing.assert_allclose(sfc_src, sfc_ref, rtol=1e-10, atol=1e-22)
    np.testing.assert_allclose(lay_src, lay_ref, rtol=1e-10, atol=1e-22)
    np.testing.assert_allclose(lev_src, lev_ref, rtol=1e-10, atol=1e-22)
    np.testing.assert_allclose(jac_src, jac_ref, rtol=1e-9, atol=1e-22)

    # sanity: real sources gathered and positive
    assert np.all(np.isfinite(lay_src))
    assert np.any(lay_src > 0.0)
    assert np.any(sfc_src > 0.0)

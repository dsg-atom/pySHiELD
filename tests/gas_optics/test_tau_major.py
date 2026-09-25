"""Validate the Stage-5 gas-optics path: major-gas absorption tau vs pyRTE.

This is the major-gas assembly of the Fortran `compute_tau_absorption`
(`gas_optical_depths_major`, kernels lines 347-398). It reuses the validated
Stage-3b kmajor interpolation stencil (`interp3d_major_gpt`) and adds the
assembly logic:
  * per-cell flavor selection -- each g-point's flavor depends on whether the
    cell is in the troposphere (`itropo`) via `gpoint_flavor[g, itropo]`, and
    the selected flavor picks that cell's `col_mix`, `fmajor`, `jeta`;
  * the `jpress + itropo` pressure-bracket offset;
  * the g-point loop that fills tau's "gpt" data axis.

The reference is pyRTE's own compiled RRTMGP Fortran: `GasOptics.tau_absorption`
with the minor-gas coefficient tables (`kminor_lower`, `kminor_upper`) zeroed,
so it returns the major-gas contribution alone. The interpolation intermediates
(`col_mix`, `fmajor`, `jeta`, `jtemp`, `jpress`, `tropo`) come from pyRTE's own
`GasOptics.interpolate`, so this test isolates the major-gas ASSEMBLY from the
interpolation stencils (validated separately).

pyRTE indices are 1-based (Fortran); they are converted here to the 0-based
lower-bracket indices the stencil expects:
  jtemp0  = jtemp_fortran - 1
  jpress0 = jpress_fortran + itropo - 1   (itropo 0=lower/tropo, 1=upper)
  jeta0   = jeta_fortran - 1

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set (pyRTE needs the
coefficient file cache):
    pytest tests/gas_optics/test_tau_major.py -q
"""

import os

import numpy as np
import pytest

from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import backend_python
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Int

from pyshield.radiation.gas_optics import NGPT, interp3d_major_gpt

# A representative column subset: correctness is per-cell independent, so a few
# columns keep the 256 stencil launches fast while all 60 layers exercise both
# troposphere states and the g-point loop exercises all 10 flavors.
SITES = [0, 25, 50, 75]
EXPTS = [0, 9]


def _pyrte():
    """Import pyRTE and stand up the LW_G256 gas optics on the rfmip atmosphere.

    Replicates the setup `GasOptics.compute` does before calling `interpolate`
    (gas mapping, `_gas_names`, the `.mapping` accessor), then returns the
    object, the atmosphere, and the gas mapping. Skips if pyRTE or its
    coefficient cache is unavailable.
    """
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


def test_tau_major():
    import xarray as xr

    go, atm, gm = _pyrte()

    # reference intermediates (pyRTE's compiled interpolation Fortran) ---------
    interp = go.interpolate(atm, gm)

    # major-only reference tau: zero the minor tables, keep everything else, and
    # let pyRTE's compiled tau_absorption Fortran run. Interp was already
    # computed above (it does not depend on kminor).
    go._dataset["kminor_lower"] = xr.zeros_like(go._dataset["kminor_lower"])
    go._dataset["kminor_upper"] = xr.zeros_like(go._dataset["kminor_upper"])
    tau_ds = go.tau_absorption(atm, interp)
    tau_ref = (
        tau_ds["tau"]
        .isel(site=SITES, expt=EXPTS)
        .transpose("site", "expt", "layer", "gpt")
        .values
    )  # (nx, ny, nlay, ngpt)

    nx, ny = len(SITES), len(EXPTS)
    nz = interp.sizes["layer"]
    ngpt = tau_ref.shape[-1]
    assert ngpt == NGPT == 256, (ngpt, NGPT)

    def sub(da, order):
        return da.isel(site=SITES, expt=EXPTS).transpose(*order).values

    # flavor-independent, 1-based -> converted below
    jtemp_f = sub(interp["temperature_index"], ("site", "expt", "layer"))
    jpress_f = sub(interp["pressure_index"], ("site", "expt", "layer"))
    tropo = sub(interp["tropopause_mask"], ("site", "expt", "layer"))

    # per-flavor
    col_mix = sub(
        interp["column_mix"], ("temp_interp", "site", "expt", "layer", "flavor")
    )  # (2, nx, ny, nz, nflav)
    fmajor = sub(
        interp["fmajor"],
        ("eta_interp", "press_interp", "temp_interp", "site", "expt", "layer", "flavor"),
    )  # (2, 2, 2, nx, ny, nz, nflav)
    jeta_f = sub(
        interp["eta_index"], ("pair", "site", "expt", "layer", "flavor")
    )  # (2, nx, ny, nz, nflav)

    gpoint_flavor = (
        go.gpoint_flavor.transpose("gpt", "atmos_layer").values
    )  # (ngpt, 2), 1-based
    kmajor = np.ascontiguousarray(
        go._dataset["kmajor"]
        .transpose("temperature", "mixing_fraction", "pressure_interp", "gpt")
        .values
    )  # (14, 9, 60, 256), matches the KMajor GlobalTable layout

    # index conversion (1-based Fortran -> 0-based stencil) --------------------
    itropo = np.where(tropo, 0, 1).astype(np.int64)  # 0=lower/tropo, 1=upper
    jtemp0 = (jtemp_f - 1).astype(np.int64)
    jpress0 = (jpress_f + itropo - 1).astype(np.int64)  # lower bracket, 0-based

    # stencil (compiled once, driven over g-points) ---------------------------
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=nx, ny=ny, nz=nz, nhalo=3, backend=backend_python
    )
    quantity_factory.add_data_dimensions({"gpt": NGPT})
    grid_indexing = stencil_factory.grid_indexing
    stencil = stencil_factory.from_origin_domain(
        func=interp3d_major_gpt,
        origin=grid_indexing.origin_compute(),
        domain=grid_indexing.domain_compute(),
    )

    def ff():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def fi():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Int)

    s1_q, s2_q = ff(), ff()
    fmaj_names = ("f111", "f211", "f121", "f221", "f112", "f212", "f122", "f222")
    fmaj_q = {name: ff() for name in fmaj_names}
    jtemp_q, jpress_q, jeta1_q, jeta2_q = fi(), fi(), fi(), fi()
    igpt_q = fi()
    res_q = ff()
    tau_q = quantity_factory.zeros([I_DIM, J_DIM, K_DIM, "gpt"], "", dtype=Float)

    # flavor-independent indices are set once
    jtemp_q.view[:] = jtemp0
    jpress_q.view[:] = jpress0

    # fmajor axes (eta, press, temp) -> f<e><p><t>
    fmaj_axes = {
        "f111": (0, 0, 0), "f211": (1, 0, 0), "f121": (0, 1, 0), "f221": (1, 1, 0),
        "f112": (0, 0, 1), "f212": (1, 0, 1), "f122": (0, 1, 1), "f222": (1, 1, 1),
    }

    si, ei, li = np.indices((nx, ny, nz))  # cell index grids for the flavor gather

    for g in range(ngpt):
        # per-cell flavor for this g-point, selected by troposphere state
        iflav = (gpoint_flavor[g, itropo] - 1).astype(np.int64)  # (nx, ny, nz), 0-based

        s1_q.view[:] = col_mix[0][si, ei, li, iflav]
        s2_q.view[:] = col_mix[1][si, ei, li, iflav]
        for name, (e, p, t) in fmaj_axes.items():
            fmaj_q[name].view[:] = fmajor[e, p, t][si, ei, li, iflav]
        jeta1_q.view[:] = (jeta_f[0][si, ei, li, iflav] - 1).astype(np.int64)
        jeta2_q.view[:] = (jeta_f[1][si, ei, li, iflav] - 1).astype(np.int64)
        igpt_q.view[:] = g

        stencil(
            scaling1=s1_q,
            scaling2=s2_q,
            f111=fmaj_q["f111"],
            f211=fmaj_q["f211"],
            f121=fmaj_q["f121"],
            f221=fmaj_q["f221"],
            f112=fmaj_q["f112"],
            f212=fmaj_q["f212"],
            f122=fmaj_q["f122"],
            f222=fmaj_q["f222"],
            jtemp=jtemp_q,
            jpress=jpress_q,
            jeta1=jeta1_q,
            jeta2=jeta2_q,
            igpt=igpt_q,
            kmajor=kmajor,
            res=res_q,
        )
        tau_q.view[:, :, :, g] = res_q.view[:]

    # compare to pyRTE's major-only tau ---------------------------------------
    np.testing.assert_allclose(tau_q.view[:], tau_ref, rtol=1e-10, atol=1e-22)

    # sanity: real absorption was gathered and varies across g-points
    assert np.all(np.isfinite(tau_q.view[:]))
    assert np.any(tau_q.view[:] > 0.0)

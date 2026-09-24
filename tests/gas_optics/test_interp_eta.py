"""Validate the Stage-2 gas-optics stencil (eta / binary-species interp).

This is the first `GlobalTable` gather in the port. `interp_eta_1flavor`
reads the real RRTMGP `vmr_ref` reference table with `table.A[itropo, igas, jt]`
-- runtime integer indices into a table with no spatial dimensions -- and
forms the mixed column, the binary-species parameter eta, and eta's table
index/fraction for the two bracketing reference temperatures of one flavor.

The gather is driven by integer index fields (itropo, jt0, jt1) that span both
atmospheres and the temperature axis, and by synthetic gas columns (including a
zero-zero cell that exercises the eta=0.5 fallback). Outputs are checked
against a numpy reimplementation using the same table and inputs. Passing this
proves the value-indexed multi-axis table gather -- the design risk flagged for
the whole port.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/gas_optics/test_interp_eta.py -q
"""

import glob
import os

import numpy as np
import pytest
import xarray as xr

from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import backend_python
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Int

from pyshield.radiation.gas_optics import interp_eta_1flavor


def _lw_coeff_file():
    cache = os.environ.get("XDG_CACHE_HOME")
    if not cache:
        pytest.skip("XDG_CACHE_HOME not set; cannot locate RRTMGP coeff file")
    hits = sorted(
        glob.glob(os.path.join(cache, "**", "rrtmgp-gas-lw*g256*.nc"), recursive=True)
    )
    if not hits:
        pytest.skip("LW coefficient file not downloaded; run make_reference.py first")
    return hits[0]


def _load_vmr_ref():
    """Return vmr_ref in Fortran layout (atmos_layer=2, absorber_ext, temperature).

    netCDF stores it as (temperature, absorber_ext, atmos_layer); the Fortran
    kernel indexes it (itropo, igas, jtemp), so transpose to that order.
    """
    ds = xr.open_dataset(_lw_coeff_file())
    vmr_nc = ds["vmr_ref"].values.astype(np.float64)  # (temp, absorber_ext, layer)
    vmr = np.ascontiguousarray(vmr_nc.transpose(2, 1, 0))  # (layer, absorber_ext, temp)
    neta = int(ds.sizes["mixing_fraction"])
    return vmr, neta


def test_interp_eta_1flavor():
    nx, ny, nz, nhalo = 4, 4, 8, 3
    vmr, neta = _load_vmr_ref()
    nlayer, nabs, ntemp = vmr.shape
    assert (nlayer, nabs, ntemp) == (2, 20, 14), (nlayer, nabs, ntemp)

    # pick two absorbers whose reference mixing ratio is strictly positive
    # everywhere (avoids div-by-zero in the ratio) -- these stand in for a
    # flavor's two key species; real flavor->gas mapping is host-side setup.
    positive = [j for j in range(nabs) if np.all(vmr[:, j, :] > 0.0)]
    assert len(positive) >= 2, positive
    igas1, igas2 = positive[0], positive[1]

    neta_m1 = float(neta - 1)
    eta_half_thresh = 2.0 * np.finfo(np.float64).tiny

    # integer index fields spanning both atmospheres and the temperature axis
    ii, jj, kk = np.indices((nx, ny, nz))
    flat = ii * (ny * nz) + jj * nz + kk
    jt0 = (flat % (ntemp - 1)).astype(np.int64)  # 0 .. ntemp-2
    jt1 = jt0 + 1  # 1 .. ntemp-1
    itropo = (kk >= nz // 2).astype(np.int64)  # 0 lower, 1 upper

    rng = np.random.default_rng(0)
    col_gas1 = rng.uniform(1.0, 10.0, (nx, ny, nz))
    col_gas2 = rng.uniform(0.0, 5.0, (nx, ny, nz))
    col_gas1[0, 0, 0] = 0.0  # zero-zero cell -> col_mix=0 -> eta=0.5 fallback
    col_gas2[0, 0, 0] = 0.0

    # numpy oracle -----------------------------------------------------------
    def bracket(jt):
        ratio = vmr[itropo, igas1, jt] / vmr[itropo, igas2, jt]
        col_mix = col_gas1 + ratio * col_gas2
        eta = np.full_like(col_mix, 0.5)
        mask = col_mix > eta_half_thresh
        eta[mask] = col_gas1[mask] / col_mix[mask]
        loceta = eta * neta_m1
        jeta = np.minimum(np.floor(loceta) + 1.0, neta_m1)
        feta = loceta - np.floor(loceta)
        return col_mix, jeta, feta

    o_col_mix0, o_jeta0, o_feta0 = bracket(jt0)
    o_col_mix1, o_jeta1, o_feta1 = bracket(jt1)

    # stencil ----------------------------------------------------------------
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=nx, ny=ny, nz=nz, nhalo=nhalo, backend=backend_python
    )
    grid_indexing = stencil_factory.grid_indexing
    stencil = stencil_factory.from_origin_domain(
        func=interp_eta_1flavor,
        externals={
            "igas1": int(igas1),
            "igas2": int(igas2),
            "neta_m1": neta_m1,
            "eta_half_thresh": eta_half_thresh,
        },
        origin=grid_indexing.origin_compute(),
        domain=grid_indexing.domain_compute(),
    )

    def ff():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def fi():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Int)

    col_gas1_q, col_gas2_q = ff(), ff()
    itropo_q, jt0_q, jt1_q = fi(), fi(), fi()
    col_mix0_q, col_mix1_q = ff(), ff()
    jeta0_q, jeta1_q, feta0_q, feta1_q = ff(), ff(), ff(), ff()

    col_gas1_q.view[:] = col_gas1
    col_gas2_q.view[:] = col_gas2
    itropo_q.view[:] = itropo
    jt0_q.view[:] = jt0
    jt1_q.view[:] = jt1

    stencil(
        col_gas1=col_gas1_q,
        col_gas2=col_gas2_q,
        itropo=itropo_q,
        jt0=jt0_q,
        jt1=jt1_q,
        vmr_ref=vmr,
        col_mix0=col_mix0_q,
        col_mix1=col_mix1_q,
        jeta0=jeta0_q,
        jeta1=jeta1_q,
        feta0=feta0_q,
        feta1=feta1_q,
    )

    # compare ----------------------------------------------------------------
    np.testing.assert_allclose(col_mix0_q.view[:], o_col_mix0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(col_mix1_q.view[:], o_col_mix1, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(jeta0_q.view[:], o_jeta0)
    np.testing.assert_array_equal(jeta1_q.view[:], o_jeta1)
    np.testing.assert_allclose(feta0_q.view[:], o_feta0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(feta1_q.view[:], o_feta1, rtol=0, atol=1e-12)

    # sanity: the gather and both branches are actually exercised
    assert set(np.unique(itropo)) == {0, 1}
    assert np.unique(jt0).size > 1
    assert o_col_mix0[0, 0, 0] == 0.0  # the zero-zero cell hit the eta=0.5 path
    assert o_jeta0.min() >= 1.0 and o_jeta0.max() <= neta_m1

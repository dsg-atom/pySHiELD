"""Validate the Stage-3 gas-optics stencil (major-gas 8-point k interp).

`interp3d_major_1gpt` is a faithful port of the Fortran `interpolate3D_byflav`
inner expression, evaluated at one fixed g-point. It is the first 4-axis
`GlobalTable` gather (the real `kmajor` absorption table) and the first use of
integer index arithmetic (jtemp+1, jpress+1, jeta+1) as gather indices.

This validates the kernel *mechanism* against a numpy transcription using the
same kmajor table and the same inputs -- not against ref_gas_optics_lw.nc
(that end-to-end tau check comes once the full flavor/g-point/minor-gas path is
assembled). Indices are synthetic-but-valid; correctness of the physical
indices is a later end-to-end concern.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/gas_optics/test_interp3d_major.py -q
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

from pyshield.radiation.gas_optics import interp3d_major_1gpt

IGPT = 100  # fixed g-point to evaluate (into kmajor axis 3, size 256)


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


def _load_kmajor():
    """Return kmajor in Fortran layout (temperature, eta, pressure, gpt).

    netCDF stores it (temperature, pressure, eta, gpt); the Fortran kernel
    indexes it (temp, eta, press, gpt), so swap the pressure and eta axes.
    """
    ds = xr.open_dataset(_lw_coeff_file())
    k_nc = ds["kmajor"].values.astype(np.float64)  # (temp, press, eta, gpt)
    kmajor = np.ascontiguousarray(k_nc.transpose(0, 2, 1, 3))  # (temp, eta, press, gpt)
    return kmajor


def test_interp3d_major_1gpt():
    nx, ny, nz, nhalo = 4, 4, 8, 3
    kmajor = _load_kmajor()
    ntemp, neta, npress, ngpt = kmajor.shape
    assert (ntemp, neta, npress, ngpt) == (14, 9, 60, 256), kmajor.shape
    assert 0 <= IGPT < ngpt

    n = nx * ny * nz
    rng = np.random.default_rng(1)

    # synthetic but valid lower-bracket indices (upper bracket = +1 in-stencil)
    ii, jj, kk = np.indices((nx, ny, nz))
    flat = ii * (ny * nz) + jj * nz + kk
    jtemp = (flat % (ntemp - 1)).astype(np.int64)  # 0 .. ntemp-2
    jpress = (flat % (npress - 1)).astype(np.int64)  # 0 .. npress-2
    jeta1 = (flat % (neta - 1)).astype(np.int64)  # 0 .. neta-2
    jeta2 = ((flat + 1) % (neta - 1)).astype(np.int64)

    scaling1 = rng.uniform(0.5, 2.0, (nx, ny, nz))
    scaling2 = rng.uniform(0.5, 2.0, (nx, ny, nz))
    fw = {name: rng.uniform(0.0, 1.0, (nx, ny, nz)) for name in
          ("f111", "f211", "f121", "f221", "f112", "f212", "f122", "f222")}

    # numpy oracle (mirrors interpolate3D_byflav at g-point IGPT) --------------
    jtp, jpp, je1p, je2p = jtemp + 1, jpress + 1, jeta1 + 1, jeta2 + 1
    g = IGPT
    o_res = scaling1 * (
        fw["f111"] * kmajor[jtemp, jeta1, jpress, g]
        + fw["f211"] * kmajor[jtemp, je1p, jpress, g]
        + fw["f121"] * kmajor[jtemp, jeta1, jpp, g]
        + fw["f221"] * kmajor[jtemp, je1p, jpp, g]
    ) + scaling2 * (
        fw["f112"] * kmajor[jtp, jeta2, jpress, g]
        + fw["f212"] * kmajor[jtp, je2p, jpress, g]
        + fw["f122"] * kmajor[jtp, jeta2, jpp, g]
        + fw["f222"] * kmajor[jtp, je2p, jpp, g]
    )

    # stencil -----------------------------------------------------------------
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=nx, ny=ny, nz=nz, nhalo=nhalo, backend=backend_python
    )
    grid_indexing = stencil_factory.grid_indexing
    stencil = stencil_factory.from_origin_domain(
        func=interp3d_major_1gpt,
        externals={"igpt": IGPT},
        origin=grid_indexing.origin_compute(),
        domain=grid_indexing.domain_compute(),
    )

    def ff():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def fi():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Int)

    scaling1_q, scaling2_q = ff(), ff()
    fw_q = {name: ff() for name in fw}
    jtemp_q, jpress_q, jeta1_q, jeta2_q = fi(), fi(), fi(), fi()
    res_q = ff()

    scaling1_q.view[:] = scaling1
    scaling2_q.view[:] = scaling2
    for name, arr in fw.items():
        fw_q[name].view[:] = arr
    jtemp_q.view[:] = jtemp
    jpress_q.view[:] = jpress
    jeta1_q.view[:] = jeta1
    jeta2_q.view[:] = jeta2

    stencil(
        scaling1=scaling1_q,
        scaling2=scaling2_q,
        f111=fw_q["f111"],
        f211=fw_q["f211"],
        f121=fw_q["f121"],
        f221=fw_q["f221"],
        f112=fw_q["f112"],
        f212=fw_q["f212"],
        f122=fw_q["f122"],
        f222=fw_q["f222"],
        jtemp=jtemp_q,
        jpress=jpress_q,
        jeta1=jeta1_q,
        jeta2=jeta2_q,
        kmajor=kmajor,
        res=res_q,
    )

    np.testing.assert_allclose(res_q.view[:], o_res, rtol=1e-12, atol=0.0)

    # sanity: real table values were gathered and the result is nontrivial
    assert np.all(np.isfinite(res_q.view[:]))
    assert np.unique(jpress).size > 1 and np.unique(jtemp).size > 1
    assert np.any(res_q.view[:] != 0.0)

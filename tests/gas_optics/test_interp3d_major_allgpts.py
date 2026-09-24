"""Validate the Stage-3b gas-optics path (all g-points, data-dim tau).

`interp3d_major_gpt` runs the same 8-point kmajor interpolation as
`interp3d_major_1gpt`, but selects the g-point with a runtime index field.
It compiles once and is driven in a Python loop over g-points; each call's
result is written into the g-point data axis of a `tau` field, so all NGPT
g-points are filled. This is how the port represents the g-point axis:
columns/layers are the framework's (i, j, k), the g-points are a host loop.

A single in-stencil g-point loop is not available in this gt4py.cartesian
frontend (no `for ... in range()`, and data-dim accesses are unrolled at
compile time, so a runtime data-dim write index is unsupported). The
loop-driven form reuses only already-proven pieces.

What this proves beyond Stage 3:
  * a spatial field carrying a g-point data dimension ("gpt"), allocated via
    quantity_factory.add_data_dimensions + a data-axis name in `dims`, filled
    slice by slice;
  * a runtime g-point index field into the kmajor gather (`.A[..., igpt]`);
  * assembly of the full per-g-point tau, matching a numpy transcription over
    all g-points using the same kmajor table.

This is still a mechanism check, not the end-to-end tau vs ref_gas_optics_lw.nc
(that needs the full flavor/band/minor-gas path).

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/gas_optics/test_interp3d_major_allgpts.py -q
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

from pyshield.radiation.gas_optics import NGPT, interp3d_major_gpt


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
    """kmajor in Fortran layout (temperature, eta, pressure, gpt)."""
    ds = xr.open_dataset(_lw_coeff_file())
    k_nc = ds["kmajor"].values.astype(np.float64)  # (temp, press, eta, gpt)
    return np.ascontiguousarray(k_nc.transpose(0, 2, 1, 3))  # (temp, eta, press, gpt)


def test_interp3d_major_allgpts():
    nx, ny, nz, nhalo = 4, 4, 8, 3
    kmajor = _load_kmajor()
    ntemp, neta, npress, ngpt = kmajor.shape
    assert (ntemp, neta, npress, ngpt) == (14, 9, 60, 256), kmajor.shape
    assert NGPT == ngpt  # the tau data-dim size must match kmajor's g-point axis

    rng = np.random.default_rng(2)

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

    # numpy oracle over ALL g-points -----------------------------------------
    # advanced indexing with a trailing ":" gathers every g-point at once:
    # kmajor[jtemp, jeta1, jpress, :] -> (nx, ny, nz, ngpt)
    jtp, jpp, je1p, je2p = jtemp + 1, jpress + 1, jeta1 + 1, jeta2 + 1
    s1 = scaling1[..., None]
    s2 = scaling2[..., None]

    def w(name):
        return fw[name][..., None]

    o_tau = s1 * (
        w("f111") * kmajor[jtemp, jeta1, jpress, :]
        + w("f211") * kmajor[jtemp, je1p, jpress, :]
        + w("f121") * kmajor[jtemp, jeta1, jpp, :]
        + w("f221") * kmajor[jtemp, je1p, jpp, :]
    ) + s2 * (
        w("f112") * kmajor[jtp, jeta2, jpress, :]
        + w("f212") * kmajor[jtp, je2p, jpress, :]
        + w("f122") * kmajor[jtp, jeta2, jpp, :]
        + w("f222") * kmajor[jtp, je2p, jpp, :]
    )
    assert o_tau.shape == (nx, ny, nz, ngpt)

    # stencil (compiled once, driven over g-points) --------------------------
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=nx, ny=ny, nz=nz, nhalo=nhalo, backend=backend_python
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

    scaling1_q, scaling2_q = ff(), ff()
    fw_q = {name: ff() for name in fw}
    jtemp_q, jpress_q, jeta1_q, jeta2_q = fi(), fi(), fi(), fi()
    igpt_q = fi()
    res_q = ff()
    tau_q = quantity_factory.zeros([I_DIM, J_DIM, K_DIM, "gpt"], "", dtype=Float)
    assert tau_q.view[:].shape == (nx, ny, nz, ngpt), tau_q.view[:].shape

    scaling1_q.view[:] = scaling1
    scaling2_q.view[:] = scaling2
    for name, arr in fw.items():
        fw_q[name].view[:] = arr
    jtemp_q.view[:] = jtemp
    jpress_q.view[:] = jpress
    jeta1_q.view[:] = jeta1
    jeta2_q.view[:] = jeta2

    # drive the single compiled stencil once per g-point, assembling tau
    for g in range(ngpt):
        igpt_q.view[:] = g
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
            igpt=igpt_q,
            kmajor=kmajor,
            res=res_q,
        )
        tau_q.view[:, :, :, g] = res_q.view[:]

    np.testing.assert_allclose(tau_q.view[:], o_tau, rtol=1e-12, atol=0.0)

    # sanity: the g-point axis is genuinely populated and varies
    assert np.all(np.isfinite(tau_q.view[:]))
    assert np.any(tau_q.view[:] != 0.0)
    assert np.unique(tau_q.view[0, 0, 0, :]).size > 1  # g-points differ

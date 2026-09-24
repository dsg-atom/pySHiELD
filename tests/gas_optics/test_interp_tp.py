"""Validate the Stage-1 gas-optics stencil (T/p interpolation indices).

Runs `interp_tp` on the numpy backend against a numpy reimplementation of
the same Fortran formulas, using synthetic profiles that span the RRTMGP
table ranges. The reference table constants (temp_ref, press_ref,
press_ref_trop) are read from the LW coefficient file pyRTE downloads.

This is the first GT4Py stencil in the port: passing it proves the NDSL
harness (single-tile factories, from_origin_domain, numpy-backend compile,
data in/out) as much as it proves the formulas.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/gas_optics/test_interp_tp.py -q
"""

import glob
import os

import numpy as np
import pytest
import xarray as xr

from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import backend_python
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float

from pyshield.radiation.gas_optics import interp_tp


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


def _table_constants():
    ds = xr.open_dataset(_lw_coeff_file())
    temp_ref = ds["temp_ref"].values.astype(np.float64)
    press_ref = ds["press_ref"].values.astype(np.float64)
    press_ref_trop = float(ds["press_ref_trop"].values)
    ntemp = temp_ref.size
    npres = press_ref.size
    return {
        "temp_ref_min": float(temp_ref[0]),
        "temp_ref_delta": float((temp_ref[-1] - temp_ref[0]) / (ntemp - 1)),
        "ntemp": float(ntemp),
        "press_ref_log_1": float(np.log(press_ref[0])),
        "press_ref_log_delta": float(
            (np.log(press_ref[-1]) - np.log(press_ref[0])) / (npres - 1)
        ),
        "npres": float(npres),
        "press_ref_trop_log": float(np.log(press_ref_trop)),
        "_temp_ref": temp_ref,
        "_press_ref": press_ref,
    }


def _synthetic_profiles(nx, ny, nz, c):
    """Profiles that span the table ranges and cross the troposphere line."""
    temp_ref_max = c["temp_ref_min"] + (c["ntemp"] - 1) * c["temp_ref_delta"]
    press_ref_max = c["_press_ref"][0]
    press_ref_min = c["_press_ref"][-1]

    # temperature varies horizontally across the whole reference range
    tcol = np.linspace(c["temp_ref_min"] + 2.0, temp_ref_max - 2.0, nx * ny)
    tlay = np.repeat(tcol.reshape(nx, ny, 1), nz, axis=2)

    # pressure decreases with height (k), log-spaced across the whole range
    pk = np.exp(np.linspace(np.log(press_ref_max), np.log(press_ref_min), nz))
    play = np.broadcast_to(pk.reshape(1, 1, nz), (nx, ny, nz)).copy()
    return play.astype(np.float64), tlay.astype(np.float64)


def _numpy_oracle(play, tlay, c):
    logp = np.log(play)
    jt = np.floor((tlay - (c["temp_ref_min"] - c["temp_ref_delta"])) / c["temp_ref_delta"])
    jt = np.minimum(c["ntemp"] - 1.0, np.maximum(1.0, jt))
    ftemp = (tlay - (c["temp_ref_min"] + (jt - 1.0) * c["temp_ref_delta"])) / c[
        "temp_ref_delta"
    ]
    locpress = 1.0 + (logp - c["press_ref_log_1"]) / c["press_ref_log_delta"]
    jp = np.minimum(c["npres"] - 1.0, np.maximum(1.0, np.floor(locpress)))
    fpress = locpress - jp
    tropo = np.where(logp > c["press_ref_trop_log"], 1.0, 0.0)
    return jt, ftemp, jp, fpress, tropo


def test_interp_tp():
    nx, ny, nz, nhalo = 4, 4, 60, 3
    c = _table_constants()

    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=nx, ny=ny, nz=nz, nhalo=nhalo, backend=backend_python
    )
    grid_indexing = stencil_factory.grid_indexing
    stencil = stencil_factory.from_origin_domain(
        func=interp_tp,
        externals={k: v for k, v in c.items() if not k.startswith("_")},
        origin=grid_indexing.origin_compute(),
        domain=grid_indexing.domain_compute(),
    )

    def field():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    play_q, tlay_q = field(), field()
    jtemp_q, ftemp_q, jpress_q, fpress_q, tropo_q = (field() for _ in range(5))

    play, tlay = _synthetic_profiles(nx, ny, nz, c)
    play_q.view[:] = play
    tlay_q.view[:] = tlay

    stencil(
        play=play_q,
        tlay=tlay_q,
        jtemp=jtemp_q,
        ftemp=ftemp_q,
        jpress=jpress_q,
        fpress=fpress_q,
        tropo=tropo_q,
    )

    jt, ftemp, jp, fpress, tropo = _numpy_oracle(play, tlay, c)

    # indices are integer-valued: require exact agreement
    np.testing.assert_array_equal(jtemp_q.view[:], jt)
    np.testing.assert_array_equal(jpress_q.view[:], jp)
    # fractions and flag
    np.testing.assert_allclose(ftemp_q.view[:], ftemp, rtol=0, atol=1e-12)
    np.testing.assert_allclose(fpress_q.view[:], fpress, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(tropo_q.view[:], tropo)

    # sanity: the profiles actually exercise more than one bin and both atmospheres
    assert np.unique(jt).size > 1
    assert np.unique(jp).size > 1
    assert set(np.unique(tropo)) == {0.0, 1.0}

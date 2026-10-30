"""Column blocking == whole tile, bit-for-bit (the memory-bounding lever).

The driver can process a rank's columns in blocks of `block_rows` nx-rows
(GEOS_RRTMGP_BLOCK_ROWS) instead of the whole tile at once, so the big
(cols, layer, gpt) optics arrays -- and thus per-rank GPU memory -- are bounded
by the block size. Radiation is column-independent, so blocking must NOT change
any number: each column is gathered, cloud-combined and solved identically
regardless of how many columns share the tile.

This test proves that: it runs the SAME synthetic state through the bridge's
step_radiation twice -- once whole-tile (blocking off) and once blocked -- on
fresh states, and asserts every output flux matches bit-for-bit. Clouds are
non-zero so the per-block cloud add_to is exercised.

The block tile must exceed the halo (SubtileGridSizer needs bx > nhalo), so the
tile here is nx=8 with block_rows=4 -> 2 blocks of a 4x4 tile. On CPU the backend
is numpy; run it on the A100 too (RTE_TEST_BACKEND=dace:gpu) to confirm the
blocked path's device concatenation.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/geos_bridge/test_blocking_equiv.py -q
"""

import datetime
import os

import numpy as np
import pytest

from ndsl.config import backend_python, backend_gpu
from ndsl.dsl.typing import Float

_BACKEND = backend_gpu if os.environ.get("RTE_TEST_BACKEND") else backend_python

NX, NY, NZ, NHALO = 8, 4, 24, 3
NCOL = NX * NY
BLOCK_ROWS = 4  # divides NX=8 -> 2 blocks; bx=4 > nhalo=3


def _require_env():
    if not os.environ.get("XDG_CACHE_HOME"):
        pytest.skip("XDG_CACHE_HOME not set; the driver fetches coeff files via pyRTE.")
    try:
        import pyrte_rrtmgp  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pyrte_rrtmgp not importable: {exc}")


def _cloudy_profile():
    """Plausible top-down column with non-zero liquid + ice cloud (exercises the
    per-block cloud add_to)."""
    p_lev = np.geomspace(50.0, 1.0e5, NZ + 1).astype(Float)
    p_lay = (0.5 * (p_lev[:-1] + p_lev[1:])).astype(Float)
    t_lay = np.full(NZ, 250.0, dtype=Float)

    def tile_lev(v):
        return np.tile(v, (NCOL, 1)).astype(Float)

    qliq = np.zeros((NCOL, NZ), dtype=Float)
    qice = np.zeros((NCOL, NZ), dtype=Float)
    qcld = np.zeros((NCOL, NZ), dtype=Float)
    mid = slice(NZ // 3, 2 * NZ // 3)
    qliq[:, mid] = 5.0e-5
    qice[:, mid] = 2.0e-5
    qcld[:, mid] = 0.8

    inputs = {
        "prsi": tile_lev(p_lev),
        "prsl": tile_lev(p_lay),
        "tlyr": tile_lev(t_lay),
        "tsfc": np.full(NCOL, 288.0, dtype=Float),
        "qvapor": np.full((NCOL, NZ), 1.0e-3, dtype=Float),
        "qo3mr": np.full((NCOL, NZ), 1.0e-7, dtype=Float),
        "co2": np.full((NCOL, NZ), 400.0e-6, dtype=Float),
        "qliquid": qliq,
        "qice": qice,
        "qcld": qcld,
    }
    sfc_inputs = {
        "tsfc": np.full(NCOL, 288.0, dtype=Float),
        "islmsk": np.zeros(NCOL, dtype=Float),
    }
    return inputs, sfc_inputs, p_lev


def _config(tmp_path, date):
    from pathlib import Path

    from pyshield.radiation import RTE_RRTMGPConfig

    return RTE_RRTMGPConfig(
        deltsw=3600.0, delt_rad=3600.0, date=date, fhswr=1.0, fhlwr=1.0,
        isolar=10, icmphys=4, ico2flg=0, ioznflg=1, ictmflg=-1, ialbflg=-1,
        iemsflg=0, ldisable_radiation_quasi_sea_ice=False,
        solar_constant_file=Path("global_solarconstant_noaa_an.txt"),
        input_dir=Path(str(tmp_path)), aerosol_file=Path(str(tmp_path)),
        sollat=0.0, nstp=6, ivflip=1, lcnorm=False, lcrick=False,
        gfs_cloud_overlap=False,
    )


def _run_once(tmp_path, date, inputs, sfc_inputs, p_lev):
    """One full bridge cycle: init -> run -> extract -> finalize. Blocking is
    whatever GEOS_RRTMGP_BLOCK_ROWS is set to at init time."""
    from pyshield.geos_bridge import geos_rrtmgp, marshal
    from pyshield.radiation.state import RTE_RRTMGPState

    lons = np.full(NCOL, 0.0, dtype=Float)
    lats = np.full(NCOL, 0.0, dtype=Float)
    sigma = (p_lev / p_lev[-1]).astype(Float)

    geos_rrtmgp.geos_rrtmgp_finalize()
    geos_rrtmgp.geos_rrtmgp_init(
        config=_config(tmp_path, date), nx=NX, ny=NY, nz=NZ,
        lons=lons, lats=lats, sigma=sigma, nhalo=NHALO,
        backend=_BACKEND, top_at_1=True,
    )
    inst = geos_rrtmgp.GEOS_RRTMGP
    st = RTE_RRTMGPState.init_zeros(inst.quantity_factory)
    marshal.marshal_inputs(st, inst.sfc_state, inputs, sfc_inputs, NX, NY, NZ, True)
    inst._driver.step_radiation(st, inst.sfc_state, date)
    out = marshal.extract_outputs(st, NX, NY, NZ, True)
    nblocks = inst._driver._nblocks
    geos_rrtmgp.geos_rrtmgp_finalize()
    return out, nblocks


def test_blocking_matches_whole_tile(tmp_path):
    _require_env()

    date = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone.utc)
    inputs, sfc_inputs, p_lev = _cloudy_profile()

    # whole tile (blocking off)
    os.environ.pop("GEOS_RRTMGP_BLOCK_ROWS", None)
    out_whole, nb_whole = _run_once(tmp_path, date, inputs, sfc_inputs, p_lev)
    assert nb_whole == 1

    # blocked
    os.environ["GEOS_RRTMGP_BLOCK_ROWS"] = str(BLOCK_ROWS)
    try:
        out_blk, nb_blk = _run_once(tmp_path, date, inputs, sfc_inputs, p_lev)
    finally:
        os.environ.pop("GEOS_RRTMGP_BLOCK_ROWS", None)
    assert nb_blk == NX // BLOCK_ROWS  # really split into >1 block

    assert set(out_whole) == set(out_blk)
    for name in out_whole:
        assert np.all(np.isfinite(out_blk[name])), name
        np.testing.assert_array_equal(
            out_blk[name], out_whole[name],
            err_msg=f"blocked vs whole tile: {name}",
        )

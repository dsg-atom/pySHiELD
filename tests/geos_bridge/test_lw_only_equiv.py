"""LW-only mode: longwave == full run bit-for-bit, shortwave skipped (zero).

In the IRRAD driving mode (RUN_RRTMGP_GT4PY=2) the model uses native shortwave
and discards the bridge's SW, so the bridge computing it is wasted work -- the
profile's single largest cost (led by _tau_rayleigh, which is SW-only). The
`GEOS_RRTMGP_LW_ONLY` flag makes step_radiation skip the entire SW path.

Skipping discarded work must not change the longwave. This test runs the SAME
synthetic state through step_radiation twice -- full, then LW-only -- on fresh
states, and asserts every LW output (flw*, hrtlw*) matches bit-for-bit while
every SW output (fsw*, hrtsw*) is zero in the LW-only run. Non-zero clouds so
the LW cloud path is exercised.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/geos_bridge/test_lw_only_equiv.py -q
"""

import datetime
import os

import numpy as np
import pytest

from ndsl.config import backend_python, backend_gpu
from ndsl.dsl.typing import Float

_BACKEND = backend_gpu if os.environ.get("RTE_TEST_BACKEND") else backend_python

NX, NY, NZ, NHALO = 4, 4, 24, 3
NCOL = NX * NY


def _require_env():
    if not os.environ.get("XDG_CACHE_HOME"):
        pytest.skip("XDG_CACHE_HOME not set; the driver fetches coeff files via pyRTE.")
    try:
        import pyrte_rrtmgp  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pyrte_rrtmgp not importable: {exc}")


def _cloudy_profile():
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
    """One full bridge cycle. LW-only is whatever GEOS_RRTMGP_LW_ONLY is at init."""
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
    lw_only = inst._driver._lw_only
    st = RTE_RRTMGPState.init_zeros(inst.quantity_factory)
    marshal.marshal_inputs(st, inst.sfc_state, inputs, sfc_inputs, NX, NY, NZ, True)
    inst._driver.step_radiation(st, inst.sfc_state, date)
    out = marshal.extract_outputs(st, NX, NY, NZ, True)
    geos_rrtmgp.geos_rrtmgp_finalize()
    return out, lw_only


def test_lw_only_matches_full_lw(tmp_path):
    _require_env()

    date = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone.utc)
    inputs, sfc_inputs, p_lev = _cloudy_profile()

    os.environ.pop("GEOS_RRTMGP_LW_ONLY", None)
    out_full, full_flag = _run_once(tmp_path, date, inputs, sfc_inputs, p_lev)
    assert full_flag is False

    os.environ["GEOS_RRTMGP_LW_ONLY"] = "1"
    try:
        out_lw, lw_flag = _run_once(tmp_path, date, inputs, sfc_inputs, p_lev)
    finally:
        os.environ.pop("GEOS_RRTMGP_LW_ONLY", None)
    assert lw_flag is True

    assert set(out_full) == set(out_lw)
    checked_lw = checked_sw = 0
    for name in out_full:
        low = name.lower()
        if "lw" in low and "sw" not in low:
            # longwave must be untouched by skipping shortwave
            np.testing.assert_array_equal(
                out_lw[name], out_full[name], err_msg=f"LW changed: {name}"
            )
            checked_lw += 1
        elif "sw" in low and "lw" not in low:
            # shortwave is skipped -> zero in the LW-only run
            assert np.all(out_lw[name] == 0.0), f"SW not zero in lw_only: {name}"
            checked_sw += 1
    assert checked_lw > 0 and checked_sw > 0, (checked_lw, checked_sw)

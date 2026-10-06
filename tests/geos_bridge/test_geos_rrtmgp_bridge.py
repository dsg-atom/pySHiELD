"""End-to-end GEOS radiation bridge == direct step_radiation (Phase 0).

Proves the bridge is a pure pass-through: it marshals synthetic whole-tile numpy
inputs through ``geos_rrtmgp_run``, and ALSO marshals the SAME inputs into an
independent fresh RTE_RRTMGPState and calls the driver's ``step_radiation``
directly; the two sets of output arrays must match bit-for-bit.

DATA GATE. Unlike the marshalling round-trip, this test constructs the real
RTE_RRTMGPDriver, which builds GasOpticsGT4Py / CloudOpticsGT4Py -- these pull
the RRTMGP + cloud-optics coefficient files through pyRTE's pooch cache. So this
test needs, on Discover:
  * ndsl and pyrte_rrtmgp importable, and
  * XDG_CACHE_HOME set (pyRTE caches/downloads the coeff files there; without it
    the fetch hits the disk-quota error).
It does NOT need a c48 restart: the atmospheric state is synthesized here. The
blocked integration test (tests/integration/test_radiation_driver.py) needs the
restart only because it fills the state from fv_core/phy/tracer .nc files.

With the chosen config (isolar=10, iemsflg=0, ico2flg=0, ictmflg=-1, ioznflg=1)
__init__ reads NO input_dir text files (sol_init/sfc_init/gas_init all take their
fixed-value early-return paths), so input_dir points at an empty tmp dir.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/geos_bridge/test_geos_rrtmgp_bridge.py -q
"""

import datetime
import os

import numpy as np
import pytest

from ndsl.config import backend_python
from ndsl.dsl.typing import Float


# NDSL requires nx > nhalo (SubtileGridSizer), so the tile must exceed the halo.
NX, NY, NZ, NHALO = 4, 4, 24, 3
NCOL = NX * NY


def _require_env():
    if not os.environ.get("XDG_CACHE_HOME"):
        pytest.skip(
            "XDG_CACHE_HOME not set; RTE_RRTMGPDriver.__init__ builds "
            "GasOptics/CloudOptics which fetch the RRTMGP + cloud-optics coeff "
            "files through pyRTE's pooch cache. Set XDG_CACHE_HOME to a nobackup "
            "path and rerun. (No c48 restart is needed; the state is synthetic.)"
        )
    try:
        import pyrte_rrtmgp  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pyrte_rrtmgp not importable: {exc}")


def _synthetic_profile():
    """A plausible top-down column profile well inside RRTMGP's valid ranges.

    Returns the whole-tile input dicts (top_at_1=True ordering: index 0 = TOA).
    Clouds are zero (clear == all-sky), which keeps the test focused on the
    pass-through and off cloud-range edge cases.
    """
    # Interface pressure: 50 Pa (TOA) -> 1.0e5 Pa (surface), geometric, top-down.
    p_lev = np.geomspace(50.0, 1.0e5, NZ + 1).astype(Float)
    p_lay = (0.5 * (p_lev[:-1] + p_lev[1:])).astype(Float)
    t_lay = np.full(NZ, 250.0, dtype=Float)

    def tile_lev(v):
        return np.tile(v, (NCOL, 1)).astype(Float)

    inputs = {
        "prsi": tile_lev(p_lev),
        "prsl": tile_lev(p_lay),
        "tlyr": tile_lev(t_lay),
        "tsfc": np.full(NCOL, 288.0, dtype=Float),
        "qvapor": np.full((NCOL, NZ), 1.0e-3, dtype=Float),  # mol/mol
        "qo3mr": np.full((NCOL, NZ), 1.0e-7, dtype=Float),   # mol/mol
        "co2": np.full((NCOL, NZ), 400.0e-6, dtype=Float),
        "qliquid": np.zeros((NCOL, NZ), dtype=Float),
        "qice": np.zeros((NCOL, NZ), dtype=Float),
        "qcld": np.zeros((NCOL, NZ), dtype=Float),
    }
    sfc_inputs = {
        "tsfc": np.full(NCOL, 288.0, dtype=Float),
        "islmsk": np.zeros(NCOL, dtype=Float),  # ocean
    }
    return inputs, sfc_inputs, p_lev


def _config(tmp_path, date):
    from pathlib import Path

    from pyshield.radiation import RTE_RRTMGPConfig

    return RTE_RRTMGPConfig(
        deltsw=3600.0,
        delt_rad=3600.0,
        date=date,
        fhswr=1.0,
        fhlwr=1.0,
        isolar=10,
        icmphys=4,
        ico2flg=0,
        ioznflg=1,
        ictmflg=-1,
        ialbflg=-1,
        iemsflg=0,
        ldisable_radiation_quasi_sea_ice=False,
        solar_constant_file=Path("global_solarconstant_noaa_an.txt"),
        input_dir=Path(str(tmp_path)),
        aerosol_file=Path(str(tmp_path)),
        sollat=0.0,
        nstp=6,
        ivflip=1,
        lcnorm=False,
        lcrick=False,
        gfs_cloud_overlap=False,
    )


def test_bridge_matches_direct_step_radiation(tmp_path):
    _require_env()

    from pyshield.geos_bridge import geos_rrtmgp
    from pyshield.geos_bridge import marshal
    from pyshield.radiation.state import RTE_RRTMGPState

    date = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone.utc)
    inputs, sfc_inputs, p_lev = _synthetic_profile()

    # Grid lon/lat and a monotonic sigma (top-down pressure / surface pressure).
    lons = np.full(NCOL, 0.0, dtype=Float)
    lats = np.full(NCOL, 0.0, dtype=Float)
    sigma = (p_lev / p_lev[-1]).astype(Float)

    config = _config(tmp_path, date)

    geos_rrtmgp.geos_rrtmgp_finalize()  # ensure clean singleton
    geos_rrtmgp.geos_rrtmgp_init(
        config=config,
        nx=NX,
        ny=NY,
        nz=NZ,
        lons=lons,
        lats=lats,
        sigma=sigma,
        nhalo=NHALO,
        backend=backend_python,
        top_at_1=True,
    )

    # Double-init guard.
    with pytest.raises(RuntimeError):
        geos_rrtmgp.geos_rrtmgp_init(
            config=config, nx=NX, ny=NY, nz=NZ, lons=lons, lats=lats, sigma=sigma,
            nhalo=NHALO, backend=backend_python, top_at_1=True,
        )

    out_bridge = geos_rrtmgp.geos_rrtmgp_run(inputs, sfc_inputs, date, top_at_1=True)

    # Direct reference: fresh independent state, SAME inputs, SAME driver.
    inst = geos_rrtmgp.GEOS_RRTMGP
    direct_state = RTE_RRTMGPState.init_zeros(inst.quantity_factory)
    marshal.marshal_inputs(
        direct_state, inst.sfc_state, inputs, sfc_inputs, NX, NY, NZ, True
    )
    inst._driver.step_radiation(direct_state, inst.sfc_state, date)
    out_direct = marshal.extract_outputs(direct_state, NX, NY, NZ, True)

    assert set(out_bridge) == set(out_direct)
    for name in out_bridge:
        np.testing.assert_array_equal(
            out_bridge[name], out_direct[name], err_msg=f"bridge vs direct: {name}"
        )
        assert np.all(np.isfinite(out_bridge[name])), name

    geos_rrtmgp.geos_rrtmgp_finalize()
    assert geos_rrtmgp.GEOS_RRTMGP is None


def test_run_before_init_raises():
    from pyshield.geos_bridge import geos_rrtmgp

    geos_rrtmgp.geos_rrtmgp_finalize()
    with pytest.raises(RuntimeError):
        geos_rrtmgp.geos_rrtmgp_run({}, {}, datetime.datetime(2020, 1, 1))

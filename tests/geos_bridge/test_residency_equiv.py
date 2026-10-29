"""Device residency == host container path, bit-for-bit (data-residency lever).

The driver keeps the big optics (tau/ssa/g and the LW layer/level Planck
sources) on the backend device from gas optics, through cloud `add_to`, into the
solver -- instead of writing them back to the host pyRTE container and letting
the solver re-upload them. That removes the PCIe round-trip that dominates the
GPU cost; it must NOT change any number, because it is the same arrays moved the
same way, only without the host hop.

This test proves the equivalence: it runs the SAME synthetic state through
`step_radiation` twice on the one driver -- once resident (ON, the driver
default) and once with `set_radiation_residency(False)` (the host path) -- on
fresh independent states, and asserts every output flux matches bit-for-bit.
Clouds are NON-zero here so the cloud `add_to` device path (the all-sky combine)
is exercised, not just the clear-sky gas-only solve.

On CPU the backend array module is numpy, so the "device tiles" ARE the host
numpy arrays and the two paths differ only by data-movement (transpose/reshape,
no arithmetic): the result is bit-identical, and `assert_array_equal` is the
right check. On a GPU backend the same assertion holds (both legs run on the
same backend); run it there to confirm the cupy plumbing of the resident path.

Run on Discover inside the fork-a venv with XDG_CACHE_HOME set:
    pytest tests/geos_bridge/test_residency_equiv.py -q
    RTE_TEST_BACKEND=dace:gpu pytest tests/geos_bridge/test_residency_equiv.py -q
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
        pytest.skip(
            "XDG_CACHE_HOME not set; the driver builds GasOptics/CloudOptics "
            "which fetch the RRTMGP + cloud-optics coeff files through pyRTE's "
            "pooch cache. Set XDG_CACHE_HOME to a nobackup path and rerun."
        )
    try:
        import pyrte_rrtmgp  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pyrte_rrtmgp not importable: {exc}")


def _cloudy_profile():
    """A plausible top-down column with NON-zero liquid + ice cloud.

    The cloud water paths drive the cloud-optics LUT and so the `add_to`
    combine; the values are modest and physical (the LUT clips effective
    radius/diameter to its table range regardless, so the combine always runs).
    """
    p_lev = np.geomspace(50.0, 1.0e5, NZ + 1).astype(Float)
    p_lay = (0.5 * (p_lev[:-1] + p_lev[1:])).astype(Float)
    t_lay = np.full(NZ, 250.0, dtype=Float)

    def tile_lev(v):
        return np.tile(v, (NCOL, 1)).astype(Float)

    # Put cloud in a mid-tropospheric band so rel/rei land in a sane range.
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


def test_residency_matches_host(tmp_path):
    _require_env()

    from pyshield.geos_bridge import geos_rrtmgp, marshal
    from pyshield.radiation.state import RTE_RRTMGPState

    date = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone.utc)
    inputs, sfc_inputs, p_lev = _cloudy_profile()

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
    driver = inst._driver

    def run():
        st = RTE_RRTMGPState.init_zeros(inst.quantity_factory)
        marshal.marshal_inputs(
            st, inst.sfc_state, inputs, sfc_inputs, NX, NY, NZ, True
        )
        driver.step_radiation(st, inst.sfc_state, date)
        return marshal.extract_outputs(st, NX, NY, NZ, True)

    # Driver default is resident; run resident first, then the host path.
    assert driver._resident is True
    out_res = run()
    driver.set_radiation_residency(False)
    out_host = run()

    assert set(out_res) == set(out_host)
    for name in out_res:
        assert np.all(np.isfinite(out_res[name])), name
        np.testing.assert_array_equal(
            out_res[name], out_host[name],
            err_msg=f"resident vs host: {name}",
        )

    geos_rrtmgp.geos_rrtmgp_finalize()

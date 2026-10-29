"""Profile one radiation step to split the per-call time by component.

After the gather-on-GPU change, step_radiation is ~20 s/call on the A100 (down
from 90). This runs cProfile around one warm call and prints the radiation
functions by cumulative time, so we see where the 20 s goes -- SW gas optics vs
LW gas optics vs cloud optics vs the four solves vs loop/launch overhead --
before choosing the next lever.

Reads results off the device each iteration (view reads force completion), so
cProfile's cumulative time tracks wall time per component closely. A final
device sync catches any tail.

Run on a gpu_a100 node (GPU env per ndsl-env-on-discover):

    GEOS_RRTMGP_BACKEND=gpu FV3_DACEMODE=BuildAndRun \
        python tests/geos_bridge/profile_step_radiation.py
"""

import cProfile
import datetime
import os
import pstats
import tempfile
from pathlib import Path

import numpy as np

os.environ.pop("GEOS_RRTMGP_DEBUG", None)

NX, NY, NZ, NHALO = 45, 90, 91, 3
NCOL = NX * NY


def _synthetic(nz):
    p_lev = np.geomspace(50.0, 1.0e5, nz + 1)
    p_lay = 0.5 * (p_lev[:-1] + p_lev[1:])
    t_lay = np.linspace(230.0, 290.0, nz)

    def tile(v):
        return np.tile(v, (NCOL, 1))

    inputs = {
        "prsi": tile(p_lev), "prsl": tile(p_lay), "tlyr": tile(t_lay),
        "tsfc": np.full(NCOL, 290.0),
        "qvapor": np.full((NCOL, nz), 1.0e-3), "qo3mr": np.full((NCOL, nz), 1.0e-7),
        "co2": np.full((NCOL, nz), 400.0e-6),
        "qliquid": np.zeros((NCOL, nz)), "qice": np.zeros((NCOL, nz)),
        "qcld": np.zeros((NCOL, nz)),
    }
    sfc_inputs = {"tsfc": np.full(NCOL, 290.0), "islmsk": np.zeros(NCOL)}
    return inputs, sfc_inputs, p_lev


def _config(date, tmp):
    from pyshield.radiation import RTE_RRTMGPConfig
    return RTE_RRTMGPConfig(
        deltsw=3600.0, delt_rad=3600.0, date=date, fhswr=1.0, fhlwr=1.0,
        isolar=10, icmphys=4, ico2flg=0, ioznflg=1, ictmflg=-1, ialbflg=-1,
        iemsflg=0, ldisable_radiation_quasi_sea_ice=False,
        solar_constant_file=Path("global_solarconstant_noaa_an.txt"),
        input_dir=Path(tmp), aerosol_file=Path(tmp), sollat=0.0, nstp=6,
        ivflip=1, lcnorm=False, lcrick=False, gfs_cloud_overlap=False,
    )


def _sync():
    try:
        import cupy
        cupy.cuda.Device().synchronize()
    except Exception:
        pass


def main():
    from pyshield.geos_bridge import geos_rrtmgp
    date = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone.utc)
    tmp = tempfile.mkdtemp()
    inputs, sfc_inputs, p_lev = _synthetic(NZ)
    lons = np.zeros(NCOL)
    lats = np.zeros(NCOL)
    sigma = p_lev / p_lev[-1]

    geos_rrtmgp.geos_rrtmgp_finalize()
    geos_rrtmgp.geos_rrtmgp_init(
        config=_config(date, tmp), nx=NX, ny=NY, nz=NZ,
        lons=lons, lats=lats, sigma=sigma, nhalo=NHALO, backend=None, top_at_1=True,
    )
    geos_rrtmgp.geos_rrtmgp_run(inputs, sfc_inputs, date, top_at_1=True)  # warm
    _sync()

    pr = cProfile.Profile()
    pr.enable()
    geos_rrtmgp.geos_rrtmgp_run(inputs, sfc_inputs, date, top_at_1=True)
    _sync()
    pr.disable()

    print("=== radiation functions by cumulative time (one call) ===")
    st = pstats.Stats(pr).sort_stats("cumulative")
    st.print_stats("pyshield", 30)
    print("=== top by total (self) time, radiation code ===")
    pstats.Stats(pr).sort_stats("tottime").print_stats("pyshield", 20)
    print("=== top overall by self time (incl cupy/numpy/gt4py internals) ===")
    pstats.Stats(pr).sort_stats("tottime").print_stats(30)

    geos_rrtmgp.geos_rrtmgp_finalize()


if __name__ == "__main__":
    main()

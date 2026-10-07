"""Env-driven RTE_RRTMGPConfig + grid construction for the radiation CFFI bridge.

The Fortran side passes only grid dims, the MPI comm, orientation, a timestep and
the date (by value). Everything else -- the physics-flag config, the coefficient /
solar / aerosol file paths, and the per-column lon/lat and sigma arrays -- is built
here from environment variables, mirroring gtFV3's env-driven pattern
(``GTFV3_BACKEND`` / ``GTFV3_NAMELIST`` in ``geos_gtfv3_init``). This keeps the
C ABI small (scalars + field arrays) and keeps run configuration out of Fortran.

Defaults match the validated Phase-0 bridge test config
(``tests/geos_bridge/test_geos_rrtmgp_bridge.py``): isolar=10, icmphys=4, ico2flg=0,
ioznflg=1, ictmflg=-1, ialbflg=-1, iemsflg=0. With those flags ``__init__`` reads NO
``input_dir`` text files (sol/sfc/gas init all take fixed-value early returns), so
``GEOS_RRTMGP_INPUT_DIR`` may point at any readable directory.
"""

import datetime
import os
from pathlib import Path

import numpy as np


def _f(name, default):
    return float(os.environ.get(name, default))


def _i(name, default):
    return int(os.environ.get(name, default))


def config_from_env(dt: float, date: datetime.datetime):
    """Build an RTE_RRTMGPConfig from env vars + the passed timestep and date."""
    from pyshield.radiation import RTE_RRTMGPConfig

    input_dir = Path(os.environ.get("GEOS_RRTMGP_INPUT_DIR", "."))
    return RTE_RRTMGPConfig(
        deltsw=float(dt),
        delt_rad=float(dt),
        date=date,
        fhswr=float(dt),
        fhlwr=float(dt),
        isolar=_i("GEOS_RRTMGP_ISOLAR", 10),
        icmphys=_i("GEOS_RRTMGP_ICMPHYS", 4),
        ico2flg=_i("GEOS_RRTMGP_ICO2FLG", 0),
        ioznflg=_i("GEOS_RRTMGP_IOZNFLG", 1),
        ictmflg=_i("GEOS_RRTMGP_ICTMFLG", -1),
        ialbflg=_i("GEOS_RRTMGP_IALBFLG", -1),
        iemsflg=_i("GEOS_RRTMGP_IEMSFLG", 0),
        ldisable_radiation_quasi_sea_ice=False,
        solar_constant_file=Path(
            os.environ.get(
                "GEOS_RRTMGP_SOLAR_FILE", "global_solarconstant_noaa_an.txt"
            )
        ),
        input_dir=input_dir,
        aerosol_file=Path(os.environ.get("GEOS_RRTMGP_AEROSOL_FILE", str(input_dir))),
        sollat=_f("GEOS_RRTMGP_SOLLAT", 0.0),
        nstp=_i("GEOS_RRTMGP_NSTP", 6),
        ivflip=_i("GEOS_RRTMGP_IVFLIP", 1),
        lcnorm=False,
        lcrick=False,
        gfs_cloud_overlap=False,
    )


def grid_from_env(nx: int, ny: int, nz: int):
    """Build whole-tile lons/lats (ncol,) and sigma (nz+1,) for the driver.

    If ``GEOS_RRTMGP_{LONS,LATS,SIGMA}_NPY`` point at ``.npy`` files they are loaded
    (so a real grid can be supplied without changing the ABI); otherwise sensible
    standalone defaults are used: lon=lat=0 and a geometric top-down sigma. A real
    GEOS wiring (Phase 2) would instead pass the model grid in through the ABI.
    """
    ncol = nx * ny

    def _load(env, default):
        path = os.environ.get(env)
        return np.load(path).astype(np.float64) if path else default

    lons = _load("GEOS_RRTMGP_LONS_NPY", np.zeros(ncol, dtype=np.float64))
    lats = _load("GEOS_RRTMGP_LATS_NPY", np.zeros(ncol, dtype=np.float64))
    sigma_path = os.environ.get("GEOS_RRTMGP_SIGMA_NPY")
    if sigma_path:
        sigma = np.load(sigma_path).astype(np.float64)
    else:
        p = np.geomspace(50.0, 1.0e5, nz + 1)
        sigma = (p / p[-1]).astype(np.float64)
    return lons.reshape(ncol), lats.reshape(ncol), sigma

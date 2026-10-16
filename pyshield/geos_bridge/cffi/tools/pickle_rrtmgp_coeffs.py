"""Pre-serialize the RRTMGP coefficient files to pickle sidecars (run on login).

GEOS links its own HDF5 statically, and inside the running model neither the
HDF5-based netcdf4 engine nor the scipy netCDF-3 engine can read the coefficient
files reliably (HDF5 symbol clash / an in-process scipy failure). The CFFI bridge
(geos_rrtmgp_interface.py) therefore reads a plain pickle of each coefficient
dataset instead of the .nc file, so the embedded interpreter runs no file-format
library at all.

This script builds those sidecars. Run it on a LOGIN node, in the Fork-A venv,
where HDF5/netCDF work normally. For each coefficient file ``X.nc`` under the
given directory it writes ``X.nc.pkl`` next to it (that exact name is what the
bridge looks for: ``<path passed to xr.load_dataset> + ".pkl"``).

Usage:
    python -m pyshield.geos_bridge.cffi.tools.pickle_rrtmgp_coeffs \
        /discover/nobackup/$USER/fork-a/.cache/pyrte_rrtmgp/rrtmgp-data-1.9

With no argument it uses the pyRTE cache directory for the current
XDG_CACHE_HOME. Only the top-level ``rrtmgp-*.nc`` coefficient files are
pickled (not the examples/reference data, which the model never reads).
"""

import glob
import os
import pickle
import sys

import xarray as xr


def _coeff_dir() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    # Mirror pyRTE's cache location.
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    # The versioned subdir name can change; pick the newest rrtmgp-data-* dir.
    candidates = sorted(glob.glob(os.path.join(base, "pyrte_rrtmgp", "rrtmgp-data-*")))
    if not candidates:
        raise SystemExit(
            f"no rrtmgp-data-* directory under {base}/pyrte_rrtmgp — "
            "pass the directory explicitly"
        )
    return candidates[-1]


def main() -> None:
    d = _coeff_dir()
    files = sorted(glob.glob(os.path.join(d, "rrtmgp-*.nc")))
    if not files:
        raise SystemExit(f"no rrtmgp-*.nc coefficient files in {d}")
    for f in files:
        out = f + ".pkl"
        ds = xr.load_dataset(f).load()  # read on login (HDF5/netCDF work here)
        with open(out, "wb") as fh:
            pickle.dump(ds, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"pickled {os.path.basename(out)}  ({os.path.getsize(out) // (1024*1024)} MB)")
    print(f"done: {len(files)} coefficient files pickled in {d}")


if __name__ == "__main__":
    main()

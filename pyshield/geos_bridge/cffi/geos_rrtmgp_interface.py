"""CFFI embedding generator for the GEOS <-> PySHiELD radiation bridge (Phase 1).

Running ``python geos_rrtmgp_interface.py`` compiles
``libgeos_rrtmgp_interface_py.so`` -- a shared object that embeds CPython and
exports the C symbols the C shim (geos_rrtmgp_interface.c) calls. This mirrors the
merged gtFV3 bridge's ``geos_gtfv3_interface.py``
(``gpu-analysis/FVdycoreCubed_GridComp/geos-gtfv3/geos_gtfv3_interface.py``):

* ``ffi.embedding_api(header)``   -- declares the exported C symbols,
* ``ffi.set_source(...)``         -- the generated C includes the header,
* ``ffi.embedding_init_code(src)``-- Python run at .so load; its ``@ffi.def_extern``
                                     bodies are the init/run/finalize trampolines,
* ``ffi.compile(target=...)``     -- builds the .so.

The trampolines reconstruct the MPI communicator (same cast trick as gtFV3) and
drive the Phase-0 pure-Python entry
``pyshield.geos_bridge.geos_rrtmgp.{geos_rrtmgp_init,geos_rrtmgp_run,
geos_rrtmgp_finalize}``. Config / backend / grid are built Python-side from env
(``pyshield.geos_bridge.cffi.geos_rrtmgp_env``); R8 marshalling is done by
``RadiationR8Conversion`` (8-byte copy-back, not gtFV3's 4-byte float32 bug).

The source string uses ``__MPI_COMM_T__`` / ``__TMPFILEBASE__`` placeholders filled
with ``str.replace`` (NOT ``str.format``), because the embedded Python contains
dict/f-string braces that would collide with ``format``.
"""

import cffi
from mpi4py import MPI

TMPFILEBASE = "geos_rrtmgp_interface_py"

ffi = cffi.FFI()

# MPI_Comm is either an int or an opaque pointer depending on the MPI build.
if MPI._sizeof(MPI.Comm) == ffi.sizeof("int"):
    _mpi_comm_t = "int"
else:
    _mpi_comm_t = "void*"

# ---------------------------------------------------------------------------
# C header: the three exported symbols. Field arrays are double* (R8).
# ---------------------------------------------------------------------------
header = """
extern void geos_rrtmgp_interface_py_init(
    __MPI_COMM_T__ comm_c,
    int nx, int ny, int nz, int nhalo, int top_at_1,
    int yy, int mm, int dd, int hh, int mn, int sc, double dt
);
extern void geos_rrtmgp_interface_py(
    __MPI_COMM_T__ comm_c,
    int nx, int ny, int nz, int top_at_1,
    int yy, int mm, int dd, int hh, int mn, int sc,
    // input fields
    double* prsi, double* prsl, double* tlyr, double* tsfc,
    double* qvapor, double* qo3mr, double* co2,
    double* qliquid, double* qice, double* qcld,
    // surface inputs
    double* sfc_tsfc, double* islmsk,
    // outputs
    double* flwu, double* flwd, double* fswu, double* fswd,
    double* flwu_clr, double* flwd_clr, double* fswu_clr, double* fswd_clr,
    double* fswn,
    double* hrtlw, double* hrtsw, double* hrtlw_clr, double* hrtsw_clr
);
extern void geos_rrtmgp_interface_py_finalize();
""".replace(
    "__MPI_COMM_T__", _mpi_comm_t
)

# ---------------------------------------------------------------------------
# Embedding init code: the Python run inside the .so. The @ffi.def_extern bodies
# are the init/run/finalize trampolines.
# ---------------------------------------------------------------------------
source = r'''
import datetime

# Make the Fork-A venv importable to this EMBEDDED interpreter. The .so embeds
# GEOSpyD's base libpython, whose default sys.path has mpi4py/numpy but NOT the
# venv's pyshield/ndsl/pyRTE. pyshield is an editable install, so its location
# lives in a .pth file that only site processing reads -- a bare sys.path entry
# is not enough. Read a colon-separated list from GEOS_RRTMGP_PYTHONPATH (then
# PYTHONPATH) and site.addsitedir each entry so .pth files are processed too.
# This does not depend on the embedded Py_Initialize honoring PYTHONPATH.
import os as _os
import site as _site
import sys as _sys

_rad_paths = (
    _os.environ.get("GEOS_RRTMGP_PYTHONPATH", "")
    + ":"
    + _os.environ.get("PYTHONPATH", "")
)
for _p in _rad_paths.split(":"):
    _p = _p.strip()
    if not _p:
        continue
    if _p not in _sys.path:
        _sys.path.insert(0, _p)  # ahead of GEOSpyD base so the venv's numpy/ndsl win
    _site.addsitedir(_p)         # also process .pth (the editable pyshield pointer)

# GEOS links its own HDF5 STATICALLY, so those symbols are global in the running
# process and would override the venv netCDF4's bundled HDF5 (a different version)
# -> "NetCDF: HDF error" when pyRTE reads the RRTMGP coefficient files. Load
# subsequent C extensions (netCDF4 -> libnetcdf -> libhdf5) with RTLD_DEEPBIND so
# each prefers its OWN dependency's symbols over the host's. Set before any import
# that pulls in netCDF4 (pyRTE's xr.load_dataset at driver init). Can be disabled
# with GEOS_RRTMGP_NO_DEEPBIND=1 if it ever interacts badly with another extension.
if hasattr(_os, "RTLD_DEEPBIND") and _os.environ.get("GEOS_RRTMGP_NO_DEEPBIND") != "1":
    _sys.setdlopenflags(_sys.getdlopenflags() | _os.RTLD_DEEPBIND)

# Read the RRTMGP coefficient files with xarray's scipy backend (pure Python, no
# HDF5). GEOS links its own HDF5 statically; any HDF5 the embedded interpreter
# loads via the netcdf4/h5netcdf engines clashes with it ("NetCDF: HDF error").
# The coefficient files are pre-converted to netCDF-3 (which scipy reads without
# HDF5). Force the engine for xarray opens that do not name one. pyRTE calls a
# plain xr.load_dataset(path), so this steers it. Override with
# GEOS_RRTMGP_NETCDF_ENGINE ("auto" leaves xarray's default, e.g. for debugging).
_rad_engine = _os.environ.get("GEOS_RRTMGP_NETCDF_ENGINE", "scipy")
if _rad_engine and _rad_engine.lower() != "auto":
    import xarray as _xr

    _rad_orig_open = _xr.open_dataset
    _rad_orig_load = _xr.load_dataset

    def _rad_open(*a, **k):
        k.setdefault("engine", _rad_engine)
        return _rad_orig_open(*a, **k)

    def _rad_load(*a, **k):
        k.setdefault("engine", _rad_engine)
        return _rad_orig_load(*a, **k)

    _xr.open_dataset = _rad_open
    _xr.load_dataset = _rad_load

from __TMPFILEBASE__ import ffi
from mpi4py import MPI

from pyshield.geos_bridge.geos_rrtmgp import (
    geos_rrtmgp_init,
    geos_rrtmgp_run,
    geos_rrtmgp_finalize,
)
from pyshield.geos_bridge.cffi.radiation_f_py_conversion import RadiationR8Conversion
from pyshield.geos_bridge.cffi.geos_rrtmgp_env import config_from_env, grid_from_env

# Held across calls (built once at init, reused each run), like gtFV3's singleton.
_CONV = None
_TOP_AT_1 = True


def _comm_c_to_py(comm_c):
    # comm_c -> comm_py (same cast trick as geos_gtfv3_interface.py).
    comm_py = MPI.Intracomm()  # internal MPI_Comm handle is MPI_COMM_NULL
    comm_ptr = MPI._addressof(comm_py)
    comm_ptr = ffi.cast('__MPI_COMM_T__*', comm_ptr)
    comm_ptr[0] = comm_c
    return comm_py


@ffi.def_extern()
def geos_rrtmgp_interface_py_init(
    comm_c,
    nx, ny, nz, nhalo, top_at_1,
    yy, mm, dd, hh, mn, sc, dt,
):
    global _CONV, _TOP_AT_1
    # comm reconstructed for symmetry / future multi-rank; the single-tile driver
    # does not currently use it.
    _comm_c_to_py(comm_c)

    _TOP_AT_1 = bool(top_at_1)
    date = datetime.datetime(
        yy, mm, dd, hh, mn, sc, tzinfo=datetime.timezone.utc
    )
    config = config_from_env(dt, date)
    lons, lats, sigma = grid_from_env(nx, ny, nz)

    geos_rrtmgp_init(
        config=config,
        nx=nx,
        ny=ny,
        nz=nz,
        lons=lons,
        lats=lats,
        sigma=sigma,
        nhalo=nhalo,
        backend=None,          # geos_rrtmgp_init reads GEOS_RRTMGP_BACKEND env
        top_at_1=_TOP_AT_1,
    )
    _CONV = RadiationR8Conversion(nx, ny, nz)


@ffi.def_extern()
def geos_rrtmgp_interface_py(
    comm_c,
    nx, ny, nz, top_at_1,
    yy, mm, dd, hh, mn, sc,
    prsi, prsl, tlyr, tsfc,
    qvapor, qo3mr, co2,
    qliquid, qice, qcld,
    sfc_tsfc, islmsk,
    flwu, flwd, fswu, fswd,
    flwu_clr, flwd_clr, fswu_clr, fswd_clr,
    fswn,
    hrtlw, hrtsw, hrtlw_clr, hrtsw_clr,
):
    if _CONV is None:
        raise RuntimeError("[GEOS RRTMGP] run before init")

    # Marshal Fortran R8 buffers -> whole-tile numpy (per RadiationR8Conversion).
    inputs = _CONV.inputs_from_fortran({
        "prsi": prsi, "prsl": prsl, "tlyr": tlyr, "tsfc": tsfc,
        "qvapor": qvapor, "qo3mr": qo3mr, "co2": co2,
        "qliquid": qliquid, "qice": qice, "qcld": qcld,
    })
    sfc_inputs = _CONV.sfc_from_fortran({
        "tsfc": sfc_tsfc, "islmsk": islmsk,
    })

    date = datetime.datetime(
        yy, mm, dd, hh, mn, sc, tzinfo=datetime.timezone.utc
    )
    outputs = geos_rrtmgp_run(inputs, sfc_inputs, date, top_at_1=bool(top_at_1))

    # Copy whole-tile outputs back into the Fortran buffers (8-byte R8 stride).
    _CONV.copy_outputs(outputs, {
        "flwu": flwu, "flwd": flwd, "fswu": fswu, "fswd": fswd,
        "flwu_clr": flwu_clr, "flwd_clr": flwd_clr,
        "fswu_clr": fswu_clr, "fswd_clr": fswd_clr,
        "fswn": fswn,
        "hrtlw": hrtlw, "hrtsw": hrtsw,
        "hrtlw_clr": hrtlw_clr, "hrtsw_clr": hrtsw_clr,
    })


@ffi.def_extern()
def geos_rrtmgp_interface_py_finalize():
    global _CONV
    geos_rrtmgp_finalize()
    _CONV = None
'''.replace(
    "__TMPFILEBASE__", TMPFILEBASE
).replace(
    "__MPI_COMM_T__", _mpi_comm_t
)

with open(TMPFILEBASE + ".h", "w") as f:
    f.write(header)

ffi.embedding_api(header)

source_header = r'''#include "{}.h"'''.format(TMPFILEBASE)
ffi.set_source(TMPFILEBASE, source_header)

ffi.embedding_init_code(source)
ffi.compile(target="lib" + TMPFILEBASE + ".so", verbose=True)

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

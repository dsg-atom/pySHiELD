"""Time one radiation step at the GEOS c180-L91 per-rank tile, GPU vs CPU.

Answers the throughput question the correctness tests do not: is a radiation
step actually faster on the A100 than on the CPU, and how much of the cost is
compute vs the host<->device transfer at the bridge boundary.

It drives the real bridge singleton (``geos_rrtmgp_init`` / ``geos_rrtmgp_run``)
at the single-tile size one MPI rank owns in the c180 L91 run: ncol = nx*ny =
45*90 = 4050 columns, nz = 91 layers. It reports three numbers per backend:

* first/compile call  -- DaCe BuildAndRun compiles the stencils here (excluded);
* full geos_rrtmgp_run per call -- marshal in -> step_radiation -> marshal out,
  i.e. exactly what GEOS pays each radiation step (includes the PCIe transfer);
* step_radiation only per call -- state already resident, so this is the pure
  compute; (full - step) is the per-call marshalling / transfer overhead.

On GPU the device is synchronized before each clock stop so the timing is real
execution, not an async kernel launch.

Run on a gpu_a100 node inside the fork-a venv, once per backend (same node, so
the CPU dacecpu build uses that node's EPYC cores and the GPU build uses the
A100 -- an apples-to-apples per-rank comparison). XDG_CACHE_HOME must be set
(pyRTE coeff fetch), as for the bridge test:

    GEOS_RRTMGP_BACKEND=cpu FV3_DACEMODE=BuildAndRun BENCH_ITERS=3 \
        python tests/geos_bridge/bench_step_radiation.py
    GEOS_RRTMGP_BACKEND=gpu FV3_DACEMODE=BuildAndRun BENCH_ITERS=10 \
        python tests/geos_bridge/bench_step_radiation.py

The CPU backend is slow per call (minutes), so use a small BENCH_ITERS for it.
"""

import datetime
import os
import tempfile
import time
from pathlib import Path

import numpy as np

# Keep the benchmark output clean even if the model env exported this.
os.environ.pop("GEOS_RRTMGP_DEBUG", None)

# Single-tile size one rank owns in the c180 L91 run.
NX, NY, NZ, NHALO = 45, 90, 91, 3
NCOL = NX * NY
NITER = int(os.environ.get("BENCH_ITERS", "5"))


def _synthetic(nz):
    """A plausible top-down column profile, tiled over all NCOL columns."""
    p_lev = np.geomspace(50.0, 1.0e5, nz + 1)
    p_lay = 0.5 * (p_lev[:-1] + p_lev[1:])
    t_lay = np.linspace(230.0, 290.0, nz)

    def tile(v):
        return np.tile(v, (NCOL, 1))

    inputs = {
        "prsi": tile(p_lev),
        "prsl": tile(p_lay),
        "tlyr": tile(t_lay),
        "tsfc": np.full(NCOL, 290.0),
        "qvapor": np.full((NCOL, nz), 1.0e-3),
        "qo3mr": np.full((NCOL, nz), 1.0e-7),
        "co2": np.full((NCOL, nz), 400.0e-6),
        "qliquid": np.zeros((NCOL, nz)),
        "qice": np.zeros((NCOL, nz)),
        "qcld": np.zeros((NCOL, nz)),
    }
    sfc_inputs = {"tsfc": np.full(NCOL, 290.0), "islmsk": np.zeros(NCOL)}
    return inputs, sfc_inputs, p_lev


def _config(date, tmp):
    from pyshield.radiation import RTE_RRTMGPConfig

    # Same fixed-value config as the bridge test (no input_dir text files read).
    return RTE_RRTMGPConfig(
        deltsw=3600.0, delt_rad=3600.0, date=date, fhswr=1.0, fhlwr=1.0,
        isolar=10, icmphys=4, ico2flg=0, ioznflg=1, ictmflg=-1, ialbflg=-1,
        iemsflg=0, ldisable_radiation_quasi_sea_ice=False,
        solar_constant_file=Path("global_solarconstant_noaa_an.txt"),
        input_dir=Path(tmp), aerosol_file=Path(tmp), sollat=0.0, nstp=6,
        ivflip=1, lcnorm=False, lcrick=False, gfs_cloud_overlap=False,
    )


def _sync(is_gpu):
    """Wait for the device so GPU timing is execution, not async launch."""
    if not is_gpu:
        return
    try:
        import cupy

        cupy.cuda.Device().synchronize()
    except Exception:
        pass


def main():
    from pyshield.geos_bridge import geos_rrtmgp, marshal

    backend = os.environ.get("GEOS_RRTMGP_BACKEND", "numpy")
    is_gpu = "gpu" in backend.lower()
    date = datetime.datetime(2020, 1, 1, 12, tzinfo=datetime.timezone.utc)
    tmp = tempfile.mkdtemp()
    inputs, sfc_inputs, p_lev = _synthetic(NZ)
    lons = np.zeros(NCOL)
    lats = np.zeros(NCOL)
    sigma = p_lev / p_lev[-1]

    geos_rrtmgp.geos_rrtmgp_finalize()
    geos_rrtmgp.geos_rrtmgp_init(
        config=_config(date, tmp), nx=NX, ny=NY, nz=NZ,
        lons=lons, lats=lats, sigma=sigma, nhalo=NHALO,
        backend=None, top_at_1=True,  # reads GEOS_RRTMGP_BACKEND
    )

    # First call compiles the stencils (DaCe BuildAndRun); exclude it.
    t0 = time.perf_counter()
    geos_rrtmgp.geos_rrtmgp_run(inputs, sfc_inputs, date, top_at_1=True)
    _sync(is_gpu)
    compile_s = time.perf_counter() - t0

    # Full per-call cost: marshal in -> step_radiation -> marshal out.
    full = []
    for _ in range(NITER):
        t0 = time.perf_counter()
        geos_rrtmgp.geos_rrtmgp_run(inputs, sfc_inputs, date, top_at_1=True)
        _sync(is_gpu)
        full.append(time.perf_counter() - t0)
    full = np.array(full)

    # step_radiation only (state already marshalled/resident) = pure compute.
    inst = geos_rrtmgp.GEOS_RRTMGP
    marshal.marshal_inputs(
        inst.state, inst.sfc_state, inputs, sfc_inputs, NX, NY, NZ, True
    )
    step = []
    for _ in range(NITER):
        t0 = time.perf_counter()
        inst._driver.step_radiation(inst.state, inst.sfc_state, date)
        _sync(is_gpu)
        step.append(time.perf_counter() - t0)
    step = np.array(step)

    print(f"[BENCH] backend={backend} ncol={NCOL} nz={NZ} iters={NITER}")
    print(f"[BENCH] first/compile call : {compile_s:8.3f} s")
    print(f"[BENCH] full run  per call : min={full.min():.4f} "
          f"mean={full.mean():.4f} max={full.max():.4f} s")
    print(f"[BENCH] step_rad  per call : min={step.min():.4f} "
          f"mean={step.mean():.4f} max={step.max():.4f} s")
    print(f"[BENCH] marshalling/xfer   : ~{full.mean() - step.mean():.4f} "
          f"s/call (full - step)")

    geos_rrtmgp.geos_rrtmgp_finalize()


if __name__ == "__main__":
    main()

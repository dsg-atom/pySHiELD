"""Prototype measurement: longwave solver, 256-g-point host loop vs folded.

Runs the current host-loop solver (LWNoScatSolverGT4Py) and the folded-domain
prototype (LWNoScatSolverBatched) on the SAME synthetic inputs at the c180-L91
per-rank size (ncol = 45*90 = 4050, nz = 91, ngpt = 256). It:

  1. checks the two agree (the folded reduction must reproduce the loop), and
  2. times both and reports peak GPU memory,

so one run answers whether folding g-points into the domain is faster and fits
in memory -- the gate for the full gas-optics rewrite.

Run on a gpu_a100 node in the fork-a venv (GPU env per ndsl-env-on-discover):

    GEOS_RRTMGP_BACKEND unused here; pass the backend via RTE_TEST_BACKEND-style
    knobs below. Simplest:

    FV3_DACEMODE=BuildAndRun BENCH_BACKEND=gpu BATCH_CHUNK=256 \
        python tests/rte_solver/bench_lw_solver_batched.py

Knobs (env): BENCH_BACKEND = gpu|cpu (default gpu); BATCH_CHUNK (default 256,
must divide 256); LOOP_ITERS (default 2, the loop solver is slow); BATCH_ITERS
(default 5).
"""

import os
import time

import numpy as np

NX, NY, NZ, NGPT = 45, 90, 91, 256
NCOL = NX * NY
NLEV = NZ + 1
CHUNK = int(os.environ.get("BATCH_CHUNK", "256"))
LOOP_ITERS = int(os.environ.get("LOOP_ITERS", "2"))
BATCH_ITERS = int(os.environ.get("BATCH_ITERS", "5"))


def _backend():
    from ndsl.config import backend_gpu, backend_python
    name = os.environ.get("BENCH_BACKEND", "gpu").lower()
    return backend_gpu if name in ("gpu", "dace:gpu", "gt:gpu") else backend_python


def _is_gpu():
    return os.environ.get("BENCH_BACKEND", "gpu").lower().startswith("gpu") or \
        "gpu" in os.environ.get("BENCH_BACKEND", "gpu").lower()


def _sync():
    if not _is_gpu():
        return
    try:
        import cupy
        cupy.cuda.Device().synchronize()
    except Exception:
        pass


def _gpu_mem_gb():
    try:
        import cupy
        pool = cupy.get_default_memory_pool()
        return pool.used_bytes() / 1e9, pool.total_bytes() / 1e9
    except Exception:
        return float("nan"), float("nan")


def _inputs(rng):
    # Physical-ish, positive; exact values do not matter -- both solvers get the
    # same data, and the test is loop-result == folded-result.
    tau = rng.uniform(1e-4, 0.3, size=(NCOL, NZ, NGPT))
    lay = rng.uniform(0.5, 8.0, size=(NCOL, NZ, NGPT))
    lev = rng.uniform(0.5, 8.0, size=(NCOL, NLEV, NGPT))
    sfc_src = rng.uniform(1.0, 9.0, size=(NCOL, NGPT))
    sfc_emis = rng.uniform(0.9, 1.0, size=(NCOL, NGPT))
    return tau, lay, lev, sfc_src, sfc_emis


def main():
    from pyshield.radiation.lw_solver_gt4py import LWNoScatSolverGT4Py, to_host
    from pyshield.radiation.lw_solver_batched import LWNoScatSolverBatched

    backend = _backend()
    top_at_1 = True
    rng = np.random.default_rng(0)
    tau, lay, lev, sfc_src, sfc_emis = _inputs(rng)

    # looped solver wants the (nx, ny, nz, ngpt) tile; folded wants (ncol, ...).
    tau4 = tau.reshape(NX, NY, NZ, NGPT)
    lay4 = lay.reshape(NX, NY, NZ, NGPT)
    lev4 = lev.reshape(NX, NY, NLEV, NGPT)
    src3 = sfc_src.reshape(NX, NY, NGPT)
    emis3 = sfc_emis.reshape(NX, NY, NGPT)

    looped = LWNoScatSolverGT4Py(
        nx=NX, ny=NY, nz=NZ, ngpt=NGPT, backend=backend, top_at_1=top_at_1
    )
    batched = LWNoScatSolverBatched(
        ncol=NCOL, nz=NZ, ngpt=NGPT, backend=backend, top_at_1=top_at_1, chunk=CHUNK
    )

    # --- correctness: folded reduction must reproduce the host loop ----------
    lu, ld = looped.solve(tau4, lay4, lev4, src3, emis3)
    lu = to_host(lu).reshape(NCOL, NLEV)
    ld = to_host(ld).reshape(NCOL, NLEV)
    bu, bd = batched.solve(tau, lay, lev, sfc_src, sfc_emis)
    np.testing.assert_allclose(bu, lu, rtol=1e-10, atol=1e-10, err_msg="up")
    np.testing.assert_allclose(bd, ld, rtol=1e-10, atol=1e-10, err_msg="down")
    up_rel = np.abs(bu - lu).max() / max(np.abs(lu).max(), 1e-30)
    print(f"[LWBENCH] correctness OK  max rel diff up={up_rel:.2e}")

    # --- time the host loop --------------------------------------------------
    looped.solve(tau4, lay4, lev4, src3, emis3); _sync()  # warm
    tl = []
    for _ in range(LOOP_ITERS):
        t0 = time.perf_counter()
        looped.solve(tau4, lay4, lev4, src3, emis3); _sync()
        tl.append(time.perf_counter() - t0)
    tl = np.array(tl)

    # --- time the folded solver ----------------------------------------------
    batched.solve(tau, lay, lev, sfc_src, sfc_emis); _sync()  # warm
    tb = []
    for _ in range(BATCH_ITERS):
        t0 = time.perf_counter()
        batched.solve(tau, lay, lev, sfc_src, sfc_emis); _sync()
        tb.append(time.perf_counter() - t0)
    tb = np.array(tb)
    used, total = _gpu_mem_gb()

    print(f"[LWBENCH] backend={os.environ.get('BENCH_BACKEND','gpu')} "
          f"ncol={NCOL} nz={NZ} ngpt={NGPT} chunk={CHUNK}")
    print(f"[LWBENCH] host loop (256x) : min={tl.min():.4f} mean={tl.mean():.4f} s/call")
    print(f"[LWBENCH] folded domain    : min={tb.min():.4f} mean={tb.mean():.4f} s/call")
    print(f"[LWBENCH] speedup          : {tl.mean()/tb.mean():.1f}x")
    print(f"[LWBENCH] GPU mem pool      : used={used:.2f} GB reserved={total:.2f} GB")


if __name__ == "__main__":
    main()

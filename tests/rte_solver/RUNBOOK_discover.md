# Discover runbook — validate the LW clear-sky solver port

Validates `LWNoScatSolverGT4Py` (`pyshield/radiation/lw_solver.py` +
`lw_solver_gt4py.py`) against pyRTE's longwave ABSORPTION solve, to rtol 1e-10.

The port is written and committed (dsg-atom/PySHiELD `develop`, commit `78e6ca1`)
and its design is proven bit-exact against a NumPy oracle, but it has **not** run
through the gt4py frontend or been compared to pyRTE yet. That is what this does.

The test is self-contained: it rebuilds the RFMIP atmosphere through pyRTE and
reads `top_at_1` from the dataset. It needs **no staged restart data** — only the
fork-a venv, `XDG_CACHE_HOME`, and login-node network (first run downloads the
LW_G256 coeff file to the cache). The numpy backend is used, so **no GPU needed**;
a login or interactive node is fine.

Substitute your username if not `rlgill`.

---

## Step 0 — re-entry block (every fresh shell)

```bash
module load python/GEOSpyD/24.11.3-0/3.12 comp/gcc/13.2.0
export CC=gcc CXX=g++ FC=gfortran
export XDG_CACHE_HOME=/discover/nobackup/$USER/fork-a/.cache
source /discover/nobackup/$USER/fork-a/venv/bin/activate
cd /discover/nobackup/$USER/fork-a/PySHiELD
```

Sanity:
```bash
which python && python --version      # -> .../fork-a/venv/bin/python, Python 3.12.9
```
`XDG_CACHE_HOME` is required — without it pyRTE's coeff download hits
`Errno 122 Disk quota exceeded` on the home filesystem.

## Step 1 — update the fork clone to the committed port

```bash
git -C /discover/nobackup/$USER/fork-a/PySHiELD fetch origin
git -C /discover/nobackup/$USER/fork-a/PySHiELD checkout develop
git -C /discover/nobackup/$USER/fork-a/PySHiELD pull --ff-only origin develop
git -C /discover/nobackup/$USER/fork-a/PySHiELD log --oneline -1   # expect 78e6ca1 (or later)
```
(Discover clone is pull-only; never push from it.)

## Step 2 — run the validation test

```bash
pytest tests/rte_solver/test_lw_solver_gt4py.py -q
```
First run also compiles the gt4py stencils (needs the `gcc` module loaded above)
and downloads the coeff file (~a minute).

- **PASS** → the LW clear-sky solver is validated against pyRTE. Record the result.
  Next stage per the port order is the SW two-stream solver, then cloud/aerosol
  optics.
- **FAIL** → go to Step 3.

---

## Step 3 — diagnostics (only if Step 2 fails)

The most likely cause of a flux mismatch is the single-angle **quadrature**
(secant `D` and `weight`) baked into the solver defaults. The solver uses
`D_DEFAULT = 1/0.6096748751 ≈ 1.6401`, `weight = 1.0` (the nmus=1 Gauss-Jacobi-5
values from `mo_rte_lw.F90`). Confirm pyRTE uses the same.

### 3a — confirm `top_at_1` and the dataset orientation
```bash
python - <<'PY'
import os
os.environ.setdefault("XDG_CACHE_HOME", f"/discover/nobackup/{os.environ['USER']}/fork-a/.cache")
from pyrte_rrtmgp import rte
from pyrte_rrtmgp.examples import RFMIP_FILES, load_example_file
from pyrte_rrtmgp.rrtmgp import DEFAULT_GAS_MAPPING, GasOptics, GasOpticsFiles, create_default_mapping
from pyrte_rrtmgp.tests.test_rfmip_clear_sky import RFMIP_GAS_MAPPING
atm = load_example_file(RFMIP_FILES.ATMOSPHERE).isel(site=[0,25,50,75], expt=[0,6,12,17])
gm = {g: RFMIP_GAS_MAPPING[g] for g in DEFAULT_GAS_MAPPING if g in RFMIP_GAS_MAPPING}
lw = GasOptics(gas_optics_file=GasOpticsFiles.LW_G256).compute(
    atm.copy(deep=True), problem_type=rte.OpticsTypes.ABSORPTION,
    gas_name_map=gm, variable_mapping=create_default_mapping(), add_to_input=False)
print("top_at_1 =", bool(lw.attrs["top_at_1"]))
PY
```
The test already reads this value, so it should match — this is just to see it.

### 3b — read pyRTE's actual LW quadrature (don't guess — read the source)
```bash
PYRTE=$(python -c "import pyrte_rrtmgp, os; print(os.path.dirname(pyrte_rrtmgp.__file__))")
echo "$PYRTE"
grep -rniE "0.6096748751|gauss|secant|n_quad|n_gauss|weight|_ds\b|diff_sec" "$PYRTE" | grep -iE "lw|long|gauss|secant|weight|quad" | head -40
```
Compare the secant/weight pyRTE feeds its LW no-scattering solver to
`D_DEFAULT`/`WEIGHT_DEFAULT`. If RRTMGP there uses `n_gauss_angles=1`, expect the
same first Gauss-Jacobi node (`D ≈ 1.6401`, `weight = 1.0`).

### 3c — if the quadrature differs, pass the corrected values (no recoding)
`D` and `weight` are constructor parameters, so edit only the test call
(`tests/rte_solver/test_lw_solver_gt4py.py`, the `LWNoScatSolverGT4Py(...)` call)
to pass the confirmed `D=`/`weight=`, and rerun Step 2.

### 3d — other suspects, in order
1. **Quadrature** (3b) — most likely.
2. **Orientation** — if fluxes look vertically reversed, `top_at_1` wiring, not the
   value (3a confirms the value).
3. **Series branch** — if the mismatch is tiny and only in optically-thin layers,
   it's the `tau_thresh` series-expansion in `lw_source_noscat`; keep that math in
   float64 (watch the NumPy float32-promotion trap).
4. **Surface term** — this run forces emissivity 1.0 (black surface), so
   `flux_up(sfc) = sfc_src`; a surface-only mismatch points there.

---

## What this does NOT cover (later stages)
Jacobians, Tang rescaling, SW two-stream, and all-sky (cloud/aerosol) are separate
stages that attach to this solver without reworking it. This runbook is LW
clear-sky broadband flux only.

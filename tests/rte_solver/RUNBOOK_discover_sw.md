# Discover runbook — validate the SW two-stream solver port

Validates `SWTwoStreamSolverGT4Py` (`pyshield/radiation/sw_solver.py` +
`sw_solver_gt4py.py`) against pyRTE's shortwave `TWO_STREAM` solve, to rtol 1e-10.

Scope: `sw_solver_2stream` with do_broadband, clear-sky, has_dif_bc=FALSE
(inc_flux_dif=0) — the shortwave counterpart of the LW clear-sky runbook. The
port's full algorithm (cell properties + direct-beam sweep + the Shonk-Hogan
`adding` two-sweep + broadband accumulate) is proven bit-for-bit against a NumPy
transcription of the Fortran, both orientations, including nighttime columns.
What has **not** run off-Discover: the gt4py frontend compile, and the pyRTE
numeric comparison. That is what this does.

The numpy backend is used, so **no GPU needed**; a login/interactive node is fine.
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
`XDG_CACHE_HOME` is required (else the SW_G224 coeff download hits
`Errno 122 Disk quota exceeded`).

## Step 1 — update the fork clone to the committed port

```bash
git -C /discover/nobackup/$USER/fork-a/PySHiELD fetch origin
git -C /discover/nobackup/$USER/fork-a/PySHiELD checkout develop
git -C /discover/nobackup/$USER/fork-a/PySHiELD pull --ff-only origin develop
git -C /discover/nobackup/$USER/fork-a/PySHiELD log --oneline -1
```
(Discover clone is pull-only; never push from it.)

## Step 2 — CONFIRM the pyRTE two-stream field wiring (do this first — don't guess)

The test sets `sw_optics["mu0"]` and `sw_optics["surface_albedo"]` (plus the
`*_direct`/`*_diffuse` variants) and uses the gas-optics `toa_source` as
`inc_flux_dir`, following the driver (`rte_rrtmgp.py` step_radiation). Confirm the
installed pyRTE two-stream solve reads exactly those names and that mu0 is a
cosine:

```bash
PYRTE=$(python -c "import pyrte_rrtmgp, os; print(os.path.dirname(pyrte_rrtmgp.__file__))")
grep -rniE "sw_flux_dir|two_?stream|rte_sw_solver_2stream|toa_source|inc_flux_dir|surface_albedo(_direct|_diffuse)?|\bmu0\b|solar_zenith" "$PYRTE" | head -60
```
Check:
- which variable the solve pulls for the direct-beam TOA flux (expect `toa_source`),
- whether direct vs diffuse surface albedo are separate names,
- whether `mu0` is consumed as a cosine (the test feeds `cos(deg2rad(sza))`) and
  broadcast across layers,
- the name the solve emits for the direct flux (`sw_flux_dir` / `sw_flux_direct`
  / `sw_flux_down_direct`) — the test tries all three and only compares if present.

If a name differs, adjust the three `sw[...] = ...` assignments and/or the
`inc_dir`/`mu0`/`alb` extraction in `test_sw_solver_gt4py.py` to match (no stencil
change needed — the solver I/O is name-agnostic).

## Step 3 — run the validation test

```bash
pytest tests/rte_solver/test_sw_solver_gt4py.py -q
```
First run compiles the two orientation stencils (needs the `gcc` module) and
downloads SW_G224.

- **PASS** → the SW two-stream clear-sky solver is validated against pyRTE
  (rtol 1e-10, atol 1e-12 on up/down, and on direct if exposed). Record it.
- **FAIL** → Step 4.

---

## Step 4 — diagnostics (only if Step 3 fails)

Order of suspects:
1. **Field wiring** (Step 2) — most likely: `mu0` not a cosine, albedo
   direct/diffuse split, or `inc_flux_dir` not `toa_source`. Dump what the
   reference actually used vs what the test fed.
2. **Orientation** — if fluxes look vertically reversed, the `top_at_1` branch
   selection. The test reads `top_at_1` from the dataset, so this would be a
   stencil block-order bug, not a value bug.
3. **mu0 clamp / nighttime mask** — if only low-sun or polar columns are off,
   it's the `min_mu0` clamp or the `mu0 <= 0` source mask.
4. **Rdir/Tdir energy clamp** — if only optically-thick, high-ssa layers are off,
   it's the `max(0, min(Rdir, 1-Tnoscat))` / `max(0, min(Tdir, 1-Tnoscat-Rdir))`
   clamps.
5. **Direct vs total down** — `adding` returns diffuse `flux_dn` only; the test's
   broadband_dn adds `flux_dir`. A down-only mismatch that equals the direct flux
   points here.

---

## What this does NOT cover (later stages)
All-sky (cloud/aerosol SW optics) and heating-rate reduction. This runbook is SW
two-stream clear-sky broadband flux only.
```

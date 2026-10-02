# Discover runbook — validate the cloud-optics port (Build 2)

Validates `CloudOpticsGT4Py` (`pyshield/radiation/cloud_optics.py` +
`cloud_optics_gt4py.py`) against pyRTE's `CloudOptics.compute` + `rte.add_to`, to
rtol ~1e-10.

Scope: the LUT branch of the Fortran `ty_cloud_optics_rrtmgp%cloud_optics` (the
per-band linear-in-effective-radius lookup for liquid and ice, then the
1scl/2str combine) plus the by-band -> by-g-point optical-props increment
(`inc_1scalar_by_1scalar_bybnd` for LW ABSORPTION,
`inc_2stream_by_2stream_bybnd` for SW TWO_STREAM). No Pade, no McICA — pyRTE's
cloud path and the distributed SW_BND/LW_BND files are LUT and by-band.

The three pieces (LUT lookup, 1scl/2str combine, band->gpt increment) are proven
bit-for-bit against a NumPy transcription of the Fortran (two independent
implementations agree exactly; see the commit message / Phase-0 prototype). What
has **not** run off-Discover: the gt4py frontend compile and the pyRTE numeric
comparison. That is what this does.

The numpy backend is used, so **no GPU needed**; a login/interactive node is
fine. Substitute your username if not `rlgill`.

---

## Step 0 — re-entry block (every fresh shell)

```bash
module load python/GEOSpyD/24.11.3-0/3.12 comp/gcc/13.2.0
export CC=gcc CXX=g++ FC=gfortran
export XDG_CACHE_HOME=/discover/nobackup/$USER/fork-a/.cache
source /discover/nobackup/$USER/fork-a/venv/bin/activate
cd /discover/nobackup/$USER/fork-a/PySHiELD
```
`XDG_CACHE_HOME` is required (else the coeff download hits
`Errno 122 Disk quota exceeded`).

## Step 1 — update the fork clone to the committed port

```bash
git -C /discover/nobackup/$USER/fork-a/PySHiELD fetch origin
git -C /discover/nobackup/$USER/fork-a/PySHiELD checkout develop
git -C /discover/nobackup/$USER/fork-a/PySHiELD pull --ff-only origin develop
git -C /discover/nobackup/$USER/fork-a/PySHiELD log --oneline -1
```
(Discover clone is pull-only; never push from it.)

## Step 2 — CONFIRM the three pyRTE confirmation points (do this first — don't guess)

```bash
PYRTE=$(python -c "import pyrte_rrtmgp, os; print(os.path.dirname(pyrte_rrtmgp.__file__))")
# (a) the cloud input variable names CloudOptics.compute reads
grep -nE "lwp|iwp|\brel\b|\brei\b|get_var|liquid|ice|effective" "$PYRTE/rrtmgp.py" | sed -n '1,40p'
# (b) the attribute holding the cloud-coeff dataset + the LUT var/dim names
python - <<'PY'
from pyrte_rrtmgp.rrtmgp import CloudOptics
from pyrte_rrtmgp.rrtmgp_data_files import CloudOpticsFiles
co = CloudOptics(cloud_optics_file=CloudOpticsFiles.LW_BND)
for a in ("_dataset", "_cloud_optics", "dataset"):
    ds = getattr(co, a, None)
    if ds is not None and hasattr(ds, "sizes"):
        print("DATASET ATTR =", a); print("dims:", dict(ds.sizes))
        print("vars:", list(ds.data_vars)); break
PY
# (c) the gas-optics band->gpt limits var
python -c "from pyrte_rrtmgp.rrtmgp import GasOptics, GasOpticsFiles; \
print([v for v in GasOptics(gas_optics_file=GasOpticsFiles.SW_G224)._dataset.data_vars if 'bnd' in v or 'gpt' in v])"
```
Confirm:
- (a) the compute reads cloud water/ice path + liquid/ice effective radius under
  the names the test maps (`lwp`, `iwp`, `rel`, `rei`). If different, adjust the
  `atm[...]` names and the mapping in `_add_clouds`/`compute`.
- (b) the coeff dataset attribute is one of `_dataset`/`_cloud_optics`/`dataset`
  (the port tries all three) and holds `lut_extliq/ssaliq/asyliq`,
  `lut_extice/ssaice/asyice`, `radliq_lwr/upr`, `radice_lwr/upr`, dims
  `nsize_liq/nsize_ice/nband/nrghice`. If names dropped the `lut_` prefix, the
  port already tries the bare names (`extliq`, ...). If dims differ, add them to
  `_dim`/`_pad_*`.
- (c) the gas-optics `bnd_limits_gpt` (2, nbnd) 1-based is the band->gpt map; the
  test builds `gpt2band` from it. Confirm the default ice-roughness index pyRTE
  uses (the port defaults `ice_roughness=0`, matching Fortran `icergh=1`).

## Step 3 — run the validation test

```bash
pytest tests/rte_solver/test_cloud_optics_gt4py.py -q
```
First run compiles `cloud_lut_band`, `inc_1scalar_by_1scalar`,
`inc_2stream_by_2stream` (needs the `gcc` module) and downloads SW_BND / LW_BND /
SW_G224 / LW_G256.

- **PASS** → the cloud-optics LUT lookup + combine + band->gpt increment are
  validated against pyRTE (per-band tau/ssa/g and the combined gas+cloud optics
  after `add_to`, rtol 1e-10). Record it.
- **FAIL** → Step 4.

---

## Step 4 — diagnostics (only if Step 3 fails)

Order of suspects:
1. **Confirmation points (Step 2)** — most likely: cloud input var names, the
   coeff dataset attribute, LUT var/dim names, or the ice-roughness index.
2. **Ice-roughness index** — if only the ice contribution is off, `ice_roughness`
   (0-based) does not match pyRTE's default; pass the right index to
   `CloudOpticsGT4Py(..., ice_roughness=N)`.
3. **LUT table axis order** — if tau is wrong but finite, the `_pad_liq`/`_pad_ice`
   axis arrangement; the port auto-transposes by shape but confirm (nsize, nband).
4. **index/fint** — if only near-boundary radii are off, the `_idx_fint` clamp
   (`min(floor(val), nsteps-2)`); the Fortran clamps the UPPER bound only.
5. **increment** — if per-band cloud optics match but the combined (post-add_to)
   optics do not, the `gpt2band` map or the 1scl-vs-2str increment selection.
   LW increments tau only; SW increments tau/ssa/g.

---

## What this does NOT cover (later stages)
The all-sky RTE solve itself (the GT4Py clear-sky solvers do not yet consume
cloud optics; the all-sky solve stays on pyRTE for now) and the heating-rate
reduction. This runbook is cloud optics (by-band properties + increment) only.
```

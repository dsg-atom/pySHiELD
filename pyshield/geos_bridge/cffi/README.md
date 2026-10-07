# GEOS -> PySHiELD radiation CFFI bridge (Phase 1)

CFFI embedding bridge that wraps the Phase-0 pure-Python entry
(`pyshield/geos_bridge/geos_rrtmgp.py`) so GEOS Fortran can drive the GT4Py/NDSL
radiation column tile. Mirrors the merged gtFV3 bridge
(`gpu-analysis/FVdycoreCubed_GridComp/geos-gtfv3/`).

## Layers (top to bottom)
- `geos_rrtmgp_interface.f90` — Fortran `bind(c)` module: `geos_rrtmgp_interface_f_init`,
  `geos_rrtmgp_interface_f` (run), `geos_rrtmgp_interface_f_finalize`. Scalars are
  `value, intent(in)`; field arrays are `real(c_double) dimension(*)` (R8, not c_float).
- `geos_rrtmgp_interface.c` — C shim: `MPI_Comm_f2c`, then calls the CFFI-exported
  `geos_rrtmgp_interface_py*` symbols.
- `geos_rrtmgp_interface.py` — CFFI generator: `ffi.embedding_api` / `set_source` /
  `embedding_init_code` / `compile(target="libgeos_rrtmgp_interface_py.so")`. Its
  `@ffi.def_extern` trampolines reconstruct the MPI comm and call
  `pyshield.geos_bridge.geos_rrtmgp.geos_rrtmgp_{init,run,finalize}`.
- `radiation_f_py_conversion.py` — R8 marshalling. Flat `c_double` buffers <-> whole-tile
  float64 numpy. **Copy-back stride is `ffi.sizeof("double")` == 8 bytes**, NOT the
  hardcoded 4-byte (float32) stride in gtFV3's `f_py_conversion.py:325`.
- `geos_rrtmgp_env.py` — builds `RTE_RRTMGPConfig` + lons/lats/sigma from env vars, so
  the Fortran ABI stays scalars + comm + field arrays.
- `driver/geos_rrtmgp_driver.f90` — standalone Fortran driver (synthetic inputs,
  init -> run -> finalize, finite/physical flux checks). Wraps init in
  `ieee_set_halting_mode(ieee_all, .false.)` (FPE-trap workaround for numpy/pyRTE import).
- `driver/build_and_run.sh` — Discover build+run script.

## Build + run on Discover (login node)
```
bash driver/build_and_run.sh
```
Needs the Fork-A venv (cffi + ndsl + pyRTE), gfortran/gcc + MPI, and `XDG_CACHE_HOME`
set (first run downloads the RRTMGP coeff files).

## Not done here
Not wired into GEOS CMake / `components.yaml` (Phase 2, gated on user OK for the
superproject). The GEOS-native `(i,j,k)` Fortran-order -> `(ncol, nlev)` layout adapter
is also Phase 2; Phase 1 assumes the C-contiguous whole-tile contract documented in
`radiation_f_py_conversion.py`.

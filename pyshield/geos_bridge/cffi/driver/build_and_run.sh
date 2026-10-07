#!/usr/bin/env bash
#
# Build + run the standalone radiation CFFI-bridge driver on Discover (Phase 1).
#
# This is NOT wired into GEOS CMake (that is Phase 2, gated on user OK for the
# superproject). It proves the full Fortran -> C -> CFFI -> Python -> CFFI -> C ->
# Fortran round trip of libgeos_rrtmgp_interface_py.so in isolation.
#
# Run on a Discover LOGIN node (the first run downloads the RRTMGP coeff files;
# compute nodes have no network). Needs the Fork-A venv (cffi + ndsl + pyRTE) and
# the gfortran/gcc + MPI toolchain.
#
# Usage:
#   bash build_and_run.sh
#
set -euo pipefail

# --- Fork-A re-entry block (discover-ndsl-env skill). Adjust USER / paths. ---
module load python/GEOSpyD/24.11.3-0/3.12 comp/gcc/13.2.0
export CC=gcc CXX=g++ FC=gfortran
: "${FORK_A:=/discover/nobackup/${USER}/fork-a}"
export XDG_CACHE_HOME="${FORK_A}/.cache"
source "${FORK_A}/venv/bin/activate"

# --- Bridge run configuration (geos_rrtmgp_env.py reads these) ---
export GEOS_RRTMGP_BACKEND="${GEOS_RRTMGP_BACKEND:-numpy}"
# input_dir only needs to be a readable dir under the chosen flags (no text files
# are read: isolar=10, iemsflg=0, ico2flg=0, ictmflg=-1, ioznflg=1, ialbflg=-1).
export GEOS_RRTMGP_INPUT_DIR="${GEOS_RRTMGP_INPUT_DIR:-$(mktemp -d)}"

# --- Locate sources. CFFI_DIR is this script's parent (…/geos_bridge/cffi). ---
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFFI_DIR="$(cd "${HERE}/.." && pwd)"
cd "${CFFI_DIR}"

# MPI compiler wrappers (gfortran/gcc under the hood via FC/CC above).
MPIFC="${MPIFC:-mpif90}"
MPICC="${MPICC:-mpicc}"

# Python lib dir (so the driver can resolve the embedded libpython at run time).
PYLIBDIR="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"

echo "== [1/5] Generate + compile libgeos_rrtmgp_interface_py.so =="
# Writes geos_rrtmgp_interface_py.{c,h,o} and libgeos_rrtmgp_interface_py.so here.
python geos_rrtmgp_interface.py

echo "== [2/5] Compile C shim =="
${MPICC} -c -I. geos_rrtmgp_interface.c -o geos_rrtmgp_interface.o

echo "== [3/5] Compile Fortran interface module =="
${MPIFC} -c geos_rrtmgp_interface.f90 -o geos_rrtmgp_interface_f.o

echo "== [4/5] Compile + link the driver =="
${MPIFC} -c driver/geos_rrtmgp_driver.f90 -o geos_rrtmgp_driver.o
${MPIFC} -o geos_rrtmgp_driver \
    geos_rrtmgp_driver.o \
    geos_rrtmgp_interface_f.o \
    geos_rrtmgp_interface.o \
    -L. -lgeos_rrtmgp_interface_py \
    -Wl,-rpath,. -Wl,-rpath,"${PYLIBDIR}"

echo "== [5/5] Run =="
export LD_LIBRARY_PATH=".:${PYLIBDIR}:${LD_LIBRARY_PATH:-}"
mpirun -np 1 ./geos_rrtmgp_driver

"""CFFI embedding bridge (Phase 1) for GEOS -> PySHiELD radiation.

Wraps the Phase-0 pure-Python entry (``pyshield.geos_bridge.geos_rrtmgp``) in a
Fortran bind(c) + C shim + CFFI embedding ``.so``, mirroring the merged gtFV3
bridge. Build the ``.so`` with ``python geos_rrtmgp_interface.py`` (see
``driver/build_and_run.sh``). Not wired into GEOS CMake -- that is Phase 2.
"""

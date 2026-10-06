"""GEOS <-> PySHiELD radiation bridge (Phase 0, pure Python).

``marshal`` is the pyRTE-free whole-tile <-> NDSL-Quantity marshalling layer
(importable and testable with only ndsl installed). ``geos_rrtmgp`` adds the
driver-construction + singleton entry points and pulls in pyRTE / rte_rrtmgp,
so it is imported lazily by callers rather than re-exported here -- that keeps
``import pyshield.geos_bridge.marshal`` free of the pyRTE dependency.
"""

from . import marshal  # noqa: F401

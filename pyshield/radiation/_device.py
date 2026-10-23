"""Host/device array helpers for the GT4Py radiation port.

On a GPU backend the ndsl ``Quantity`` views are cupy arrays; on CPU they are
numpy. The host-side driver loops in ``gas_optics_gt4py`` / ``cloud_optics_gt4py``
and the solvers assign per-g-point slices into those views and accumulate
broadband results. For that to work on GPU the slices and accumulators must live
on the same device as the views.

These helpers keep that code backend-native: ``array_module`` returns the array
module a ``Quantity`` was allocated on (cupy or numpy), so callers build their
accumulators and convert their inputs with it, and everything stays on the
device. Results are pulled back to host only at the boundaries -- test
assertions, diagnostics, and the GEOS bridge marshalling -- via ``to_host``.

On a CPU backend the module is numpy and every call is a no-op, so the CPU path
is unchanged.
"""

import numpy as np

try:
    import cupy as _cp  # device arrays under a GPU backend; absent on CPU-only installs
except ImportError:  # pragma: no cover - CPU-only environments
    _cp = None


def array_module(quantity):
    """Return numpy or cupy, matching the backend a ``Quantity`` lives on."""
    if _cp is not None and isinstance(quantity.view[:], _cp.ndarray):
        return _cp
    return np


def to_host(a):
    """Return a host numpy array for a numpy or cupy array (no-op for numpy)."""
    if _cp is not None and isinstance(a, _cp.ndarray):
        return _cp.asnumpy(a)
    return np.asarray(a)

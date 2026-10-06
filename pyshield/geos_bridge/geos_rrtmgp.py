"""Pure-Python GEOS <-> PySHiELD radiation bridge entry (Phase 0).

Phase 0 of the GEOS CFFI radiation integration. This is PURE PYTHON ONLY: no
CFFI / f90 / C (that is Phase 1), no GEOS, no superproject. It gives a future
CFFI bridge a singleton it can drive by passing whole-tile numpy arrays, exactly
the way ``geos_gtfv3.py`` lets GEOS drive the dynamical core.

Structure mirrors the gtFV3 driver
(``gpu-analysis/FVdycoreCubed_GridComp/geos-gtfv3/geos_gtfv3.py``):

* a module-global singleton instance,
* ``geos_rrtmgp_init`` -- construct ONE RTE_RRTMGPDriver (and its state /
  surface-state buffers) once, with a double-init guard,
* ``geos_rrtmgp_run`` -- marshal whole-tile numpy inputs into the state, run
  ``step_radiation``, extract whole-tile numpy outputs,
* ``geos_rrtmgp_finalize``.

The marshalling itself lives in :mod:`pyshield.geos_bridge.marshal` (pyRTE-free,
unit-testable). This module adds the driver construction + singleton plumbing.

PHYSICS SCOPE (Build-2 literal): ``step_radiation`` is driven AS-IS. It computes
clouds (progcld4/5), albedo (set_albedo) and emissivity (set_sfcemis) GFS-style
from the base state + SurfaceState. Phase 0 therefore populates the BASE state +
SurfaceState and lets PySHiELD do its GFS physics. The Build-1 GEOS-field
injection (bypassing progcld / set_albedo with GEOS's own cloud/albedo fields)
is deliberately NOT done here; the injection point is flagged in ``run``.
"""

import datetime
import os

import numpy as np

from ndsl import QuantityFactory, StencilFactory
from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM
from ndsl.dsl.typing import Float

from pyshield.constants import P_REF
from pyshield.radiation._config import RTE_RRTMGPConfig
from pyshield.radiation.rte_rrtmgp import RTE_RRTMGPDriver
from pyshield.radiation.state import RTE_RRTMGPState
from pyshield.stencils.surface import SurfaceState

from . import marshal


def _calc_sigma(ak: np.ndarray, bk: np.ndarray, k_toa: int = 0) -> np.ndarray:
    """Hybrid-coordinate sigma for cld_init (inlined copy of the one-line
    pyshield.stencils.physics.calc_sigma, to avoid importing that heavy
    stencil module)."""
    return (ak + bk * P_REF - ak[k_toa]) / (P_REF - ak[k_toa])


class GEOSRRTMGP:
    """Holds one radiation driver + its reusable state buffers for the bridge.

    The driver, the single-tile factories, and the RTE_RRTMGPState /
    SurfaceState buffers are allocated once here and reused on every ``run``
    call; marshalling overwrites the inputs each step and ``step_radiation``
    overwrites every output, so no per-step allocation is needed.
    """

    def __init__(
        self,
        config: RTE_RRTMGPConfig,
        nx: int,
        ny: int,
        nz: int,
        lons: np.ndarray,
        lats: np.ndarray,
        sigma: np.ndarray,
        nhalo: int = 3,
        backend: str = "numpy",
        top_at_1: bool = True,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nhalo = nhalo
        self.backend = backend
        # GEOS is top-down; the compiled solvers are bottom-up. top_at_1 drives
        # the vertical flip in the marshalling layer. GEOS computes it at
        # runtime, so it is a bridge input (default True = top-down GEOS).
        self.top_at_1 = top_at_1

        self.stencil_factory: StencilFactory
        self.quantity_factory: QuantityFactory
        self.stencil_factory, self.quantity_factory = get_factories_single_tile(
            nx=nx, ny=ny, nz=nz, nhalo=nhalo, backend=backend
        )

        # Grid lon/lat as (I, J) Quantities (gas_init / set_sfcemis read
        # gridlon.view[:], gridlat.view[:], which is the (nx, ny) compute tile).
        self._gridlon = self.quantity_factory.zeros([I_DIM, J_DIM], "radians", dtype=Float)
        self._gridlat = self.quantity_factory.zeros([I_DIM, J_DIM], "radians", dtype=Float)
        self._gridlon.view[:] = np.asarray(lons, dtype=Float).reshape(nx, ny)
        self._gridlat.view[:] = np.asarray(lats, dtype=Float).reshape(nx, ny)

        self._driver = RTE_RRTMGPDriver(
            config=config,
            gridlon=self._gridlon,
            gridlat=self._gridlat,
            sigma=np.asarray(sigma),
            quantity_factory=self.quantity_factory,
            stencil_factory=self.stencil_factory,
        )

        # Reusable state + surface-state buffers (zeroed once).
        self.state = RTE_RRTMGPState.init_zeros(self.quantity_factory)
        self.sfc_state = SurfaceState.init_zeros(self.quantity_factory)

    def run(
        self,
        inputs: dict,
        sfc_inputs: dict,
        date: datetime.datetime,
        top_at_1: bool = None,
    ) -> dict:
        """Marshal in -> step_radiation -> marshal out. Pure pass-through.

        Args:
            inputs: ``{field: whole_tile_array}`` for RTE_RRTMGPState base
                fields (see marshal.RAD_INPUT_FIELDS;
                marshal.RAD_REQUIRED_INPUT_FIELDS must be present/non-zero).
            sfc_inputs: ``{field: whole_tile_array}`` for SurfaceState fields
                (marshal.SFC_INPUT_FIELDS).
            date: datetime for the radiation step.
            top_at_1: override the construction-time orientation for this call;
                defaults to the instance setting.

        Returns:
            ``{field: whole_tile_array}`` for marshal.RAD_OUTPUT_FIELDS.
        """
        flip = self.top_at_1 if top_at_1 is None else top_at_1

        marshal.marshal_inputs(
            self.state,
            self.sfc_state,
            inputs,
            sfc_inputs,
            self.nx,
            self.ny,
            self.nz,
            flip,
        )

        # --- BUILD-1 INJECTION POINT -------------------------------------
        # Build-2 (this code) lets step_radiation compute clouds (progcld4/5),
        # albedo (set_albedo) and emissivity (set_sfcemis) GFS-style from the
        # base state above. Build-1 would instead inject GEOS's OWN cloud
        # optical properties / cloud fraction / albedo / emissivity here --
        # overwriting state.clwp/cip/clwr/cir + state.albedo/sfc_emis after
        # marshalling but before step_radiation -- and bypass the GFS physics.
        # That bypass is intentionally NOT done in Phase 0.
        # -----------------------------------------------------------------

        self._driver.step_radiation(self.state, self.sfc_state, date)

        return marshal.extract_outputs(
            self.state, self.nx, self.ny, self.nz, flip
        )

    def finalize(self):
        # No external resources to release in Phase 0; present for symmetry with
        # the gtFV3 bridge and as the Phase-1 hook (timers, device buffers).
        pass


# ---------------------------------------------------------------------------
# Module-global singleton + the three CFFI-facing entry functions, mirroring
# geos_gtfv3.py (GEOS_DYCORE / geos_gtfv3_init / geos_gtfv3 / _finalize).
# ---------------------------------------------------------------------------
GEOS_RRTMGP = None


def geos_rrtmgp_init(
    config: RTE_RRTMGPConfig,
    nx: int,
    ny: int,
    nz: int,
    lons: np.ndarray,
    lats: np.ndarray,
    sigma: np.ndarray = None,
    ak: np.ndarray = None,
    bk: np.ndarray = None,
    nhalo: int = 3,
    backend: str = None,
    top_at_1: bool = True,
) -> None:
    """Construct the one radiation driver and hold it module-global.

    Args:
        config: RTE_RRTMGPConfig (date, flags, input_dir, solar/aerosol files).
        nx, ny, nz: single-tile compute dims (ncol = nx*ny).
        lons, lats: whole-tile ``(ncol,)`` grid longitude/latitude (radians).
        sigma: ``(nz+1,)`` sigma coordinate for cld_init. If None it is built
            from ``ak``/``bk`` via ``calc_sigma`` (GEOS can pass either).
        ak, bk: hybrid-coordinate coefficients, used only if ``sigma`` is None.
        nhalo: NDSL halo width (default 3).
        backend: gt4py backend; defaults to env GEOS_RRTMGP_BACKEND or "numpy".
        top_at_1: True if the whole-tile arrays are top-down (GEOS default).
    """
    global GEOS_RRTMGP
    if GEOS_RRTMGP is not None:
        raise RuntimeError("[GEOS RRTMGP] Double init")

    if backend is None:
        backend = os.environ.get("GEOS_RRTMGP_BACKEND", "numpy")

    if sigma is None:
        if ak is None or bk is None:
            raise ValueError("geos_rrtmgp_init needs either sigma or (ak, bk)")
        sigma = _calc_sigma(np.asarray(ak), np.asarray(bk), 0)

    GEOS_RRTMGP = GEOSRRTMGP(
        config=config,
        nx=nx,
        ny=ny,
        nz=nz,
        lons=lons,
        lats=lats,
        sigma=sigma,
        nhalo=nhalo,
        backend=backend,
        top_at_1=top_at_1,
    )


def geos_rrtmgp_run(
    inputs: dict,
    sfc_inputs: dict,
    date: datetime.datetime,
    top_at_1: bool = None,
) -> dict:
    """Run one radiation step through the singleton. Requires init."""
    if GEOS_RRTMGP is None:
        raise RuntimeError("[GEOS RRTMGP] Bad init, did you call geos_rrtmgp_init?")
    return GEOS_RRTMGP.run(inputs, sfc_inputs, date, top_at_1=top_at_1)


def geos_rrtmgp_finalize():
    global GEOS_RRTMGP
    if GEOS_RRTMGP is not None:
        GEOS_RRTMGP.finalize()
        GEOS_RRTMGP = None

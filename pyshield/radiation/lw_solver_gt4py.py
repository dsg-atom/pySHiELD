"""GT4Py longwave no-scattering RTE solver, host driver.

`LWNoScatSolverGT4Py` wraps the `lw_solver.py` stencils with the host-side
bookkeeping that the gas-optics port (`gas_optics_gt4py.py`) established: build
the stencils once in `__init__`, fold the per-column inputs into the (nx, ny)
stencil tile, drive the 256 g-points with a Python loop around the compile-once
stencil, and fold the broadband flux back out.

Scope is the simple path of `lw_solver_noscat` -- do_broadband, nmus = 1, no
Jacobians, no rescaling, clear-sky -- so this is the longwave half of what pyRTE
currently does on the CPU in `lw_optics.rte.solve`. It is validated in isolation
(inputs = the staged gas-optics reference `ref_gas_optics_lw.nc`, so solver error
is not compounded with gas-optics error) against pyRTE's own ABSORPTION solve on
the same tau/sources.

The quadrature constants for nmus = 1 are the Gauss-Jacobi-5 values from
`mo_rte_lw.F90` (Hogan 2023): secant D = 1 / 0.6096748751, weight = 1.0. They and
`top_at_1` are compile-time externals; confirm them against a pyRTE LW solve on
Discover before trusting a validation (see the module/test notes).
"""

import numpy as np

from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float

from .lw_solver import (
    lw_noscat_gpoint_top_at_1,
    lw_noscat_gpoint_top_at_n,
    scale_broadband,
)

# nmus = 1 Gauss-Jacobi-5 quadrature (mo_rte_lw.F90): secant and weight.
D_DEFAULT = 1.0 / 0.6096748751
WEIGHT_DEFAULT = 1.0

PI = float(np.arccos(-1.0))
# Series-expansion cutoff: sqrt(sqrt(epsilon(float64))), matching the Fortran
# tau_thresh = sqrt(sqrt(epsilon(tau))).
TAU_THRESH = float(np.sqrt(np.sqrt(np.finfo(np.float64).eps)))


class LWNoScatSolverGT4Py:
    """Longwave clear-sky no-scattering RTE solver as GT4Py stencils."""

    def __init__(
        self,
        nx,
        ny,
        nz,
        ngpt,
        backend,
        top_at_1,
        D=D_DEFAULT,
        weight=WEIGHT_DEFAULT,
        nhalo=3,
    ):
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.ngpt = int(ngpt)
        self.nlev = self.nz + 1
        self.top_at_1 = bool(top_at_1)
        self.D = float(D)
        self.weight = float(weight)

        # One K axis of length nlev carries both layer and level fields.
        sf, qf = get_factories_single_tile(
            nx=self.nx, ny=self.ny, nz=self.nlev, nhalo=nhalo, backend=backend
        )
        gi = sf.grid_indexing
        self._sf, self._qf, self._gi = sf, qf, gi

        externals = dict(
            D=self.D, weight=self.weight, pi=PI, tau_thresh=TAU_THRESH
        )
        gpoint_func = (
            lw_noscat_gpoint_top_at_1 if self.top_at_1 else lw_noscat_gpoint_top_at_n
        )
        self._gpoint = sf.from_origin_domain(
            func=gpoint_func,
            externals=externals,
            origin=gi.origin_compute(),
            domain=gi.domain_compute(),
        )
        self._scale = sf.from_origin_domain(
            func=scale_broadband,
            externals=dict(pi=PI, weight=self.weight),
            origin=gi.origin_compute(),
            domain=gi.domain_compute(),
        )

        # Persistent quantities (reused across g-points).
        self._tau = self._ff()
        self._lay = self._ff()
        self._lev = self._ff()
        self._emis = self._ff()
        self._src = self._ff()
        self._inc = self._ff()
        self._trans = self._ff()
        self._sdn = self._ff()
        self._sup = self._ff()
        self._fdn = self._ff()
        self._fup = self._ff()
        self._bb_up = self._ff()
        self._bb_dn = self._ff()

    def _ff(self):
        return self._qf.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def solve(self, tau, lay_source, lev_source, sfc_src, sfc_emis, inc_flux=None):
        """Broadband longwave up/down flux from tau and the Planck sources.

        All inputs are float64 arrays folded to the (nx, ny) tile:
            tau, lay_source : (nx, ny, nlay, ngpt)
            lev_source      : (nx, ny, nlev, ngpt)      nlev = nlay + 1
            sfc_src, sfc_emis, inc_flux : (nx, ny, ngpt)
        `inc_flux` defaults to 0 (longwave clear-sky). Returns broadband_up,
        broadband_dn, each (nx, ny, nlev) in W/m2.
        """
        nx, ny, nz, nlev, ngpt = self.nx, self.ny, self.nz, self.nlev, self.ngpt
        tau = np.asarray(tau, dtype=np.float64)
        lay_source = np.asarray(lay_source, dtype=np.float64)
        lev_source = np.asarray(lev_source, dtype=np.float64)
        sfc_src = np.asarray(sfc_src, dtype=np.float64)
        sfc_emis = np.asarray(sfc_emis, dtype=np.float64)
        if inc_flux is None:
            inc_flux = np.zeros((nx, ny, ngpt), dtype=np.float64)
        else:
            inc_flux = np.asarray(inc_flux, dtype=np.float64)

        assert tau.shape == (nx, ny, nz, ngpt), tau.shape
        assert lay_source.shape == (nx, ny, nz, ngpt), lay_source.shape
        assert lev_source.shape == (nx, ny, nlev, ngpt), lev_source.shape

        self._bb_up.view[:] = 0.0
        self._bb_dn.view[:] = 0.0

        for g in range(ngpt):
            # layer fields fill K = 0..nlay-1; the last interface slot stays 0.
            self._tau.view[:, :, :nz] = tau[:, :, :, g]
            self._lay.view[:, :, :nz] = lay_source[:, :, :, g]
            self._lev.view[:, :, :nlev] = lev_source[:, :, :, g]
            # per-column boundary constants, broadcast across K (read only at the
            # boundary interface by the stencil).
            self._emis.view[:] = sfc_emis[:, :, g][:, :, None]
            self._src.view[:] = sfc_src[:, :, g][:, :, None]
            self._inc.view[:] = inc_flux[:, :, g][:, :, None]
            self._gpoint(
                tau=self._tau,
                lay_source=self._lay,
                lev_source=self._lev,
                sfc_emis=self._emis,
                sfc_src=self._src,
                inc_flux=self._inc,
                trans=self._trans,
                source_dn=self._sdn,
                source_up=self._sup,
                flux_dn=self._fdn,
                flux_up=self._fup,
                broadband_up=self._bb_up,
                broadband_dn=self._bb_dn,
            )

        self._scale(broadband_up=self._bb_up, broadband_dn=self._bb_dn)

        bb_up = np.array(self._bb_up.view[:, :, :nlev])
        bb_dn = np.array(self._bb_dn.view[:, :, :nlev])
        return bb_up, bb_dn

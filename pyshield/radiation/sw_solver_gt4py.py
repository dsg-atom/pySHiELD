"""GT4Py shortwave two-stream RTE solver, host driver.

`SWTwoStreamSolverGT4Py` wraps the `sw_solver.py` stencils with the host-side
bookkeeping established by the gas-optics and longwave-solver ports: build the
stencil once in `__init__`, fold the per-column inputs into the (nx, ny) stencil
tile, drive the 224 shortwave g-points with a Python loop around the
compile-once stencil, and fold the broadband up/down/direct flux back out.

Scope is the simple path of `sw_solver_2stream` -- do_broadband, clear-sky,
has_dif_bc = FALSE (inc_flux_dif = 0) -- so this is the shortwave half of what
pyRTE currently does on the CPU in `sw_optics.rte.solve` (problem_type
`TWO_STREAM`). It is validated in isolation against pyRTE's own two-stream solve
on identical tau/ssa/g/mu0/albedo/inc_flux_dir.

There is NO pi*weight scaling and NO Gauss-angle loop (those are longwave); the
broadband accumulation per g-point is up += flux_up, dn += flux_dn + flux_dir,
dir += flux_dir, done inside the stencil. `top_at_1` selects which of the two
orientation stencils is compiled, exactly as in `lw_solver_gt4py.py`.
"""

import numpy as np

from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float

from .sw_solver import (
    sw_2stream_gpoint_top_at_1,
    sw_2stream_gpoint_top_at_n,
)

# Fortran ancillary constants (mo_rte_solver_kernels.F90, sw_dif_and_source),
# all in float64: min_k = 1e4*epsilon, min_mu0 = sqrt(epsilon).
EPS = float(np.finfo(np.float64).eps)
MIN_K = 1.0e4 * EPS
MIN_MU0 = float(np.sqrt(EPS))


class SWTwoStreamSolverGT4Py:
    """Shortwave two-stream clear-sky RTE solver as GT4Py stencils."""

    def __init__(self, nx, ny, nz, ngpt, backend, top_at_1, nhalo=3):
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.ngpt = int(ngpt)
        self.nlev = self.nz + 1
        self.top_at_1 = bool(top_at_1)

        # One K axis of length nlev carries both layer and level fields.
        sf, qf = get_factories_single_tile(
            nx=self.nx, ny=self.ny, nz=self.nlev, nhalo=nhalo, backend=backend
        )
        gi = sf.grid_indexing
        self._sf, self._qf, self._gi = sf, qf, gi

        externals = dict(eps=EPS, min_k=MIN_K, min_mu0=MIN_MU0)
        gpoint_func = (
            sw_2stream_gpoint_top_at_1
            if self.top_at_1
            else sw_2stream_gpoint_top_at_n
        )
        self._gpoint = sf.from_origin_domain(
            func=gpoint_func,
            externals=externals,
            origin=gi.origin_compute(),
            domain=gi.domain_compute(),
        )

        # Persistent quantities (reused across g-points).
        self._tau = self._ff()
        self._ssa = self._ff()
        self._g = self._ff()
        self._mu0 = self._ff()
        self._alb_dir = self._ff()
        self._alb_dif = self._ff()
        self._inc_dir = self._ff()
        self._rdif = self._ff()
        self._tdif = self._ff()
        self._rdir = self._ff()
        self._tdir = self._ff()
        self._tnoscat = self._ff()
        self._sup = self._ff()
        self._sdn = self._ff()
        self._fdir = self._ff()
        self._albedo = self._ff()
        self._src = self._ff()
        self._denom = self._ff()
        self._fup = self._ff()
        self._fdn = self._ff()
        self._bb_up = self._ff()
        self._bb_dn = self._ff()
        self._bb_dir = self._ff()

    def _ff(self):
        return self._qf.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def _fill_mu0(self, mu0):
        """mu0 -> (nx, ny, nlev). Per-column mu0 broadcast across all K.

        Accepts (nx, ny) [per column] or (nx, ny, nz) [per layer]. For per-layer
        input the extra surface-level slot K = nz is filled with the last layer's
        value; for the per-column case (the validation case) every K is equal.
        """
        nx, ny, nz, nlev = self.nx, self.ny, self.nz, self.nlev
        self._mu0.view[:] = 0.0
        if mu0.ndim == 2:
            self._mu0.view[:] = mu0[:, :, None]
        else:
            assert mu0.shape == (nx, ny, nz), mu0.shape
            self._mu0.view[:, :, :nz] = mu0
            self._mu0.view[:, :, nz] = mu0[:, :, -1]

    def solve(self, tau, ssa, g, mu0, sfc_alb_dir, sfc_alb_dif, inc_flux_dir):
        """Broadband shortwave up/down/direct flux from the two-stream optics.

        Inputs (float64, folded to the (nx, ny) tile):
            tau, ssa, g  : (nx, ny, nlay, ngpt)
            mu0          : (nx, ny) or (nx, ny, nlay)   cosine solar zenith
            sfc_alb_dir  : (nx, ny, ngpt)               direct surface albedo
            sfc_alb_dif  : (nx, ny, ngpt)               diffuse surface albedo
            inc_flux_dir : (nx, ny, ngpt)               TSI-scaled toa_source
        Returns broadband_up, broadband_dn, broadband_dir, each (nx, ny, nlev)
        in W/m2. broadband_dn is the TOTAL (diffuse + direct) downward flux.
        """
        nx, ny, nz, nlev, ngpt = self.nx, self.ny, self.nz, self.nlev, self.ngpt
        tau = np.asarray(tau, dtype=np.float64)
        ssa = np.asarray(ssa, dtype=np.float64)
        g = np.asarray(g, dtype=np.float64)
        mu0 = np.asarray(mu0, dtype=np.float64)
        sfc_alb_dir = np.asarray(sfc_alb_dir, dtype=np.float64)
        sfc_alb_dif = np.asarray(sfc_alb_dif, dtype=np.float64)
        inc_flux_dir = np.asarray(inc_flux_dir, dtype=np.float64)

        assert tau.shape == (nx, ny, nz, ngpt), tau.shape
        assert ssa.shape == (nx, ny, nz, ngpt), ssa.shape
        assert g.shape == (nx, ny, nz, ngpt), g.shape
        assert sfc_alb_dir.shape == (nx, ny, ngpt), sfc_alb_dir.shape
        assert sfc_alb_dif.shape == (nx, ny, ngpt), sfc_alb_dif.shape
        assert inc_flux_dir.shape == (nx, ny, ngpt), inc_flux_dir.shape

        self._fill_mu0(mu0)

        self._bb_up.view[:] = 0.0
        self._bb_dn.view[:] = 0.0
        self._bb_dir.view[:] = 0.0

        for gpt in range(ngpt):
            # layer fields fill K = 0..nlay-1; the last interface slot stays 0.
            self._tau.view[:, :, :nz] = tau[:, :, :, gpt]
            self._ssa.view[:, :, :nz] = ssa[:, :, :, gpt]
            self._g.view[:, :, :nz] = g[:, :, :, gpt]
            # per-column boundary constants, broadcast across K.
            self._alb_dir.view[:] = sfc_alb_dir[:, :, gpt][:, :, None]
            self._alb_dif.view[:] = sfc_alb_dif[:, :, gpt][:, :, None]
            self._inc_dir.view[:] = inc_flux_dir[:, :, gpt][:, :, None]
            self._gpoint(
                tau=self._tau,
                ssa=self._ssa,
                g=self._g,
                mu0=self._mu0,
                sfc_alb_dir=self._alb_dir,
                sfc_alb_dif=self._alb_dif,
                inc_flux_dir=self._inc_dir,
                rdif=self._rdif,
                tdif=self._tdif,
                rdir=self._rdir,
                tdir=self._tdir,
                tnoscat=self._tnoscat,
                src_up=self._sup,
                src_dn=self._sdn,
                flux_dir=self._fdir,
                albedo=self._albedo,
                src=self._src,
                denom=self._denom,
                flux_up=self._fup,
                flux_dn=self._fdn,
                broadband_up=self._bb_up,
                broadband_dn=self._bb_dn,
                broadband_dir=self._bb_dir,
            )

        bb_up = np.array(self._bb_up.view[:, :, :nlev])
        bb_dn = np.array(self._bb_dn.view[:, :, :nlev])
        bb_dir = np.array(self._bb_dir.view[:, :, :nlev])
        return bb_up, bb_dn, bb_dir

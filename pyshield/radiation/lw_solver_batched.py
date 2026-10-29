"""Prototype: longwave no-scattering solver with the g-point axis folded into
the compute domain, instead of the 256-iteration host loop in
``lw_solver_gt4py.LWNoScatSolverGT4Py``.

The solver is a per-column vertical sweep with no horizontal coupling (the
stencils read only K-neighbors), so laying out ``(column, g-point)`` as the
``(I, J)`` tile and calling the stencil ONCE computes every g-point in a single
launch. The broadband flux is then a reduction over the g-point axis instead of
the per-call ``broadband += flux`` the looped solver does. g-points are processed
in chunks of ``chunk`` so peak memory is bounded (``chunk`` must divide ngpt).

The stencils themselves are reused unchanged from ``lw_solver.py`` -- this is a
layout + host-driver change, which is the whole point of the prototype: prove the
folded layout is faster and fits in memory before touching the gas-optics gather.
"""

import numpy as np

from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float

from ._device import array_module, to_host
from .lw_solver import lw_noscat_gpoint_top_at_1, lw_noscat_gpoint_top_at_n
from .lw_solver_gt4py import D_DEFAULT, PI, TAU_THRESH, WEIGHT_DEFAULT


class LWNoScatSolverBatched:
    """Longwave clear-sky no-scattering solver, g-points folded into the tile."""

    def __init__(
        self, ncol, nz, ngpt, backend, top_at_1,
        D=D_DEFAULT, weight=WEIGHT_DEFAULT, chunk=None,
    ):
        self.ncol = int(ncol)
        self.nz = int(nz)
        self.ngpt = int(ngpt)
        self.nlev = self.nz + 1
        self.top_at_1 = bool(top_at_1)
        self.D = float(D)
        self.weight = float(weight)
        self.chunk = int(chunk) if chunk else self.ngpt
        if self.ngpt % self.chunk != 0:
            raise ValueError(f"chunk {self.chunk} must divide ngpt {self.ngpt}")

        # Tile: I = columns, J = g-points within a chunk, K = nlev. The stencil
        # reads only K-neighbors, so no horizontal halo is needed (nhalo = 0).
        sf, qf = get_factories_single_tile(
            nx=self.ncol, ny=self.chunk, nz=self.nlev, nhalo=0, backend=backend
        )
        gi = sf.grid_indexing
        self._qf = qf
        func = lw_noscat_gpoint_top_at_1 if self.top_at_1 else lw_noscat_gpoint_top_at_n
        self._gpoint = sf.from_origin_domain(
            func=func,
            externals=dict(D=self.D, weight=self.weight, pi=PI, tau_thresh=TAU_THRESH),
            origin=gi.origin_compute(),
            domain=gi.domain_compute(),
        )
        self._tau = self._ff(); self._lay = self._ff(); self._lev = self._ff()
        self._emis = self._ff(); self._src = self._ff(); self._inc = self._ff()
        self._trans = self._ff(); self._sdn = self._ff(); self._sup = self._ff()
        self._fdn = self._ff(); self._fup = self._ff()
        self._bbu = self._ff(); self._bbd = self._ff()  # stencil scratch, unused

    def _ff(self):
        return self._qf.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def solve(self, tau, lay_source, lev_source, sfc_src, sfc_emis, inc_flux=None):
        """Broadband up/down flux from folded (column, g-point) inputs.

        Shapes (host or device arrays):
            tau, lay_source : (ncol, nz, ngpt)
            lev_source      : (ncol, nlev, ngpt)
            sfc_src, sfc_emis, inc_flux : (ncol, ngpt)
        Returns broadband_up, broadband_dn, each (ncol, nlev), as host numpy.
        """
        ncol, nz, nlev, ngpt, gc = self.ncol, self.nz, self.nlev, self.ngpt, self.chunk
        xp = array_module(self._tau)
        tau = xp.asarray(tau, dtype=np.float64)
        lay_source = xp.asarray(lay_source, dtype=np.float64)
        lev_source = xp.asarray(lev_source, dtype=np.float64)
        sfc_src = xp.asarray(sfc_src, dtype=np.float64)
        sfc_emis = xp.asarray(sfc_emis, dtype=np.float64)
        inc = (xp.zeros((ncol, ngpt), np.float64) if inc_flux is None
               else xp.asarray(inc_flux, dtype=np.float64))

        assert tau.shape == (ncol, nz, ngpt), tau.shape
        assert lev_source.shape == (ncol, nlev, ngpt), lev_source.shape

        bb_up = xp.zeros((ncol, nlev), dtype=np.float64)
        bb_dn = xp.zeros((ncol, nlev), dtype=np.float64)

        for g0 in range(0, ngpt, gc):
            g1 = g0 + gc
            # (ncol, nz|nlev, gc) -> (ncol, gc, nz|nlev) into the I,J,K tile.
            self._tau.view[:, :, :nz] = tau[:, :, g0:g1].transpose(0, 2, 1)
            self._lay.view[:, :, :nz] = lay_source[:, :, g0:g1].transpose(0, 2, 1)
            self._lev.view[:, :, :nlev] = lev_source[:, :, g0:g1].transpose(0, 2, 1)
            # per-column boundary constants, broadcast across K.
            self._emis.view[:] = sfc_emis[:, g0:g1][:, :, None]
            self._src.view[:] = sfc_src[:, g0:g1][:, :, None]
            self._inc.view[:] = inc[:, g0:g1][:, :, None]
            self._gpoint(
                tau=self._tau, lay_source=self._lay, lev_source=self._lev,
                sfc_emis=self._emis, sfc_src=self._src, inc_flux=self._inc,
                trans=self._trans, source_dn=self._sdn, source_up=self._sup,
                flux_dn=self._fdn, flux_up=self._fup,
                broadband_up=self._bbu, broadband_dn=self._bbd,
            )
            # broadband = sum over g-points; reduce this chunk's J axis and add.
            bb_up += self._fup.view[:, :, :nlev].sum(axis=1)
            bb_dn += self._fdn.view[:, :, :nlev].sum(axis=1)

        bb_up *= PI * self.weight
        bb_dn *= PI * self.weight
        return to_host(bb_up), to_host(bb_dn)

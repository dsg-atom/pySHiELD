"""GT4Py cloud-optics core wired to the pyRTE driver interface.

`CloudOpticsGT4Py` is a drop-in for pyRTE's `CloudOptics` at the point the driver
(`rte_rrtmgp.py`) calls `.compute(...)` and then `cloud_props.rte.add_to(...)`.
It reproduces pyRTE's own cloud optics -- which is itself the LUT branch of the
Fortran `ty_cloud_optics_rrtmgp%cloud_optics` -- with:

    cloud_lut_band          per-band LUT lookup in effective radius (the device
                            kernel; liquid and ice gathered separately)
    inc_1scalar_by_1scalar  longwave band->g-point increment (ABSORPTION)
    inc_2stream_by_2stream  shortwave band->g-point increment (TWO_STREAM)

validated one by one against the Fortran (see `tests/rte_solver`). The container
is overwrite-first, exactly like `GasOpticsGT4Py`: `.compute` calls pyRTE
`CloudOptics.compute(...)` to get a fully-structured output dataset (right vars,
dims, coords, attrs, and the `.rte`/`.mapping` accessors), then overwrites the
same-named data vars in place with the GT4Py results. `.add_to` folds the by-band
cloud optics into the by-g-point gas optics, overwriting the gas-optics dataset
in place so the downstream `rte.solve` sees the combined all-sky optics.

Cloud PROPERTIES (clwp, cip, liquid/ice effective radii) come from the already-
GT4Py `progcld4`/`progcld5` and are the inputs; this port does NOT recompute
them. The physical model is purely by-band cloud optics (RRTMGP), so there is no
McICA / subcolumn sampling here (that was the RRTMG era).

LUT tables are read from the pyRTE CloudOptics coefficient dataset and padded
once to the stencil alias shape, mirroring `GasOpticsGT4Py`'s table handling.
float32-stored fields (water paths, radii) are upcast to float64 to match the
bytes the compiled Fortran evaluates; the float-factor promotion trap is avoided
by the explicit `.astype(np.float64)`.

pyRTE indices are 1-based and converted to 0-based here.
"""

import numpy as np

from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Int

from pyrte_rrtmgp.rrtmgp import CloudOptics

from .cloud_optics import (
    NBND_CLD,
    NSIZE_CLD,
    cloud_lut_band,
    inc_1scalar_by_1scalar,
    inc_2stream_by_2stream,
)

# 3*tiny(float64): the `eps` guard in the Fortran optical-props kernels
# (mo_optical_props_kernels.F90) and in the cloud-optics combine.
EPS = 3.0 * float(np.finfo(np.float64).tiny)


class CloudOpticsGT4Py:
    """pyRTE `CloudOptics` with the LUT gather + increment replaced by GT4Py."""

    def __init__(
        self, cloud_optics_file, nx, ny, nz, backend, nhalo=3, ice_roughness=1
    ):
        # pyRTE's CloudOptics.compute hardcodes ice_roughness = 1 (middle of the
        # 3 nrghice categories) when slicing extice/ssaice/asyice, so default to
        # 1 to match the reference.
        self._pyrte = CloudOptics(cloud_optics_file=cloud_optics_file)
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self._backend = backend
        self._icergh = int(ice_roughness)

        ds = self._coeff_dataset()

        # dims (fall back to common alternate names if pyRTE renamed them).
        self.nsize_liq = int(self._dim(ds, "nsize_liq"))
        self.nsize_ice = int(self._dim(ds, "nsize_ice"))
        self.nbnd = int(self._dim(ds, "nband", "bnd"))
        assert self.nsize_liq <= NSIZE_CLD, (self.nsize_liq, NSIZE_CLD)
        assert self.nsize_ice <= NSIZE_CLD, (self.nsize_ice, NSIZE_CLD)
        assert self.nbnd <= NBND_CLD, (self.nbnd, NBND_CLD)

        # LUT constants + step sizes (Fortran load_lut).
        self.radliq_lwr = float(self._var(ds, "radliq_lwr"))
        self.radliq_upr = float(self._var(ds, "radliq_upr"))
        # Ice LUT is indexed by effective DIAMETER (diamice_*); liquid by radius.
        self.radice_lwr = float(self._var(ds, "diamice_lwr", "radice_lwr"))
        self.radice_upr = float(self._var(ds, "diamice_upr", "radice_upr"))
        self.liq_step = (self.radliq_upr - self.radliq_lwr) / (self.nsize_liq - 1)
        self.ice_step = (self.radice_upr - self.radice_lwr) / (self.nsize_ice - 1)

        # LUT tables, padded to (NSIZE_CLD, NBND_CLD).
        self._ext_liq = self._pad_liq(ds, "lut_extliq", "extliq")
        self._ssa_liq = self._pad_liq(ds, "lut_ssaliq", "ssaliq")
        self._asy_liq = self._pad_liq(ds, "lut_asyliq", "asyliq")
        self._ext_ice = self._pad_ice(ds, "lut_extice", "extice")
        self._ssa_ice = self._pad_ice(ds, "lut_ssaice", "ssaice")
        self._asy_ice = self._pad_ice(ds, "lut_asyice", "asyice")

        # factories + stencils (layer-sized).
        sf, qf = get_factories_single_tile(
            nx=self.nx, ny=self.ny, nz=self.nz, nhalo=nhalo, backend=backend
        )
        gi = sf.grid_indexing
        self._sf, self._qf, self._gi = sf, qf, gi
        self._lut = sf.from_origin_domain(
            func=cloud_lut_band,
            origin=gi.origin_compute(),
            domain=gi.domain_compute(),
        )
        self._inc1 = sf.from_origin_domain(
            func=inc_1scalar_by_1scalar,
            origin=gi.origin_compute(),
            domain=gi.domain_compute(),
        )
        self._inc2 = sf.from_origin_domain(
            func=inc_2stream_by_2stream,
            externals=dict(eps=EPS),
            origin=gi.origin_compute(),
            domain=gi.domain_compute(),
        )

    # --------------------------------------------------------- table loading
    def _coeff_dataset(self):
        """The raw cloud-coefficient dataset pyRTE loaded (like GasOptics)."""
        for attr in ("_ds", "_dataset", "_cloud_optics", "dataset"):
            ds = getattr(self._pyrte, attr, None)
            if ds is not None and hasattr(ds, "sizes"):
                return ds
        raise AttributeError(
            "could not find the pyRTE CloudOptics coefficient dataset; "
            "confirm the attribute name against the installed pyRTE"
        )

    @staticmethod
    def _dim(ds, *names):
        for n in names:
            if n in ds.sizes:
                return ds.sizes[n]
        raise KeyError(f"none of {names} in dataset dims {list(ds.sizes)}")

    @staticmethod
    def _var(ds, *names):
        for n in names:
            if n in ds:
                return ds[n].values
        raise KeyError(f"none of {names} in dataset vars {list(ds.data_vars)}")

    def _get(self, ds, *names):
        for n in names:
            if n in ds:
                return ds[n]
        raise KeyError(f"none of {names} in dataset vars {list(ds.data_vars)}")

    def _pad_liq(self, ds, *names):
        """Liquid LUT -> (NSIZE_CLD, NBND_CLD), arranged (nsize_liq, nband)."""
        a = np.ascontiguousarray(self._get(ds, *names).values, dtype=np.float64)
        if a.shape[0] == self.nbnd and a.shape[1] == self.nsize_liq:
            a = a.T  # tolerate (nband, nsize) layout
        assert a.shape == (self.nsize_liq, self.nbnd), a.shape
        out = np.zeros((NSIZE_CLD, NBND_CLD), dtype=np.float64)
        out[: self.nsize_liq, : self.nbnd] = a
        return out

    def _pad_ice(self, ds, *names):
        """Ice LUT -> (NSIZE_CLD, NBND_CLD) at the selected ice roughness.

        The 3-D ice table has axes {nsize_ice, nband, nrghice} in some order;
        the roughness axis is identified by its size, selected at `_icergh`, and
        the remaining two are arranged to (nsize_ice, nband).
        """
        da = self._get(ds, *names)
        a = np.ascontiguousarray(da.values, dtype=np.float64)
        nrgh = self._dim(ds, "nrghice")
        rgh_axis = next(i for i, s in enumerate(a.shape) if s == nrgh)
        a = np.take(a, self._icergh, axis=rgh_axis)  # -> 2-D (two of the dims)
        if a.shape[0] == self.nbnd and a.shape[1] == self.nsize_ice:
            a = a.T
        assert a.shape == (self.nsize_ice, self.nbnd), a.shape
        out = np.zeros((NSIZE_CLD, NBND_CLD), dtype=np.float64)
        out[: self.nsize_ice, : self.nbnd] = a
        return out

    # ----------------------------------------------------------------- utils
    def _ff(self):
        return self._qf.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def _fi(self):
        return self._qf.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Int)

    def _tile(self, da, rest):
        """DataArray -> (nx, ny, *rest), folding the non-core column dims.

        Cloud input fields (lwp/iwp/rel/rei) may lack some non-core dims (e.g. a
        site-only cloud state with no `expt`); broadcast over the missing ones,
        exactly as pyRTE's own compute broadcasts the cloud inputs against the
        atmosphere, so the fold matches the reference.
        """
        for d in self._noncore:
            if d not in da.dims:
                da = da.expand_dims({d: self._noncore_sizes[d]})
        arr = da.transpose(*self._noncore, *rest).values
        rest_shape = tuple(da.sizes[r] for r in rest)
        return np.ascontiguousarray(arr).reshape(self.nx, self.ny, *rest_shape)

    def _write(self, base, var, result, core_dims):
        """Overwrite base[var] (dims = non-core + core_dims) with `result`."""
        da = base[var]
        b = da.transpose(*self._noncore, *core_dims)
        core_shape = result.shape[2:]
        flat = result.reshape(self.nx * self.ny, *core_shape)
        arr = flat.reshape(*[base.sizes[d] for d in self._noncore], *core_shape)
        base[var] = b.copy(data=arr).transpose(*da.dims)

    def _idx_fint(self, re, offset, step, nsteps):
        """Host side of `compute_all_from_table`: re -> (idx0, fint).

        idx0 is Fortran `index - 1`, clamped to [0, nsteps-2]; the upper clamp is
        the Fortran `min(..., nsteps-1)`, the lower clamp only touches no-cloud
        cells (re arbitrary there) whose tau is zeroed anyway by lwp = 0.
        """
        val = (re - offset) / step
        idx0 = np.minimum(np.floor(val).astype(np.int64), nsteps - 2)
        idx0 = np.clip(idx0, 0, nsteps - 2)
        fint = val - idx0
        return idx0.astype(np.int64), fint

    def _spec_dim(self, ds, var="tau"):
        """Spectral dim of an optical-props var (the by-band 'gpt'/'bnd')."""
        layer = ds.mapping.get_dim("layer")
        noncore = [d for d in ds[var].dims if d != layer]
        # the spectral dim is the one carrying the band count; the rest are the
        # per-column non-core dims.
        for d in ds[var].dims:
            if d != layer and ds.sizes[d] == self.nbnd:
                return d, [c for c in ds[var].dims if c not in (layer, d)]
        # fall back: the last dim is spectral (pyRTE puts gpt last).
        d = ds[var].dims[-1]
        return d, [c for c in ds[var].dims if c not in (layer, d)]

    # --------------------------------------------------------------- compute
    def compute(
        self,
        atmosphere,
        problem_type,
        variable_mapping=None,
        add_to_input=False,
    ):
        # 1. pyRTE builds the structured by-band output (and sets mapping).
        base = self._pyrte.compute(
            atmosphere,
            problem_type=problem_type,
            add_to_input=add_to_input,
            variable_mapping=variable_mapping,
        )
        is_2str = ("ssa" in base) and ("g" in base)
        self._is_2str = is_2str

        layer_dim = base.mapping.get_dim("layer")
        self._layer_dim = layer_dim
        spec_dim, noncore = self._spec_dim(base, "tau")
        self._spec_dim = spec_dim
        self._noncore = noncore
        self._noncore_sizes = {d: int(base.sizes[d]) for d in noncore}
        ncol = int(np.prod([base.sizes[d] for d in noncore]))
        assert ncol == self.nx * self.ny, (ncol, self.nx, self.ny)

        # 2. cloud inputs (upcast to float64 to match the Fortran bytes).
        m = atmosphere.mapping
        lwp = self._tile(atmosphere[m.get_var("lwp")], [layer_dim]).astype(np.float64)
        iwp = self._tile(atmosphere[m.get_var("iwp")], [layer_dim]).astype(np.float64)
        rel = self._tile(atmosphere[m.get_var("rel")], [layer_dim]).astype(np.float64)
        rei = self._tile(atmosphere[m.get_var("rei")], [layer_dim]).astype(np.float64)

        # 3. per-band LUT lookup, liquid then ice.
        ltau, ltaussa, ltaussag = self._lut_phase(
            lwp, rel, self.radliq_lwr, self.liq_step, self.nsize_liq,
            self._ext_liq, self._ssa_liq, self._asy_liq,
        )
        itau, itaussa, itaussag = self._lut_phase(
            iwp, rei, self.radice_lwr, self.ice_step, self.nsize_ice,
            self._ext_ice, self._ssa_ice, self._asy_ice,
        )

        # 4. combine liquid + ice (Fortran cloud_optics select type).
        if is_2str:
            tau = ltau + itau
            taussa = ltaussa + itaussa
            taussag = ltaussag + itaussag
            ssa = taussa / np.maximum(EPS, tau)
            g = taussag / np.maximum(EPS, taussa)
            self._write(base, "tau", tau, [layer_dim, spec_dim])
            self._write(base, "ssa", ssa, [layer_dim, spec_dim])
            self._write(base, "g", g, [layer_dim, spec_dim])
        else:
            # ABSORPTION (1scl): tau = (ltau-ltaussa) + (itau-itaussa).
            tau = (ltau - ltaussa) + (itau - itaussa)
            self._write(base, "tau", tau, [layer_dim, spec_dim])

        return base

    def _lut_phase(self, wp, re, offset, step, nsteps, ext, ssa, asy):
        """All-band LUT lookup for one phase -> three (nx, ny, nz, nbnd)."""
        nx, ny, nz, nbnd = self.nx, self.ny, self.nz, self.nbnd
        idx0, fint = self._idx_fint(re, offset, step, nsteps)

        lwp_q, fint_q = self._ff(), self._ff()
        idx_q, iband_q = self._fi(), self._fi()
        tau_q, taussa_q, taussag_q = self._ff(), self._ff(), self._ff()
        lwp_q.view[:] = wp
        fint_q.view[:] = fint
        idx_q.view[:] = idx0

        tau = np.zeros((nx, ny, nz, nbnd), dtype=np.float64)
        taussa = np.zeros((nx, ny, nz, nbnd), dtype=np.float64)
        taussag = np.zeros((nx, ny, nz, nbnd), dtype=np.float64)
        for b in range(nbnd):
            iband_q.view[:] = b
            self._lut(
                lwp=lwp_q, fint=fint_q, idx=idx_q, iband=iband_q,
                ext=ext, ssa=ssa, asy=asy,
                tau=tau_q, taussa=taussa_q, taussag=taussag_q,
            )
            tau[:, :, :, b] = tau_q.view[:]
            taussa[:, :, :, b] = taussa_q.view[:]
            taussag[:, :, :, b] = taussag_q.view[:]
        return tau, taussa, taussag

    # ---------------------------------------------------------------- add_to
    @staticmethod
    def gpt2band_from_limits(band_lims_gpt):
        """0-based (ngpt,) band index per g-point from (2, nbnd) 1-based limits.

        `band_lims_gpt` is pyRTE's `bnd_limits_gpt`: start/end g-point (1-based,
        inclusive) of each band. Mirrors the gpt_lims loop in the Fortran
        inc_*_bybnd kernels.
        """
        band_lims_gpt = np.asarray(band_lims_gpt)
        nbnd = band_lims_gpt.shape[1]
        ngpt = int(band_lims_gpt.max())
        gpt2band = np.empty(ngpt, dtype=np.int64)
        for b in range(nbnd):
            s0 = int(band_lims_gpt[0, b]) - 1
            e0 = int(band_lims_gpt[1, b])
            gpt2band[s0:e0] = b
        return gpt2band

    def add_to(self, gas_optics, cloud_output, gpt2band):
        """Fold the by-band cloud optics into the by-g-point gas optics in place.

        `cloud_output` is the dataset returned by `compute` (by band);
        `gas_optics` is the pyRTE gas-optics output to be incremented;
        `gpt2band` is the 0-based (ngpt,) band index per g-point
        (`gpt2band_from_limits(gas_optics_file bnd_limits_gpt)`). The gas-optics
        dataset is overwritten in place (TWO_STREAM: tau/ssa/g; ABSORPTION:
        tau), reproducing `cloud_props.rte.add_to(gas_optics)`.
        """
        nx, ny, nz = self.nx, self.ny, self.nz
        gpt2band = np.asarray(gpt2band, dtype=np.int64)

        glayer = gas_optics.mapping.get_dim("layer")
        gspec, gnoncore = self._gas_spec_dim(gas_optics)
        ngpt = int(gas_optics.sizes[gspec])
        assert gpt2band.shape == (ngpt,), (gpt2band.shape, ngpt)

        # Tile cloud (by band) AND gas (by g-point) with the SAME non-core column
        # ordering so the (nx, ny) fold aligns column-for-column between them.
        save_noncore = self._noncore
        self._noncore = gnoncore  # _tile reads self._noncore
        ctau = self._tile(cloud_output["tau"], [self._layer_dim, self._spec_dim])
        gtau = self._tile(gas_optics["tau"], [glayer, gspec])
        if self._is_2str:
            cssa = self._tile(cloud_output["ssa"], [self._layer_dim, self._spec_dim])
            cg = self._tile(cloud_output["g"], [self._layer_dim, self._spec_dim])
            gssa = self._tile(gas_optics["ssa"], [glayer, gspec])
            gg = self._tile(gas_optics["g"], [glayer, gspec])

        # per-g-point increment around the compile-once stencil.
        t1_q, t2_q = self._ff(), self._ff()
        if self._is_2str:
            s1_q, g1_q, s2_q, g2_q = self._ff(), self._ff(), self._ff(), self._ff()

        for gpt in range(ngpt):
            b = int(gpt2band[gpt])
            t1_q.view[:] = gtau[:, :, :, gpt]
            t2_q.view[:] = ctau[:, :, :, b]
            if self._is_2str:
                s1_q.view[:] = gssa[:, :, :, gpt]
                g1_q.view[:] = gg[:, :, :, gpt]
                s2_q.view[:] = cssa[:, :, :, b]
                g2_q.view[:] = cg[:, :, :, b]
                self._inc2(
                    tau1=t1_q, ssa1=s1_q, g1=g1_q,
                    tau2=t2_q, ssa2=s2_q, g2=g2_q,
                )
                gtau[:, :, :, gpt] = t1_q.view[:]
                gssa[:, :, :, gpt] = s1_q.view[:]
                gg[:, :, :, gpt] = g1_q.view[:]
            else:
                self._inc1(tau1=t1_q, tau2=t2_q)
                gtau[:, :, :, gpt] = t1_q.view[:]

        # write combined optics back into the gas-optics dataset in place.
        self._write(gas_optics, "tau", gtau, [glayer, gspec])
        if self._is_2str:
            self._write(gas_optics, "ssa", gssa, [glayer, gspec])
            self._write(gas_optics, "g", gg, [glayer, gspec])
        self._noncore = save_noncore
        return gas_optics

    def _gas_spec_dim(self, ds, var="tau"):
        layer = ds.mapping.get_dim("layer")
        d = "gpt" if "gpt" in ds[var].dims else ds[var].dims[-1]
        noncore = [c for c in ds[var].dims if c not in (layer, d)]
        return d, noncore

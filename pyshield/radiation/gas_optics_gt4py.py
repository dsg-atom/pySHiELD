"""GT4Py gas-optics core wired to the pyRTE driver interface.

`GasOpticsGT4Py` is a drop-in for pyRTE's `GasOptics` at the point the driver
(`rte_rrtmgp.py`) calls `.compute(...)`. It keeps two pyRTE pieces on the host --
the interpolation (`interpolate`) and the RTE solver (`.rte.solve`) -- and
replaces only the gas-optics *gather* core (the per-g-point table gathers that
dominate the kernel cost) with the GT4Py stencils validated one by one against
compiled pyRTE:

    interp3d_major_gpt   major-gas kmajor gather (also pfrac, scaling=1)
    interp2d_minor_gpt   minor-gas kminor gather with density/complement scaling
    interp2d_rayl_gpt    shortwave Rayleigh krayl gather
    planck_interp1d      longwave totplnk 1-D temperature interpolation

Container is overwrite-first: `.compute` calls pyRTE `compute(...)` to get a
fully-structured output dataset (right vars, dims, coords, attrs, and the
`.rte`/`.mapping` accessors the solver needs), then overwrites the same-named
data vars in place with the GT4Py results. Nothing about the solve path changes.

Output vars overwritten:
    shortwave (TWO_STREAM)  tau, ssa, g
    longwave  (ABSORPTION)  tau, surface_source, layer_source,
                            level_source, surface_source_jacobian

`toa_source` (shortwave) has no device kernel -- it is a host broadcast of the
TSI-normalized per-g-point solar-source vector -- so it is reproduced here in
xarray straight from the solar-source coefficient tables (`_toa_source`),
mirroring pyRTE `compute_sources`. Unlike the gather vars its non-core dims are
whatever `total_solar_irradiance` carries (per-site for RFMIP), not the full
column set, so it is written outside the (nx, ny) tile machinery.

The gather assembly here is the same code proven in tests/gas_optics
(test_tau_full.py / test_tau_full_sw.py, test_sw.py, test_planck.py); the only
generalization is that those tests index a fixed RFMIP (site, expt) pair while
this class folds an arbitrary set of non-core "column" dims into the stencil
tile (nx, ny). For the driver the column axis is nx_tile*ny_tile row-major, so
folding it back to (nx, ny) is exact; for the RFMIP validation the two non-core
dims (site, expt) fold to (nx, ny) as an identity. pyRTE indices are 1-based and
converted to 0-based here; float32-stored fields are upcast to float64 to match
the bytes the compiled Fortran evaluates.
"""

import numpy as np

from ndsl.boilerplate import get_factories_single_tile
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float, Int

from pyrte_rrtmgp.rrtmgp import DEFAULT_GAS_MAPPING, GasOptics

from .gas_optics import (
    NBND,
    NMINORK,
    NPLANCKTEMP,
    interp2d_minor_gpt,
    interp2d_rayl_gpt,
    interp3d_major_gpt,
    planck_interp1d,
)

# 8-point major-gas weight -> (eta, press, temp) corner it multiplies.
_FMAJ_AXES = {
    "f111": (0, 0, 0), "f211": (1, 0, 0), "f121": (0, 1, 0), "f221": (1, 1, 0),
    "f112": (0, 0, 1), "f212": (1, 0, 1), "f122": (0, 1, 1), "f222": (1, 1, 1),
}


class GasOpticsGT4Py:
    """pyRTE `GasOptics` with the gather core replaced by GT4Py stencils."""

    def __init__(self, gas_optics_file, nx, ny, nz, backend, nhalo=3):
        self._pyrte = GasOptics(gas_optics_file=gas_optics_file)
        self.is_sw = type(self._pyrte).__name__ == "SWGasOptics"
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self._backend = backend

        ds = self._pyrte._dataset
        self.ngpt = int(ds.sizes["gpt"])

        # --- factories + stencils (layer-sized; a level-sized pair for LW) ----
        sf, qf = get_factories_single_tile(
            nx=self.nx, ny=self.ny, nz=self.nz, nhalo=nhalo, backend=backend
        )
        gi = sf.grid_indexing
        self._sf, self._qf, self._gi = sf, qf, gi
        self._major = sf.from_origin_domain(
            func=interp3d_major_gpt, origin=gi.origin_compute(), domain=gi.domain_compute()
        )
        self._minor = sf.from_origin_domain(
            func=interp2d_minor_gpt, origin=gi.origin_compute(), domain=gi.domain_compute()
        )
        if self.is_sw:
            self._rayl = sf.from_origin_domain(
                func=interp2d_rayl_gpt, origin=gi.origin_compute(), domain=gi.domain_compute()
            )
        else:
            self._planck_lay = sf.from_origin_domain(
                func=planck_interp1d, origin=gi.origin_compute(), domain=gi.domain_compute()
            )
            sf_lev, qf_lev = get_factories_single_tile(
                nx=self.nx, ny=self.ny, nz=self.nz + 1, nhalo=nhalo, backend=backend
            )
            gi_lev = sf_lev.grid_indexing
            self._qf_lev = qf_lev
            self._planck_lev = sf_lev.from_origin_domain(
                func=planck_interp1d,
                origin=gi_lev.origin_compute(),
                domain=gi_lev.domain_compute(),
            )

        # --- coefficient tables, padded once to the stencil alias shapes ------
        # kmajor -> (14,9,60,256); SW (224) is zero-padded on the g-point axis,
        # LW (256) is an exact fit.
        kmaj = np.ascontiguousarray(
            ds["kmajor"]
            .transpose("temperature", "mixing_fraction", "pressure_interp", "gpt")
            .values
        )
        self._kmajor = np.zeros((14, 9, 60, 256), dtype=np.float64)
        self._kmajor[:, :, :, : self.ngpt] = kmaj

        # kminor lower/upper -> (14,9,NMINORK), plus the reduction metadata used
        # to select and scale contributing intervals at compute time.
        self._minor_tabs = {}
        for suffix in ("lower", "upper"):
            kmin = np.ascontiguousarray(
                ds[f"kminor_{suffix}"]
                .transpose("temperature", "mixing_fraction", f"contributors_{suffix}")
                .values
            )
            nk = kmin.shape[2]
            assert nk <= NMINORK, (suffix, nk, NMINORK)
            kmin_pad = np.zeros((14, 9, NMINORK), dtype=np.float64)
            kmin_pad[:, :, :nk] = kmin
            names = self._pyrte.extract_names(ds[f"minor_gases_{suffix}"].data)
            scal_names = self._pyrte.extract_names(ds[f"scaling_gas_{suffix}"].data)
            self._minor_tabs[suffix] = dict(
                kmin=kmin_pad,
                names=names,
                idx_minor=[self._idx_of(n) for n in names],
                idx_scal=[self._idx_of(n) for n in scal_names],
                limits=ds[f"minor_limits_gpt_{suffix}"]
                .transpose(f"minor_absorber_intervals_{suffix}", "pair")
                .values,  # (nminor,2) 1-based
                kstart=ds[f"kminor_start_{suffix}"].values,  # 1-based
                sdens=ds[f"minor_scales_with_density_{suffix}"].values.astype(bool),
                scomp=ds[f"scale_by_complement_{suffix}"].values.astype(bool),
            )

        self._idx_h2o = self._pyrte._selected_gas_names_ext.index("h2o")
        # (ngpt,2) 1-based flavor per g-point per atmosphere half.
        self._gpoint_flavor = self._pyrte.gpoint_flavor.transpose(
            "gpt", "atmos_layer"
        ).values

        if self.is_sw:
            # krayl halves (14,9,224); same (temp,eta) layout as kminor.
            self._krayl = {
                b: np.ascontiguousarray(
                    ds[f"rayl_{b}"]
                    .transpose("temperature", "mixing_fraction", "gpt")
                    .values
                )
                for b in ("lower", "upper")
            }
        else:
            # plank_fraction (14,9,60,256, same layout as kmajor) + totplnk (196,16).
            self._pfracin = np.ascontiguousarray(
                ds["plank_fraction"]
                .transpose("temperature", "mixing_fraction", "pressure_interp", "gpt")
                .values
            )
            self._totplnk = np.ascontiguousarray(
                ds["totplnk"].transpose("temperature_Planck", "bnd").values
            )
            assert self._totplnk.shape == (NPLANCKTEMP, NBND), self._totplnk.shape
            self._band_lims = ds["bnd_limits_gpt"].transpose("pair", "bnd").values
            assert self._band_lims.shape[1] == NBND, self._band_lims.shape
            self._nplancktemp = int(ds.sizes["temperature_Planck"])
            tmin = float(ds["temp_ref"].min())
            tmax = float(ds["temp_ref"].max())
            self._temp_ref_min = tmin
            self._totplnk_delta = (tmax - tmin) / (self._nplancktemp - 1)

    # ------------------------------------------------------------------ utils
    def _idx_of(self, name):
        """0-based index into `_selected_gas_names_ext` for a minor/scaling gas.

        Empty name -> 0; the Fortran uses idx_scaling > 0 as the "has a scaling
        gas" gate, so a padded/blank scaling name reads as no scaling.
        """
        name = name.strip()
        if not name:
            return 0
        return int(self._pyrte.get_idx_minor(np.array([name]))[0])

    def _ff(self, factory):
        return factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    def _fi(self, factory):
        return factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Int)

    def _tile(self, da, rest):
        """DataArray -> ndarray (nx, ny, *rest), folding the non-core dims.

        `rest` are the per-cell axes to keep (layer, gpt-corner weights, ...).
        The non-core "column" dims are moved first, in `self._noncore` order,
        then reshaped to (nx, ny): the driver's single flattened column axis
        splits into the tile, and RFMIP's (site, expt) fold as an identity.
        """
        arr = da.transpose(*self._noncore, *rest).values
        rest_shape = tuple(da.sizes[r] for r in rest)
        return np.ascontiguousarray(arr).reshape(self.nx, self.ny, *rest_shape)

    def _write(self, base, var, result, core_dims):
        """Overwrite base[var] (dims = non-core + core_dims) with `result`.

        `result` is (nx, ny, *core_shape); it is folded back to the non-core
        column layout and re-inserted in base[var]'s original dim order so the
        RTE solver sees an unchanged container.
        """
        da = base[var]
        b = da.transpose(*self._noncore, *core_dims)
        core_shape = result.shape[2:]
        flat = result.reshape(self.nx * self.ny, *core_shape)
        arr = flat.reshape(*[base.sizes[d] for d in self._noncore], *core_shape)
        base[var] = b.copy(data=arr).transpose(*da.dims)

    def _gas_mapping(self, atmosphere, gas_name_map):
        """Reproduce pyRTE `compute`'s gas_mapping so `interpolate` matches."""
        gas_mapping = {}
        if gas_name_map is None:
            data_vars = list(atmosphere.data_vars)
            for gas, valid_names in DEFAULT_GAS_MAPPING.items():
                for v in data_vars:
                    if v in valid_names:
                        gas_mapping[gas] = v
        else:
            for gas in DEFAULT_GAS_MAPPING:
                if gas in gas_name_map:
                    gas_mapping[gas] = gas_name_map[gas]
        return gas_mapping

    @staticmethod
    def _idx_frac(temp, temp_ref_min, totplnk_delta, nplancktemp):
        """Host side of Fortran interpolate1D: temperature -> (idx0, frac)."""
        val0 = (temp - temp_ref_min) / totplnk_delta
        iv = val0.astype(np.int64)  # trunc toward zero, as Fortran int()
        frac = val0 - iv
        idx0 = np.clip(iv, 0, nplancktemp - 2).astype(np.int64)
        return idx0, frac

    def _toa_source(self, atmosphere, base):
        """Reproduce pyRTE `SWGasOptics.compute_sources` -> toa_source.

        Host glue, no device kernel: the per-g-point solar-source vector is
        combined from its quiet/facular/sunspot coefficient tables (the two
        magnetic offsets are the pyRTE constants), then either scaled to a
        supplied `total_solar_irradiance` per column or normalized to the
        default TSI and broadcast over the non-core column dims. This mirrors
        pyRTE exactly and overwrites base["toa_source"] in place; it is written
        against toa_source's OWN dims (whatever `total_solar_irradiance` carries,
        e.g. per-site for RFMIP), which are a subset of the gather vars' non-core
        dims, so it does not use the (nx, ny) tile machinery.
        """
        a_offset = 0.1495954
        b_offset = 0.00066696
        ds = self._pyrte._dataset
        solar_source = (
            ds["solar_source_quiet"]
            + (ds["mg_default"] - a_offset) * ds["solar_source_facular"]
            + (ds["sb_default"] - b_offset) * ds["solar_source_sunspot"]
        )
        if "total_solar_irradiance" in atmosphere:
            tsi = atmosphere["total_solar_irradiance"]
            toa_flux = solar_source.broadcast_like(tsi)
            def_tsi = toa_flux.sum(dim="gpt")
            toa = (toa_flux * (tsi / def_tsi)).rename("toa_source")
        else:
            norm = 1.0 / solar_source.sum(dim="gpt")
            toa = (solar_source * ds["tsi_default"] * norm).rename("toa_source")
            non_default = [
                d
                for d in atmosphere.dims
                if d not in (self._layer_dim, self._level_dim, "gpt")
            ]
            for dim in non_default:
                toa = toa.expand_dims({dim: atmosphere[dim]})
        dst = base["toa_source"]
        base["toa_source"] = dst.copy(data=toa.transpose(*dst.dims).values)

    # ----------------------------------------------------------------- driver
    def compute(
        self,
        atmosphere,
        problem_type,
        gas_name_map=None,
        variable_mapping=None,
        add_to_input=False,
    ):
        # 1. pyRTE builds the structured output (and, as a side effect, sets
        #    self._pyrte._gas_names, atmosphere.mapping, and top_at_1).
        base = self._pyrte.compute(
            atmosphere,
            problem_type=problem_type,
            add_to_input=add_to_input,
            gas_name_map=gas_name_map,
            variable_mapping=variable_mapping,
        )

        # 2. interpolation intermediates (re-run; overwrite-first accepts this).
        gas_mapping = self._gas_mapping(atmosphere, gas_name_map)
        interp = self._pyrte.interpolate(atmosphere, gas_mapping)

        # 3. non-core "column" dims from the output tau; fold to (nx, ny).
        layer_dim = base.mapping.get_dim("layer")
        level_dim = base.mapping.get_dim("level")
        self._layer_dim, self._level_dim = layer_dim, level_dim
        self._noncore = [
            d for d in base["tau"].dims if d not in (layer_dim, level_dim, "gpt")
        ]
        ncol = int(np.prod([base.sizes[d] for d in self._noncore]))
        assert ncol == self.nx * self.ny, (ncol, self.nx, self.ny)

        tau_abs = self._tau_absorption(atmosphere, interp)

        if self.is_sw:
            tau_rayl = self._tau_rayleigh(interp)
            tau = tau_abs + tau_rayl
            tiny = 2.0 * np.finfo(np.float64).tiny
            ssa = np.where(tau > tiny, tau_rayl / tau, 0.0)
            g = np.zeros_like(tau)
            self._write(base, "tau", tau, [layer_dim, "gpt"])
            self._write(base, "ssa", ssa, [layer_dim, "gpt"])
            self._write(base, "g", g, [layer_dim, "gpt"])
            self._toa_source(atmosphere, base)
        else:
            self._write(base, "tau", tau_abs, [layer_dim, "gpt"])
            sfc, lay, lev, jac = self._planck(atmosphere, interp, base)
            self._write(base, "surface_source", sfc, ["gpt"])
            self._write(base, "layer_source", lay, [layer_dim, "gpt"])
            self._write(base, "level_source", lev, [level_dim, "gpt"])
            self._write(base, "surface_source_jacobian", jac, ["gpt"])

        return base

    # ------------------------------------------------------------- gathers
    def _shared_interp(self, atmosphere, interp):
        """Tiled interpolation intermediates shared across the gathers."""
        ld = self._layer_dim
        nx, ny, nz = self.nx, self.ny, self.nz
        iv = {}
        iv["jtemp0"] = (self._tile(interp["temperature_index"], [ld]) - 1).astype(np.int64)
        jpress_f = self._tile(interp["pressure_index"], [ld])
        tropo = self._tile(interp["tropopause_mask"], [ld]).astype(bool)
        iv["tropo"] = tropo
        itropo = np.where(tropo, 0, 1).astype(np.int64)  # 0=lower/tropo, 1=upper
        iv["itropo"] = itropo
        iv["jpress0"] = (jpress_f + itropo - 1).astype(np.int64)
        # (nx,ny,2,nz,nflav) -> col_mix[:,:,0/1]
        iv["col_mix"] = self._tile(interp["column_mix"], ["temp_interp", ld, "flavor"])
        # (nx,ny,2,2,2,nz,nflav)
        iv["fmajor"] = self._tile(
            interp["fmajor"], ["eta_interp", "press_interp", "temp_interp", ld, "flavor"]
        )
        # (nx,ny,2,2,nz,nflav)
        iv["fminor"] = self._tile(
            interp["fminor"], ["eta_interp", "temp_interp", ld, "flavor"]
        )
        # (nx,ny,2,nz,nflav)
        iv["jeta"] = self._tile(interp["eta_index"], ["pair", ld, "flavor"])
        iv["si"], iv["ei"], iv["li"] = np.indices((nx, ny, nz))
        return iv

    def _tau_absorption(self, atmosphere, interp):
        """Major-gas + minor-gas absorption tau -> (nx, ny, nz, ngpt)."""
        nx, ny, nz, ngpt = self.nx, self.ny, self.nz, self.ngpt
        ld = self._layer_dim
        iv = self._shared_interp(atmosphere, interp)
        si, ei, li = iv["si"], iv["ei"], iv["li"]
        gpf = self._gpoint_flavor
        col_mix, fmajor, fminor, jeta = (
            iv["col_mix"], iv["fmajor"], iv["fminor"], iv["jeta"]
        )

        qf = self._qf
        s1_q, s2_q = self._ff(qf), self._ff(qf)
        fmaj_q = {n: self._ff(qf) for n in _FMAJ_AXES}
        jtemp_q, jpress_q = self._fi(qf), self._fi(qf)
        jeta1_q, jeta2_q, igpt_q, kg_q = (
            self._fi(qf), self._fi(qf), self._fi(qf), self._fi(qf)
        )
        fmn_q = {n: self._ff(qf) for n in ("fmn11", "fmn21", "fmn12", "fmn22")}
        res_q = self._ff(qf)
        jtemp_q.view[:] = iv["jtemp0"]

        # --- major gas ------------------------------------------------------
        tau_major = np.zeros((nx, ny, nz, ngpt), dtype=np.float64)
        jpress_q.view[:] = iv["jpress0"]
        for g in range(ngpt):
            iflav = (gpf[g, iv["itropo"]] - 1).astype(np.int64)
            s1_q.view[:] = col_mix[:, :, 0][si, ei, li, iflav]
            s2_q.view[:] = col_mix[:, :, 1][si, ei, li, iflav]
            for name, (e, p, t) in _FMAJ_AXES.items():
                fmaj_q[name].view[:] = fmajor[:, :, e, p, t][si, ei, li, iflav]
            jeta1_q.view[:] = (jeta[:, :, 0][si, ei, li, iflav] - 1).astype(np.int64)
            jeta2_q.view[:] = (jeta[:, :, 1][si, ei, li, iflav] - 1).astype(np.int64)
            igpt_q.view[:] = g
            self._major(
                scaling1=s1_q, scaling2=s2_q,
                f111=fmaj_q["f111"], f211=fmaj_q["f211"], f121=fmaj_q["f121"],
                f221=fmaj_q["f221"], f112=fmaj_q["f112"], f212=fmaj_q["f212"],
                f122=fmaj_q["f122"], f222=fmaj_q["f222"],
                jtemp=jtemp_q, jpress=jpress_q, jeta1=jeta1_q, jeta2=jeta2_q,
                igpt=igpt_q, kmajor=self._kmajor, res=res_q,
            )
            tau_major[:, :, :, g] = res_q.view[:]

        # --- minor gas (lower then upper) -----------------------------------
        col_gas = self._tile(interp["gases_columns"], [ld, "gas"]).astype(np.float64)
        pvar = atmosphere.mapping.get_var("pres_layer")
        tvar = atmosphere.mapping.get_var("temp_layer")
        tmpl = interp["temperature_index"]
        play = self._tile(atmosphere[pvar].broadcast_like(tmpl), [ld]).astype(np.float64)
        tlay = self._tile(atmosphere[tvar].broadcast_like(tmpl), [ld]).astype(np.float64)

        tau_minor = np.zeros((nx, ny, nz, ngpt), dtype=np.float64)
        gas_names = self._pyrte._gas_names
        for suffix, atmos_half, layer_mask in (
            ("lower", 0, iv["tropo"]),
            ("upper", 1, ~iv["tropo"]),
        ):
            tab = self._minor_tabs[suffix]
            kmin = tab["kmin"]
            names = tab["names"]
            mask = np.isin(names, gas_names)
            limits, kstart = tab["limits"], tab["kstart"]
            sdens, scomp = tab["sdens"], tab["scomp"]
            gpt_flv = gpf[:, atmos_half]  # 1-based flavor per g-point

            for imnr in range(len(names)):
                if not mask[imnr]:
                    continue
                gptS0, gptE0 = int(limits[imnr, 0]) - 1, int(limits[imnr, 1]) - 1
                ks0 = int(kstart[imnr]) - 1
                iflav = int(gpt_flv[gptS0]) - 1
                idx_minor = tab["idx_minor"][imnr]
                idx_scal = tab["idx_scal"][imnr]

                scaling = col_gas[:, :, :, idx_minor].copy()
                if sdens[imnr]:
                    scaling = scaling * (0.01 * play / tlay)
                    if idx_scal > 0:
                        vmr = 1.0 / col_gas[:, :, :, 0]
                        dry = 1.0 / (1.0 + col_gas[:, :, :, self._idx_h2o] * vmr)
                        fac = col_gas[:, :, :, idx_scal] * vmr * dry
                        scaling = scaling * ((1.0 - fac) if scomp[imnr] else fac)
                scaling = np.where(layer_mask, scaling, 0.0)

                fmn_q["fmn11"].view[:] = fminor[:, :, 0, 0][si, ei, li, iflav]
                fmn_q["fmn21"].view[:] = fminor[:, :, 1, 0][si, ei, li, iflav]
                fmn_q["fmn12"].view[:] = fminor[:, :, 0, 1][si, ei, li, iflav]
                fmn_q["fmn22"].view[:] = fminor[:, :, 1, 1][si, ei, li, iflav]
                jeta1_q.view[:] = (jeta[:, :, 0][si, ei, li, iflav] - 1).astype(np.int64)
                jeta2_q.view[:] = (jeta[:, :, 1][si, ei, li, iflav] - 1).astype(np.int64)

                for g in range(gptS0, gptE0 + 1):
                    kg_q.view[:] = ks0 + (g - gptS0)
                    self._minor(
                        fmn11=fmn_q["fmn11"], fmn21=fmn_q["fmn21"],
                        fmn12=fmn_q["fmn12"], fmn22=fmn_q["fmn22"],
                        jtemp=jtemp_q, jeta1=jeta1_q, jeta2=jeta2_q, kg=kg_q,
                        kminor=kmin, res=res_q,
                    )
                    tau_minor[:, :, :, g] += scaling * res_q.view[:]

        return tau_major + tau_minor

    def _tau_rayleigh(self, interp):
        """Shortwave Rayleigh tau -> (nx, ny, nz, ngpt)."""
        nx, ny, nz, ngpt = self.nx, self.ny, self.nz, self.ngpt
        ld = self._layer_dim
        # Rayleigh needs only fminor / eta_index / jtemp / tropo; build directly.
        jtemp0 = (self._tile(interp["temperature_index"], [ld]) - 1).astype(np.int64)
        tropo = self._tile(interp["tropopause_mask"], [ld]).astype(bool)
        fminor = self._tile(
            interp["fminor"], ["eta_interp", "temp_interp", ld, "flavor"]
        )
        jeta = self._tile(interp["eta_index"], ["pair", ld, "flavor"])
        col_dry = self._tile(interp["gases_columns"].sel(gas="dry_air"), [ld]).astype(
            np.float64
        )
        col_h2o = self._tile(interp["gases_columns"].sel(gas="h2o"), [ld]).astype(
            np.float64
        )
        gpf = self._gpoint_flavor
        si, ei, li = np.indices((nx, ny, nz))

        qf = self._qf
        fmn_q = {n: self._ff(qf) for n in ("fmn11", "fmn21", "fmn12", "fmn22")}
        jtemp_q, jeta1_q, jeta2_q, kg_q = (
            self._fi(qf), self._fi(qf), self._fi(qf), self._fi(qf)
        )
        res_q = self._ff(qf)
        jtemp_q.view[:] = jtemp0

        tau_rayl = np.zeros((nx, ny, nz, ngpt), dtype=np.float64)
        for g in range(ngpt):
            k_half = {}
            for b, bi in (("lower", 0), ("upper", 1)):
                iflav = (gpf[g, bi] - 1).astype(np.int64)
                fmn_q["fmn11"].view[:] = fminor[:, :, 0, 0][si, ei, li, iflav]
                fmn_q["fmn21"].view[:] = fminor[:, :, 1, 0][si, ei, li, iflav]
                fmn_q["fmn12"].view[:] = fminor[:, :, 0, 1][si, ei, li, iflav]
                fmn_q["fmn22"].view[:] = fminor[:, :, 1, 1][si, ei, li, iflav]
                jeta1_q.view[:] = (jeta[:, :, 0][si, ei, li, iflav] - 1).astype(np.int64)
                jeta2_q.view[:] = (jeta[:, :, 1][si, ei, li, iflav] - 1).astype(np.int64)
                kg_q.view[:] = g
                self._rayl(
                    fmn11=fmn_q["fmn11"], fmn21=fmn_q["fmn21"],
                    fmn12=fmn_q["fmn12"], fmn22=fmn_q["fmn22"],
                    jtemp=jtemp_q, jeta1=jeta1_q, jeta2=jeta2_q, kg=kg_q,
                    krayl=self._krayl[b], res=res_q,
                )
                k_half[b] = res_q.view[:].copy()
            k = np.where(tropo, k_half["lower"], k_half["upper"])
            tau_rayl[:, :, :, g] = k * (col_h2o + col_dry)
        return tau_rayl

    def _planck(self, atmosphere, interp, base):
        """Longwave Planck sources -> (surface, layer, level, jacobian)."""
        nx, ny, nz, ngpt = self.nx, self.ny, self.nz, self.ngpt
        ld, lvd = self._layer_dim, self._level_dim
        nlev = nz + 1

        # pfrac reuses the major gather with scaling=1 and the pfracin table.
        iv = self._shared_interp(atmosphere, interp)
        si, ei, li = iv["si"], iv["ei"], iv["li"]
        gpf = self._gpoint_flavor
        fmajor, jeta = iv["fmajor"], iv["jeta"]

        qf = self._qf
        s1_q, s2_q = self._ff(qf), self._ff(qf)
        s1_q.view[:] = 1.0
        s2_q.view[:] = 1.0
        fmaj_q = {n: self._ff(qf) for n in _FMAJ_AXES}
        jtemp_q, jpress_q = self._fi(qf), self._fi(qf)
        jeta1_q, jeta2_q, igpt_q = self._fi(qf), self._fi(qf), self._fi(qf)
        res_q = self._ff(qf)
        jtemp_q.view[:] = iv["jtemp0"]
        jpress_q.view[:] = iv["jpress0"]

        pfrac = np.zeros((nx, ny, nz, ngpt), dtype=np.float64)
        for g in range(ngpt):
            iflav = (gpf[g, iv["itropo"]] - 1).astype(np.int64)
            for name, (e, p, t) in _FMAJ_AXES.items():
                fmaj_q[name].view[:] = fmajor[:, :, e, p, t][si, ei, li, iflav]
            jeta1_q.view[:] = (jeta[:, :, 0][si, ei, li, iflav] - 1).astype(np.int64)
            jeta2_q.view[:] = (jeta[:, :, 1][si, ei, li, iflav] - 1).astype(np.int64)
            igpt_q.view[:] = g
            self._major(
                scaling1=s1_q, scaling2=s2_q,
                f111=fmaj_q["f111"], f211=fmaj_q["f211"], f121=fmaj_q["f121"],
                f221=fmaj_q["f221"], f112=fmaj_q["f112"], f212=fmaj_q["f212"],
                f122=fmaj_q["f122"], f222=fmaj_q["f222"],
                jtemp=jtemp_q, jpress=jpress_q, jeta1=jeta1_q, jeta2=jeta2_q,
                igpt=igpt_q, kmajor=self._pfracin, res=res_q,
            )
            pfrac[:, :, :, g] = res_q.view[:]

        # temperatures (upcast to float64 to match the Fortran index math)
        tlay_var = atmosphere.mapping.get_var("temp_layer")
        tlev_var = atmosphere.mapping.get_var("temp_level")
        tsfc_var = atmosphere.mapping.get_var("surface_temperature")
        tmpl = interp["temperature_index"]
        tlay = self._tile(atmosphere[tlay_var].broadcast_like(tmpl), [ld]).astype(np.float64)
        tlev = self._tile(atmosphere[tlev_var], [lvd]).astype(np.float64)
        tsfc = self._tile(atmosphere[tsfc_var], []).astype(np.float64)

        sfc_lay0 = (nz - 1) if bool(base.attrs["top_at_1"]) else 0

        tmin, delta, ntp = self._temp_ref_min, self._totplnk_delta, self._nplancktemp
        tsfc_b = np.broadcast_to(tsfc[:, :, None], (nx, ny, nz))
        idx_lay, frac_lay = self._idx_frac(tlay, tmin, delta, ntp)
        idx_sfc, frac_sfc = self._idx_frac(tsfc_b, tmin, delta, ntp)
        idx_sfcd, frac_sfcd = self._idx_frac(tsfc_b + 1.0, tmin, delta, ntp)

        frac_lay_q, idx_lay_q = self._ff(qf), self._fi(qf)
        frac_sfc_q, idx_sfc_q = self._ff(qf), self._fi(qf)
        frac_sfcd_q, idx_sfcd_q = self._ff(qf), self._fi(qf)
        iband_q = self._fi(qf)
        planck_lay_q, planck_sfc_q, planck_sfcd_q = self._ff(qf), self._ff(qf), self._ff(qf)
        frac_lay_q.view[:] = frac_lay
        idx_lay_q.view[:] = idx_lay
        frac_sfc_q.view[:] = frac_sfc
        idx_sfc_q.view[:] = idx_sfc
        frac_sfcd_q.view[:] = frac_sfcd
        idx_sfcd_q.view[:] = idx_sfcd

        qfl = self._qf_lev
        frac_lev_q, idx_lev_q = self._ff(qfl), self._fi(qfl)
        iband_lev_q, planck_lev_q = self._fi(qfl), self._ff(qfl)
        idx_lev, frac_lev = self._idx_frac(tlev, tmin, delta, ntp)
        frac_lev_q.view[:] = frac_lev
        idx_lev_q.view[:] = idx_lev

        sfc_src = np.zeros((nx, ny, ngpt), dtype=np.float64)
        jac_src = np.zeros((nx, ny, ngpt), dtype=np.float64)
        lay_src = np.zeros((nx, ny, nz, ngpt), dtype=np.float64)
        lev_src = np.zeros((nx, ny, nlev, ngpt), dtype=np.float64)

        for ibnd in range(self._band_lims.shape[1]):
            gptS0 = int(self._band_lims[0, ibnd]) - 1
            gptE0 = int(self._band_lims[1, ibnd]) - 1
            iband_q.view[:] = ibnd
            iband_lev_q.view[:] = ibnd
            self._planck_lay(frac=frac_lay_q, idx=idx_lay_q, iband=iband_q,
                             totplnk=self._totplnk, planck=planck_lay_q)
            self._planck_lay(frac=frac_sfc_q, idx=idx_sfc_q, iband=iband_q,
                             totplnk=self._totplnk, planck=planck_sfc_q)
            self._planck_lay(frac=frac_sfcd_q, idx=idx_sfcd_q, iband=iband_q,
                             totplnk=self._totplnk, planck=planck_sfcd_q)
            self._planck_lev(frac=frac_lev_q, idx=idx_lev_q, iband=iband_lev_q,
                             totplnk=self._totplnk, planck=planck_lev_q)

            pl_lay = planck_lay_q.view[:]
            pl_sfc = planck_sfc_q.view[:, :, 0]
            pl_sfcd = planck_sfcd_q.view[:, :, 0]
            pl_lev = planck_lev_q.view[:]

            for g in range(gptS0, gptE0 + 1):
                pf = pfrac[:, :, :, g]
                sfc_src[:, :, g] = pf[:, :, sfc_lay0] * pl_sfc
                jac_src[:, :, g] = pf[:, :, sfc_lay0] * (pl_sfcd - pl_sfc)
                lay_src[:, :, :, g] = pf * pl_lay
                lev_src[:, :, 0, g] = pf[:, :, 0] * pl_lev[:, :, 0]
                lev_src[:, :, nz, g] = pf[:, :, nz - 1] * pl_lev[:, :, nz]
                lev_src[:, :, 1:nz, g] = (
                    np.sqrt(pf[:, :, 0 : nz - 1] * pf[:, :, 1:nz]) * pl_lev[:, :, 1:nz]
                )

        return sfc_src, lay_src, lev_src, jac_src

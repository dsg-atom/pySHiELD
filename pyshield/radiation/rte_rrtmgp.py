import datetime

import numpy as np
from pyrte_rrtmgp import rte
from pyrte_rrtmgp.config import DEFAULT_DIM_MAPPING
from pyrte_rrtmgp.input_mapping import AtmosphericMapping
from pyrte_rrtmgp.rrtmgp import CloudOptics
from pyrte_rrtmgp.rrtmgp_data_files import CloudOpticsFiles, GasOpticsFiles

import ndsl.constants as constants
from ndsl import QuantityFactory, StencilFactory
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.gt4py import FORWARD, PARALLEL, computation
from ndsl.dsl.gt4py import function as gtfunction
from ndsl.dsl.gt4py import interval, log
from ndsl.dsl.typing import Bool, Float, FloatField, FloatFieldIJ
from pyshield.stencils.surface import SurfaceState

from ._config import RTE_RRTMGPConfig
from .gas_optics_gt4py import GasOpticsGT4Py
from .lw_solver_gt4py import LWNoScatSolverGT4Py
from .sw_solver_gt4py import SWTwoStreamSolverGT4Py
from .rad_astro import coszmn, sol_init, solar_update
from .rad_clouds import cld_init, progcld4, progcld5
from .rad_gases import co2_update, gas_init, get_gases_bottomup, get_gases_topdown
from .rad_sfc import set_albedo, set_sfcemis, sfc_init
from .state import RTE_RRTMGPState

GRAV = 9.80665
CP_DRY = 1004.64
QMIN = 1.0e-10
QME5 = 1.0e-7
QME6 = 1.0e-7

# Longwave / shortwave g-point counts for the GEOS-L91 coefficient files
# (LW_G256, SW_G224); the clear-sky GT4Py solvers are compiled for these.
NGPT_LW = 256
NGPT_SW = 224


def _tile(da, noncore, rest, nx, ny):
    """DataArray -> (nx, ny, *rest), folding the non-core column dim(s) first.

    Mirrors the fold used by the validated solver tests
    (tests/rte_solver/test_{lw,sw}_solver_gt4py.py). In the driver `noncore` is
    the single flattened `column` dim of length nx*ny (row-major), so this
    unflattens column -> (nx, ny).
    """
    arr = da.transpose(*noncore, *rest).values
    rest_shape = tuple(da.sizes[r] for r in rest)
    return np.ascontiguousarray(arr).reshape(nx, ny, *rest_shape)


def _col(da, noncore, nx, ny):
    """Per-column DataArray -> (nx, ny)."""
    arr = da.transpose(*noncore).values
    return np.ascontiguousarray(arr).reshape(nx, ny)


@gtfunction
def calc_heating_rate(flux_up, flux_down, p_lev):
    """
    Calculates heating rates based on pressures and fluxes,
    assuming k increases with height

    Args:
        flux_up: upward flux
        flux_down: downward flux
        p_lev: model interface pressure
    Returns:
        heating_rate: layer heating rate
    """
    return (
        (flux_up[0, 0, 1] - flux_up - flux_down[0, 0, 1] + flux_down)
        * GRAV
        / (CP_DRY * (p_lev[0, 0, 1] - p_lev))
    )


def calc_tlvl_gfs(
    plyr: FloatField,
    plvl: FloatField,
    tgrs: FloatField,
    tskin: FloatFieldIJ,
    qvapor: FloatField,
    tvly: FloatField,
    tsfa: FloatFieldIJ,
    tlvl: FloatField,
):
    """
    Calculates interface(level) temperatures needed for radiation as in the gfs physics
    """
    with computation(FORWARD):
        with interval(0, 1):
            tsfa = tgrs
            tlvl = tskin
            tem2da = log(plyr)
            tem2db = log(plvl)
            qvapor = max(QME6, qvapor)
            tvly = tgrs * (1.0 + constants.ZVIR * qvapor)
        with interval(1, -1):
            qvapor = max(QME6, qvapor)
            tvly = tgrs * (1.0 + constants.ZVIR * qvapor)
            tlvl = tgrs[0, 0, -1] + (tgrs - tgrs[0, 0, -1]) * (
                log(plvl) - log(plyr[0, 0, -1])
            ) / (log(plyr) - log(plyr[0, 0, -1]))
        with interval(-1, None):
            tlvl = tgrs[0, 0, -1]


def calc_tlvl_am5(
    plyr: FloatField,
    plvl: FloatField,
    tlyr: FloatField,
    tskin: FloatFieldIJ,
    tlvl: FloatField,
):
    """
    Calculates interface(level) temperatures needed for radiation as in the am5 physics
    Assumes k=0 at the top of the atmosphere
    """
    with computation(FORWARD):
        with interval(0, 1):
            tlvl = tlyr
        with interval(1, -1):
            tlvl = (
                (plyr[0, 0, -1] * tlyr[0, 0, -1] * (plvl - plyr))
                + (plyr * tlyr * (plyr[0, 0, -1] - plvl))
            ) / (plvl * (plyr[0, 0, -1] - plyr))
        with interval(-1, None):
            tlvl = tskin


def calc_net_flux_and_heating(
    sw_flux_up: FloatField,
    sw_flux_down: FloatField,
    lw_flux_up: FloatField,
    lw_flux_down: FloatField,
    sw_flux_up_clear: FloatField,
    sw_flux_down_clear: FloatField,
    lw_flux_up_clear: FloatField,
    lw_flux_down_clear: FloatField,
    sw_flux_net: FloatFieldIJ,
    p_lev: FloatField,
    sw_heating_rate: FloatField,
    lw_heating_rate: FloatField,
    sw_heating_rate_clear: FloatField,
    lw_heating_rate_clear: FloatField,
):
    """
    Calculates heating rates and net shortwave surface flux
    """
    with computation(FORWARD), interval(0, 1):
        sw_flux_net = sw_flux_down - sw_flux_up
        sw_heating_rate = calc_heating_rate(sw_flux_up, sw_flux_down, p_lev)
        lw_heating_rate = calc_heating_rate(lw_flux_up, lw_flux_down, p_lev)
        sw_heating_rate_clear = calc_heating_rate(
            sw_flux_up_clear, sw_flux_down_clear, p_lev
        )
        lw_heating_rate_clear = calc_heating_rate(
            lw_flux_up_clear, lw_flux_down_clear, p_lev
        )
    with computation(PARALLEL), interval(1, -1):
        sw_heating_rate = calc_heating_rate(sw_flux_up, sw_flux_down, p_lev)
        lw_heating_rate = calc_heating_rate(lw_flux_up, lw_flux_down, p_lev)
        sw_heating_rate_clear = calc_heating_rate(
            sw_flux_up_clear, sw_flux_down_clear, p_lev
        )
        lw_heating_rate_clear = calc_heating_rate(
            lw_flux_up_clear, lw_flux_down_clear, p_lev
        )


class RTE_RRTMGPDriver:
    def __init__(
        self,
        config: RTE_RRTMGPConfig,
        gridlon: FloatFieldIJ,
        gridlat: FloatFieldIJ,
        sigma: np.ndarray,
        quantity_factory: QuantityFactory,
        stencil_factory: StencilFactory,
    ):
        grid_indexing = stencil_factory.grid_indexing
        iyear = config.date.year
        imonth = config.date.month
        iday = config.date.day
        ihr = config.date.hour
        self.saved_iyear = iyear
        self.saved_imonth = imonth
        self.saved_iday = iday
        self.deltsw = config.deltsw
        self.delt_rad = config.delt_rad
        self.isolar = config.isolar
        self.ico2flg = config.ico2flg
        self.ictmflg = config.ictmflg
        self.ialbflg = config.ialbflg
        self.ldisable_radiation_quasi_sea_ice = config.ldisable_radiation_quasi_sea_ice
        self._first_step = True

        self.solhr = ihr + config.date.minute / 60.0 + config.date.second / 3600.0
        self.slag = 0.0
        self.sdec = 0.0
        self.cdec = 0.0
        self.anginc = 0.0
        self.solcon = 0.0
        self.solc0 = 0.0
        self.nstp = 0

        # Allocate quantities
        self._coszdg = quantity_factory.zeros(
            [I_DIM, J_DIM],
            "radians",
            dtype=Float,
        )
        self._daymask = quantity_factory.zeros(
            [I_DIM, J_DIM],
            "",
            dtype=Bool,
        )
        self._co2_cyc = quantity_factory.zeros(
            [I_DIM, J_DIM],
            "",
            dtype=Bool,
        )
        self._co2_arr = quantity_factory.zeros(
            [I_DIM, J_DIM],
            "",
            dtype=Bool,
        )
        self._tvly = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            "degK",
            dtype=Float,
        )
        self._tsfca = quantity_factory.zeros(
            [I_DIM, J_DIM],
            "degK",
            dtype=Float,
        )
        self._cnvw = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            "",
            dtype=Float,
        )
        self._cnvc = quantity_factory.zeros(
            [I_DIM, J_DIM, K_DIM],
            "",
            dtype=Float,
        )
        self._coslat = quantity_factory.zeros(
            [I_DIM, J_DIM],
            "",
            dtype=Float,
        )

        self.gridlon = gridlon
        self.gridlat = gridlat
        config.input_dir.joinpath(config.solar_constant_file)
        # Init solar params
        self.isolflg, self._solar_constants, self.solc0 = sol_init(
            self.isolar,
            config.input_dir.joinpath(config.solar_constant_file),
            iyear,
        )
        # Here is where we will initialize aerosols once they're supported

        # Init gases
        (
            self.n2o,
            self.ch4,
            self.o2,
            self.co,
            self.n2,
            self.cfc11,
            self.cfc12,
            self.cfc22,
            self.cfc113,
            self.ccl4,
            self.co2_glb,
            co2_arr,
            co2_cyc,
            self.co2_mvr_data,
            self.co2_glb_data,
            self.co2_cyc_data,
        ) = gas_init(
            config.input_dir,
            config.ico2flg,
            config.ioznflg,
            config.ictmflg,
            iyear,
            imonth,
            gridlon.view[:],
            gridlat.view[:],
        )

        self._co2_cyc.view[:] = co2_cyc
        self._co2_arr.view[:] = co2_arr

        # Init sfc albedo and emissivity
        self.albedo = np.zeros((gridlon.view[:].shape[0], gridlon.view[:].shape[1], 4))
        self.sfcemis = np.zeros((gridlon.view[:].shape[0], gridlon.view[:].shape[1]))
        sfcemis_datafile = config.input_dir.joinpath("sfc_emissivity_idx.txt")
        self.iemslw, self._sfcemis_map = sfc_init(
            config.ialbflg,
            config.iemsflg,
            config.ldisable_radiation_quasi_sea_ice,
            sfcemis_datafile,
        )

        # Init clouds:
        self._llyr = cld_init(sigma, config.ivflip)

        self._cloud_optics_lw = CloudOptics(cloud_optics_file=CloudOpticsFiles.LW_BND)
        self._cloud_optics_sw = CloudOptics(cloud_optics_file=CloudOpticsFiles.SW_BND)

        # Gas optics: GT4Py gather core (drop-in for pyRTE GasOptics). The class
        # keeps pyRTE's interpolate + RTE solve and replaces only the per-g-point
        # table-gather core with the validated GT4Py stencils. It folds the
        # flattened radx "column" axis into an (nx, ny) compute tile, so it needs
        # the local compute dims and the framework backend. radx is the compute
        # domain with halos/padding removed (state.to_rterrtmgp_xr), so its column
        # count is nic*njc = domain_compute()[:2] and its layer count is npz.
        nx, ny, nz = grid_indexing.domain_compute()
        gas_optics_backend = quantity_factory.backend
        self._gas_optics_lw = GasOpticsGT4Py(
            gas_optics_file=GasOpticsFiles.LW_G256,
            nx=nx,
            ny=ny,
            nz=nz,
            backend=gas_optics_backend,
        )
        self._gas_optics_sw = GasOpticsGT4Py(
            gas_optics_file=GasOpticsFiles.SW_G224,
            nx=nx,
            ny=ny,
            nz=nz,
            backend=gas_optics_backend,
        )

        # Clear-sky RTE solvers: validated GT4Py ports (drop-in for the
        # clear-sky half of pyRTE's rte.solve). Built once here, mirroring the
        # GasOpticsGT4Py construction above (same nx, ny, nz and backend).
        # top_at_1 is a property of the gas-optics output's vertical ordering
        # and is only known at compute time; this driver's inputs are bottom-up
        # (calc_tlvl_gfs places the surface at K=0), so the compiled assumption
        # is top_at_1 = False. step_radiation asserts the runtime optics match
        # before using these solvers.
        self._nx, self._ny, self._nz = nx, ny, nz
        self._solver_top_at_1 = False
        self._lw_solver = LWNoScatSolverGT4Py(
            nx=nx,
            ny=ny,
            nz=nz,
            ngpt=NGPT_LW,
            backend=gas_optics_backend,
            top_at_1=self._solver_top_at_1,
        )
        self._sw_solver = SWTwoStreamSolverGT4Py(
            nx=nx,
            ny=ny,
            nz=nz,
            ngpt=NGPT_SW,
            backend=gas_optics_backend,
            top_at_1=self._solver_top_at_1,
        )
        self._gas_mapping = {
            "h2o": "qvapor",
            "o3": "qo3mr",
            "co": "co",
            "n2o": "n2o",
            "o2": "o2",
            "co2": "co2",
            "n2": "n2",
        }
        self._var_mapping = {
            "pres_layer": "prsl",
            "pres_level": "prsi",
            "temp_layer": "tlyr",
            "temp_level": "tlvl",
            "surface_temperature": "tsfc",
            "lwp": "clwp",
            "iwp": "cip",
            "rel": "clwr",
            "rei": "cir",
            "solar_zenith_angle": "solar_zenith_angle",
            "surface_albedo": "surface_albedo",
            "surface_albedo_direct": "surface_albedo_direct",
            "surface_albedo_diffuse": "surface_albedo_diffuse",
            "surface_emissivity": "surface_emissivity",
            "surface_emissivity_jacobian": "surface_emissivity_jacobian",
        }
        self._atm_map = AtmosphericMapping(
            dim_mapping=DEFAULT_DIM_MAPPING,
            var_mapping=self._var_mapping,
        )

        self._calc_tlvl = stencil_factory.from_origin_domain(
            func=calc_tlvl_gfs,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        if config.icmphys == 4:
            self._cldscheme = 4
            self._progcld4 = stencil_factory.from_origin_domain(
                func=progcld4,
                externals={
                    "ivflip": config.ivflip,
                    "lcrick": config.lcrick,
                    "lcnorm": config.lcnorm,
                },
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
        elif config.icmphys == 5:
            self._cldscheme = 5
            self._progcld5 = stencil_factory.from_origin_domain(
                func=progcld5,
                externals={
                    "gfs_cloud_overlap": config.gfs_cloud_overlap,
                    "ivflip": config.ivflip,
                    "lcrick": config.lcrick,
                    "lcnorm": config.lcnorm,
                },
                origin=grid_indexing.origin_compute(),
                domain=grid_indexing.domain_compute(),
            )
            raise NotImplementedError(
                f"radiation cloud microphysics control flag {config.icmphys} "
                "does not have cnvw or cnvc yet"
            )
        else:
            raise NotImplementedError(
                f"radiation cloud microphysics control flag {config.icmphys} "
                "not implemented, please choose 4 or 5"
            )
        self._coszmn = stencil_factory.from_origin_domain(
            func=coszmn,
            externals={
                "daily_mean": config.daily_mean,
                "fixed_sollat": config.fixed_sollat,
                "nstp": config.nstp,
                "sollat": config.sollat,
            },
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )
        if config.ictmflg == -2:
            if config.ivflip == 0:
                self._get_gases = stencil_factory.from_origin_domain(
                    func=get_gases_topdown,
                    externals={
                        "ico2flg": config.ico2flg,
                    },
                    origin=grid_indexing.origin_compute(),
                    domain=grid_indexing.domain_compute(),
                )
            else:
                self._get_gases = stencil_factory.from_origin_domain(
                    func=get_gases_bottomup,
                    externals={
                        "ico2flg": config.ico2flg,
                    },
                    origin=grid_indexing.origin_compute(),
                    domain=grid_indexing.domain_compute(),
                )

        self._calc_net_flux_and_heating = stencil_factory.from_origin_domain(
            func=calc_net_flux_and_heating,
            origin=grid_indexing.origin_compute(),
            domain=grid_indexing.domain_compute(),
        )

    def _accumulate_radiation_inputs(
        self, state: RTE_RRTMGPState, sfc_state: SurfaceState, sdate: datetime.datetime
    ):
        """
        For RTE-RRTMGP we need level and layer profiles of temperature and pressure,
        the species used for the spectral calculations:
            humidity, cloud water (and size), cloud ice (and size),
            CO2, O3, N2O, N2, O2, CH4, CO
        albedo and surface emissivities, the solar zenith angle
        Here we extract that info from the model state and time,
        and make sure units are correct.
        """

        self._update_inputs_if_needed(state, sdate)

        self._calc_tlvl(
            state.prsl,
            state.prsi,
            state.tlyr,
            state.tsfc,
            state.qvapor,
            self._tvly,
            self._tsfca,
            state.tlvl,
        )

        self._coszmn(
            self.gridlon,
            self.gridlat,
            self._coslat,
            Float(sdate.hour),
            self.slag,
            self.sdec,
            self.cdec,
            self.anginc,
            state.mu0,
            self._coszdg,
            self._daymask,
        )

        if self.ictmflg == -2:
            self._get_gases(
                state.co2,
                state.prsl,
                self.co2_glb,
                self._co2_cyc,
                self._co2_arr,
            )

        if self._cldscheme == 4:
            self._progcld4(
                state.prsl,
                state.prsi,
                state.tlyr,
                self._tvly,
                state.qliquid,
                sfc_state.islmsk,
                state.qcld,
                state.clwp,
                state.clwr,
                state.cip,
                state.cir,
            )
        elif self._cldscheme == 5:
            self._progcld5(
                state.prsl,
                state.prsi,
                state.tlyr,
                self._tvly,
                state.qliquid,
                self._cnvw,
                self._cnvc,
                sfc_state.islmsk,
                state.qcld,
                state.clwp,
                state.clwr,
                state.cip,
                state.cir,
            )

        set_albedo(
            self.ialbflg,
            sfc_state.islmsk.field,
            sfc_state.snowd.field,
            sfc_state.sncovr.field,
            sfc_state.snoalb.field,
            sfc_state.zorl.field,
            state.mu0.field,
            sfc_state.tsfc.field,
            sfc_state.hprim.field,
            sfc_state.alvsf.field,
            sfc_state.alnsf.field,
            sfc_state.alvwf.field,
            sfc_state.alnwf.field,
            sfc_state.facsf.field,
            sfc_state.facwf.field,
            sfc_state.fice.field,
            sfc_state.tisfc.field,
            sfc_state.albedo.field,
            self.albedo,
            self.ldisable_radiation_quasi_sea_ice,
        )
        if self.ialbflg == -1 or self.ialbflg == -2:
            state.albedo.view[:] = self.albedo[:, :, 0]
            # TODO: Add support for diffuse and direct albedos,
            # other values for ialbflg abnf the rest of the sfc parameterization code
        set_sfcemis(
            self.gridlon.view[:],
            self.gridlat.view[:],
            sfc_state.islmsk.field,
            sfc_state.snowd.field,
            sfc_state.sncovr.field,
            sfc_state.zorl.field,
            sfc_state.tsfc.field,
            sfc_state.hprim.field,
            self.iemslw,
            self.ialbflg,
            self.ldisable_radiation_quasi_sea_ice,
            self.sfcemis,
            self._sfcemis_map,
            sfc_state.sfcemis.field,
        )
        state.sfc_emis.view[:] = self.sfcemis

    def _update_inputs_if_needed(
        self, state: RTE_RRTMGPState, sdate: datetime.datetime
    ):
        """
        Updates input data from external sources when model date differs
        from the saved date
        """
        lsol_chg = False
        if (self.isolflg not in [0, 10]) and sdate.year != self.saved_iyear:
            lsol_chg = True
        (
            self.slag,
            self.sdec,
            self.cdec,
            self.anginc,
            self.solcon,
            self.solc0,
            self.nstp,
            self.saved_iyear,
        ) = solar_update(
            sdate,
            self.solc0,
            self.deltsw,
            self.delt_rad,
            lsol_chg,
            self.saved_iyear,
            self.isolflg,
            self._solar_constants,
        )

        # Here is where we update ozone and aerosols when enabled
        update_co2 = False
        if (sdate.month != self.saved_imonth) or (self._first_step):
            update_co2 = True
            self.saved_imonth = sdate.month

        (
            self.co2_glb,
            self._co2_arr.view[:],
            self._co2_cyc.view[:],
        ) = co2_update(
            sdate.year,
            sdate.month,
            self.ico2flg,
            update_co2,
            self.ictmflg,
            self.co2_glb,
            self._co2_arr.view[:],
            self._co2_cyc.view[:],
            self.gridlon.view[:],
            self.gridlat.view[:],
            self.co2_glb_data,
            self.co2_mvr_data,
            self.co2_cyc_data,
        )

    def _assign_constant_gases(self, xds):
        xds["ch4"] = Float(self.ch4)
        xds["n2o"] = Float(self.n2o)
        xds["n2"] = Float(self.n2)
        xds["o2"] = Float(self.o2)
        xds["co"] = Float(self.co)
        xds["ch4"] = Float(self.ch4)
        xds["cfc11"] = Float(
            self.cfc11,
        )
        xds["cfc12"] = Float(
            self.cfc12,
        )
        xds["cfc22"] = Float(
            self.cfc22,
        )
        xds["cfc113"] = Float(
            self.cfc113,
        )
        xds["ccl4"] = Float(
            self.ccl4,
        )

    def prep_radiation(
        self, state: RTE_RRTMGPState, sfc_state: SurfaceState, date: datetime.datetime
    ):
        """
        Method to prepare radiation inputs for flux calculations. Updates solar, gas,
        and surface variables, gathers the necessary atmospheric data, and packages it
        as an xarray dataset that can be ingested by pyRTE-RRTMGP.

        Args:
            state (RTE_RRTMGPState): input state containing atmospheric information
            sfc_state (SurfaceState): contains surface properties such as
                surface type, snow cover, etc.
            date (datetime.datetime): datetime for radiation calculations

        Returns:
            radx: (xarray.Dataset): An xarray dataset ready to be passed into
                pyRTE-RRTMGP
        """
        self._accumulate_radiation_inputs(state, sfc_state, date)
        radx = state.to_rterrtmgp_xr()
        self._assign_constant_gases(radx)
        return radx

    def _solve_sw_clear(self, sw_optics):
        """Clear-sky shortwave broadband fluxes via the GT4Py two-stream solver.

        Extracts tau/ssa/g/mu0/surface_albedo/toa_source from the gas-optics
        object and folds the flattened `column` dim into the (nx, ny) tile
        exactly as tests/rte_solver/test_sw_solver_gt4py.py does, then calls the
        validated SWTwoStreamSolverGT4Py. Returns broadband (up, down, direct),
        each (nx, ny, nlev); `down` is the total (diffuse + direct) flux.
        """
        nx, ny = self._nx, self._ny
        layer_dim = sw_optics.mapping.get_dim("layer")
        level_dim = sw_optics.mapping.get_dim("level")
        assert bool(sw_optics.attrs["top_at_1"]) == self._solver_top_at_1, (
            "SW gas-optics vertical ordering (top_at_1="
            f"{bool(sw_optics.attrs['top_at_1'])}) does not match the compiled "
            f"solver assumption (top_at_1={self._solver_top_at_1})"
        )
        noncore = [
            d for d in sw_optics["tau"].dims if d not in (layer_dim, level_dim, "gpt")
        ]
        ngpt = int(sw_optics.sizes["gpt"])

        tau = _tile(sw_optics["tau"], noncore, [layer_dim, "gpt"], nx, ny)
        ssa = _tile(sw_optics["ssa"], noncore, [layer_dim, "gpt"], nx, ny)
        gg = _tile(sw_optics["g"], noncore, [layer_dim, "gpt"], nx, ny)

        # toa_source follows total_solar_irradiance's dims and may lack some
        # non-core dims; broadcast it over the full non-core set before tiling,
        # the same way the SW test does.
        toa_da = sw_optics["toa_source"]
        for d in noncore:
            if d not in toa_da.dims:
                toa_da = toa_da.expand_dims({d: int(sw_optics.sizes[d])})
        inc_dir = _tile(toa_da, noncore, ["gpt"], nx, ny)

        mu0 = _col(sw_optics["mu0"], noncore, nx, ny)
        alb = _col(sw_optics["surface_albedo"], noncore, nx, ny)
        alb_gpt = np.repeat(alb[:, :, None], ngpt, axis=2)

        return self._sw_solver.solve(
            tau=tau,
            ssa=ssa,
            g=gg,
            mu0=mu0,
            sfc_alb_dir=alb_gpt,
            sfc_alb_dif=alb_gpt,
            inc_flux_dir=inc_dir,
        )

    def _solve_lw_clear(self, lw_optics):
        """Clear-sky longwave broadband fluxes via the GT4Py no-scattering solver.

        Extracts tau and the Planck sources (layer/level/surface) plus the
        surface emissivity from the gas-optics object and folds the flattened
        `column` dim into the (nx, ny) tile exactly as
        tests/rte_solver/test_lw_solver_gt4py.py does, then calls the validated
        LWNoScatSolverGT4Py (incident flux = 0, clear-sky). Returns broadband
        (up, down), each (nx, ny, nlev).
        """
        nx, ny = self._nx, self._ny
        layer_dim = lw_optics.mapping.get_dim("layer")
        level_dim = lw_optics.mapping.get_dim("level")
        assert bool(lw_optics.attrs["top_at_1"]) == self._solver_top_at_1, (
            "LW gas-optics vertical ordering (top_at_1="
            f"{bool(lw_optics.attrs['top_at_1'])}) does not match the compiled "
            f"solver assumption (top_at_1={self._solver_top_at_1})"
        )
        noncore = [
            d for d in lw_optics["tau"].dims if d not in (layer_dim, level_dim, "gpt")
        ]
        ngpt = int(lw_optics.sizes["gpt"])

        tau = _tile(lw_optics["tau"], noncore, [layer_dim, "gpt"], nx, ny)
        lay = _tile(lw_optics["layer_source"], noncore, [layer_dim, "gpt"], nx, ny)
        lev = _tile(lw_optics["level_source"], noncore, [level_dim, "gpt"], nx, ny)
        sfc = _tile(lw_optics["surface_source"], noncore, ["gpt"], nx, ny)

        # Surface emissivity here is a single per-column value (radx["sfc_emis"],
        # no gpt dim); broadcast it across g-points. (If a future input already
        # carries a gpt axis, tile it directly.)
        emis_da = lw_optics["surface_emissivity"]
        if "gpt" in emis_da.dims:
            emis = _tile(emis_da, noncore, ["gpt"], nx, ny)
        else:
            emis_col = _col(emis_da, noncore, nx, ny)
            emis = np.repeat(emis_col[:, :, None], ngpt, axis=2)

        return self._lw_solver.solve(
            tau=tau,
            lay_source=lay,
            lev_source=lev,
            sfc_src=sfc,
            sfc_emis=emis,
        )

    def step_radiation(
        self, state: RTE_RRTMGPState, sfc_state: SurfaceState, date: datetime.datetime
    ):
        """
        Method to compute radiative fluxes for a given atmospheric and surface state
        at a given date and time. The Radiation State is updated with longwave and
        shortwave fluxes and heating rates in-place.

        Args:
            state (RTE_RRTMGPState): input state containing atmospheric information,
                will be updated with radiative fluxes and heating rates
            sfc_state (SurfaceState): contains surface properties such as
                surface type, snow cover, etc.
            date (datetime.datetime): datetime for radiation calculations
        """
        self.solhr = date.hour + date.minute / 60.0 + date.second / 3600.0
        radx = self.prep_radiation(state, sfc_state, date)

        # Do SW fluxes:
        sw_optics = self._gas_optics_sw.compute(
            radx,
            problem_type=rte.OpticsTypes.TWO_STREAM,
            add_to_input=False,
            gas_name_map=self._gas_mapping,
            variable_mapping=self._atm_map,
        )
        sw_optics["surface_albedo"] = radx["albedo"]
        sw_optics["mu0"] = radx["mu0"]
        # Clear-sky SW solve: validated GT4Py two-stream solver (replaces the
        # pyRTE sw_optics.rte.solve for the clear-sky path only). Input
        # extraction mirrors tests/rte_solver/test_sw_solver_gt4py.py.
        bb_up_sw, bb_dn_sw, _bb_dir_sw = self._solve_sw_clear(sw_optics)
        state.fswd_clr.view[:] = bb_dn_sw.reshape(state.fswd_clr.view[:].shape)
        state.fswu_clr.view[:] = bb_up_sw.reshape(state.fswu_clr.view[:].shape)

        sw_cloud_optical_props = self._cloud_optics_sw.compute(
            radx,
            problem_type=rte.OpticsTypes.TWO_STREAM,
            add_to_input=False,
            variable_mapping=self._atm_map,
        )
        sw_cloud_optical_props.rte.add_to(sw_optics)
        # TODO: all-sky SW solve stays on pyRTE until cloud optics is ported to
        # GT4Py (the clear-sky GT4Py solver above does not consume cloud optics).
        fluxes_sw = sw_optics.rte.solve(add_to_input=False)
        state.fswd.view[:] = fluxes_sw.sw_flux_down[:].reshape(state.fswd.view[:].shape)
        state.fswu.view[:] = fluxes_sw.sw_flux_up[:].reshape(state.fswu.view[:].shape)

        # And do LW fluxes
        lw_optics = self._gas_optics_lw.compute(
            radx,
            problem_type=rte.OpticsTypes.ABSORPTION,
            add_to_input=False,
            gas_name_map=self._gas_mapping,
            variable_mapping=self._atm_map,
        )
        lw_optics["surface_emissivity"] = radx["sfc_emis"]
        # Clear-sky LW solve: validated GT4Py no-scattering solver (replaces the
        # pyRTE lw_optics.rte.solve for the clear-sky path only). Input
        # extraction mirrors tests/rte_solver/test_lw_solver_gt4py.py.
        bb_up_lw, bb_dn_lw = self._solve_lw_clear(lw_optics)
        state.flwd_clr.view[:] = bb_dn_lw.reshape(state.flwd_clr.view[:].shape)
        state.flwu_clr.view[:] = bb_up_lw.reshape(state.flwu_clr.view[:].shape)
        lw_cloud_optical_props = self._cloud_optics_lw.compute(
            radx,
            problem_type=rte.OpticsTypes.ABSORPTION,
            add_to_input=False,
            variable_mapping=self._atm_map,
        )
        lw_cloud_optical_props.rte.add_to(lw_optics)
        # TODO: all-sky LW solve stays on pyRTE until cloud optics is ported to
        # GT4Py; the all-sky LW path may also need the rescaling that the
        # clear-sky GT4Py no-scattering solver does not implement.
        fluxes_lw = lw_optics.rte.solve(add_to_input=False)

        state.flwd.view[:] = fluxes_lw.lw_flux_down[:].reshape(state.flwd.view[:].shape)
        state.flwu.view[:] = fluxes_lw.lw_flux_up[:].reshape(state.flwu.view[:].shape)

        self._calc_net_flux_and_heating(
            state.fswu,
            state.fswd,
            state.flwu,
            state.flwd,
            state.fswu_clr,
            state.fswd_clr,
            state.flwu_clr,
            state.flwd_clr,
            state.fswn,
            state.prsi,
            state.hrtsw,
            state.hrtlw,
            state.hrtsw_clr,
            state.hrtlw_clr,
        )

        if self._first_step:
            self._first_step = False

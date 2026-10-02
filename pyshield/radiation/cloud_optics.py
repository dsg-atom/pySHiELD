"""GT4Py stencils for the RRTMGP cloud-optics LUT lookup and the by-band ->
by-g-point optical-props increment.

These reimplement, as NDSL/GT4Py stencils, two Fortran pieces that pyRTE runs on
the CPU for the all-sky path:

  1. `compute_all_from_table` in `rrtmgp-frontend/mo_cloud_optics_rrtmgp.F90`
     (lines 633-674): the per-band cloud optical properties as a LINEAR
     interpolation, in effective radius, of the look-up tables loaded from the
     SW_BND / LW_BND coefficient files. (pyRTE's `CloudOptics.compute`
     transcribes this LUT path; the Pade path in the same Fortran module is not
     used -- the distributed `_BND` files and pyRTE both take the LUT branch.)

         index = min(floor((re - offset)/step_size) + 1, nsteps - 1)   [1-based]
         fint  = (re - offset)/step_size - (index - 1)
         t     = lwp * (ext(index) + fint*(ext(index+1) - ext(index)))
         ts    = t   * (ssa(index) + fint*(ssa(index+1) - ssa(index)))
         tssg  = ts  * (asy(index) + fint*(asy(index+1) - asy(index)))

     Done once for liquid (clwp, rel) and once for ice (ciwp, rei); the host
     combines them (see `cloud_optics_gt4py.py`). Where there is no cloud the
     water path is zero so `t = lwp*ext = 0` reproduces the Fortran mask-false
     branch exactly (0.0). The 1-D table gather is the cloud-optics analogue of
     the `planck_interp1d` gather in `gas_optics.py`; the integer size index and
     the fraction are prepared on the host and the band is driven by a host loop
     around this compile-once stencil, writing one band slice per call.

  2. the by-band increment kernels in `rte-kernels/mo_optical_props_kernels.F90`
     (`inc_1scalar_by_1scalar_bybnd`, `inc_2stream_by_2stream_bybnd`): the
     `rte.add_to` step that folds the by-band cloud optics into the by-g-point
     gas optics. Each g-point reads the cloud value of ITS band (the host passes
     the right band slice per g-point), so the band->g-point expansion is a host
     loop around these compile-once pointwise stencils.

All math is kept in float64 (the Fortran working precision).
"""

from ndsl.dsl.gt4py import PARALLEL, computation, interval
from ndsl.dsl.typing import Float, FloatField, IntField
from ndsl.dsl.gt4py import GlobalTable

# Cloud LUT alias shape. The real tables are (nsize_liq|nsize_ice, nband); the
# standard RRTMGP _BND files have nsize_liq=58, nsize_ice=46, nband=14 (SW) / 16
# (LW). One padded alias serves liquid, ice, SW and LW: the size axis is clamped
# to the real nsize-2 on the host and the band loop only visits the real nband,
# so the padded rows/columns are never read. The caller asserts the real tables
# fit.
NSIZE_CLD = 64
NBND_CLD = 16
CldTable = GlobalTable[(Float, (NSIZE_CLD, NBND_CLD))]


def cloud_lut_band(
    lwp: FloatField,
    fint: FloatField,
    idx: IntField,
    iband: IntField,
    ext: CldTable,
    ssa: CldTable,
    asy: CldTable,
    tau: FloatField,
    taussa: FloatField,
    taussag: FloatField,
):
    """One band of `compute_all_from_table` for one phase (liquid or ice).

    `lwp` is the condensed-water path (g/m2), `idx` the 0-based lower size index
    (Fortran `index - 1`, host-clamped to [0, nsteps-2]), `fint` the
    interpolation fraction, `iband` the 0-based band (constant per call). `ext`,
    `ssa`, `asy` are the extinction / single-scattering-albedo / asymmetry LUTs
    for this phase. Outputs are tau, tau*ssa, tau*ssa*g for this band.
    """
    with computation(PARALLEL), interval(...):
        idxp = idx + 1
        et = ext.A[idx, iband] + fint * (ext.A[idxp, iband] - ext.A[idx, iband])
        st = ssa.A[idx, iband] + fint * (ssa.A[idxp, iband] - ssa.A[idx, iband])
        at = asy.A[idx, iband] + fint * (asy.A[idxp, iband] - asy.A[idx, iband])
        t = lwp * et
        ts = t * st
        tau = t
        taussa = ts
        taussag = ts * at


def inc_1scalar_by_1scalar(tau1: FloatField, tau2: FloatField):
    """`inc_1scalar_by_1scalar_bybnd`: add a band's cloud absorption optical
    depth to one g-point's gas absorption optical depth. `tau2` is the cloud
    value of this g-point's band (host-selected), broadcast across the column.
    """
    with computation(PARALLEL), interval(...):
        tau1 = tau1 + tau2


def inc_2stream_by_2stream(
    tau1: FloatField,
    ssa1: FloatField,
    g1: FloatField,
    tau2: FloatField,
    ssa2: FloatField,
    g2: FloatField,
):
    """`inc_2stream_by_2stream_bybnd`: combine one g-point's gas two-stream
    optics (tau1, ssa1, g1) with its band's cloud two-stream optics (tau2, ssa2,
    g2, host-selected). `eps` is an external (= 3*tiny(float64)); the temporaries
    tau12 / tauscat12 are formed before any field is overwritten so g1, ssa1,
    tau1 all read their pre-increment values, exactly as the Fortran does.
    """
    from __externals__ import eps

    with computation(PARALLEL), interval(...):
        tau12 = tau1 + tau2
        tauscat12 = tau1 * ssa1 + tau2 * ssa2
        g1 = (tau1 * ssa1 * g1 + tau2 * ssa2 * g2) / max(eps, tauscat12)
        ssa1 = tauscat12 / max(eps, tau12)
        tau1 = tau12

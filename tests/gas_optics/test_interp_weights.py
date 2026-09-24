"""Validate the Stage-4 gas-optics stencil (fmajor/fminor weights).

`interp_weights` is the tail of the Fortran `interpolation` routine (kernels
lines 158-166): it turns the temperature fraction `ftemp` and pressure
fraction `fpress` (Stage 1) and the two eta fractions `feta1`, `feta2`
(Stage 2, the two temperature brackets) into the eight major-species weights
`fmajor(eta-level, press-level, temp-level)` that the Stage-3 kmajor
interpolation consumes, and the four minor-species weights
`fminor(eta-level, temp-level)` that the later minor-gas 2-D interpolation
consumes. Pure float math, no table gather.

The output is checked against a numpy transcription of the same expressions,
plus the partition-of-unity property: the eight fmajor sum to 1 per cell and
the four fminor sum to 1 per cell.

Run on Discover inside the fork-a venv:
    pytest tests/gas_optics/test_interp_weights.py -q
"""

import numpy as np

from ndsl.boilerplate import get_factories_single_tile
from ndsl.config import backend_python
from ndsl.constants import I_DIM, J_DIM, K_DIM
from ndsl.dsl.typing import Float

from pyshield.radiation.gas_optics import interp_weights


def test_interp_weights():
    nx, ny, nz, nhalo = 4, 4, 8, 3
    rng = np.random.default_rng(3)

    # fractions live in [0, 1); include the exact endpoints to hit the corners
    ftemp = rng.uniform(0.0, 1.0, (nx, ny, nz))
    fpress = rng.uniform(0.0, 1.0, (nx, ny, nz))
    feta1 = rng.uniform(0.0, 1.0, (nx, ny, nz))
    feta2 = rng.uniform(0.0, 1.0, (nx, ny, nz))
    ftemp[0, 0, 0], fpress[0, 0, 0] = 0.0, 0.0
    feta1[0, 0, 0], feta2[0, 0, 0] = 0.0, 1.0

    # numpy oracle -----------------------------------------------------------
    ft1 = 1.0 - ftemp
    fp0 = 1.0 - fpress
    m11 = (1.0 - feta1) * ft1
    m21 = feta1 * ft1
    m12 = (1.0 - feta2) * ftemp
    m22 = feta2 * ftemp
    o_fmn = {"fmn11": m11, "fmn21": m21, "fmn12": m12, "fmn22": m22}
    o_fmaj = {
        "f111": fp0 * m11, "f211": fp0 * m21, "f121": fpress * m11, "f221": fpress * m21,
        "f112": fp0 * m12, "f212": fp0 * m22, "f122": fpress * m12, "f222": fpress * m22,
    }

    # stencil ----------------------------------------------------------------
    stencil_factory, quantity_factory = get_factories_single_tile(
        nx=nx, ny=ny, nz=nz, nhalo=nhalo, backend=backend_python
    )
    grid_indexing = stencil_factory.grid_indexing
    stencil = stencil_factory.from_origin_domain(
        func=interp_weights,
        origin=grid_indexing.origin_compute(),
        domain=grid_indexing.domain_compute(),
    )

    def ff():
        return quantity_factory.zeros([I_DIM, J_DIM, K_DIM], "", dtype=Float)

    ftemp_q, fpress_q, feta1_q, feta2_q = ff(), ff(), ff(), ff()
    fmn_q = {name: ff() for name in o_fmn}
    fmaj_q = {name: ff() for name in o_fmaj}

    ftemp_q.view[:] = ftemp
    fpress_q.view[:] = fpress
    feta1_q.view[:] = feta1
    feta2_q.view[:] = feta2

    stencil(
        ftemp=ftemp_q,
        fpress=fpress_q,
        feta1=feta1_q,
        feta2=feta2_q,
        fmn11=fmn_q["fmn11"],
        fmn21=fmn_q["fmn21"],
        fmn12=fmn_q["fmn12"],
        fmn22=fmn_q["fmn22"],
        f111=fmaj_q["f111"],
        f211=fmaj_q["f211"],
        f121=fmaj_q["f121"],
        f221=fmaj_q["f221"],
        f112=fmaj_q["f112"],
        f212=fmaj_q["f212"],
        f122=fmaj_q["f122"],
        f222=fmaj_q["f222"],
    )

    # compare ----------------------------------------------------------------
    for name, o in o_fmn.items():
        np.testing.assert_allclose(fmn_q[name].view[:], o, rtol=0, atol=1e-15, err_msg=name)
    for name, o in o_fmaj.items():
        np.testing.assert_allclose(fmaj_q[name].view[:], o, rtol=0, atol=1e-15, err_msg=name)

    # partition of unity: the eight fmajor and the four fminor each sum to 1
    fmaj_sum = sum(fmaj_q[name].view[:] for name in o_fmaj)
    fmn_sum = sum(fmn_q[name].view[:] for name in o_fmn)
    np.testing.assert_allclose(fmaj_sum, 1.0, rtol=0, atol=1e-13)
    np.testing.assert_allclose(fmn_sum, 1.0, rtol=0, atol=1e-13)

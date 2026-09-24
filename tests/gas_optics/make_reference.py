"""Generate the golden gas-optics reference for the GT4Py port.

The GT4Py gas-optics stencils (interpolation -> tau -> Planck source) replace
pyRTE-RRTMGP's CPU-Fortran `GasOptics.compute` in `rte_rrtmgp.py`. To validate
each stencil we need a fixed target: the exact fields pyRTE produces from a
known atmosphere. This script writes that target.

Input is pyRTE's own rfmip clear-sky example atmosphere (100 sites x 18
experiments x 60 layers x 61 levels) -- it ships with the rrtmgp-data tarball
and needs no GFS restart data. We run `GasOptics.compute` with the same
coefficient files GEOS uses at L91 (LW_G256, SW_G224), then save every field
the compute call adds to the dataset.

For LW (problem_type=ABSORPTION) those fields are the absorption optical depth
`tau` and the Planck sources `layer_source`, `level_source`, `surface_source`,
`surface_source_jacobian`. For SW (problem_type=TWO_STREAM) they are `tau`,
`ssa`, and `g`. We don't hard-code the list: we snapshot the variable names
before the call and save whatever is new, so the reference always matches
whatever the installed pyRTE version emits.

Usage (on Discover, inside the fork-a venv):
    python tests/gas_optics/make_reference.py --band lw --out ref_gas_optics_lw.nc
    python tests/gas_optics/make_reference.py --band sw --out ref_gas_optics_sw.nc

Requires XDG_CACHE_HOME set to a nobackup path (see the ndsl-env re-entry
block) so the coefficient download does not hit the home-directory quota.
"""

import argparse

import numpy as np
import xarray as xr

from pyrte_rrtmgp import rte
from pyrte_rrtmgp.examples import RFMIP_FILES, load_example_file
from pyrte_rrtmgp.rrtmgp import GasOptics
from pyrte_rrtmgp.rrtmgp_data_files import GasOpticsFiles
from pyrte_rrtmgp.tests.test_rfmip_clear_sky import RFMIP_GAS_MAPPING


BANDS = {
    "lw": (GasOpticsFiles.LW_G256, rte.OpticsTypes.ABSORPTION),
    "sw": (GasOpticsFiles.SW_G224, rte.OpticsTypes.TWO_STREAM),
}


def strip_bad_attrs(dataset: xr.Dataset) -> xr.Dataset:
    """Drop attributes netCDF cannot serialize.

    pyRTE attaches metadata such as `top_at_1` whose value is an xarray
    DataArray. netCDF attributes must be plain scalars/strings/arrays, so
    writing fails. We only need the numeric fields for the reference, so we
    drop any attribute that is not a valid netCDF attribute type (and coerce
    numpy booleans to int).
    """
    valid = (str, bytes, int, float, complex, np.ndarray, np.number, list, tuple)

    def clean(attrs: dict) -> dict:
        out = {}
        for key, value in attrs.items():
            if isinstance(value, (bool, np.bool_)):
                out[key] = int(value)
            elif isinstance(value, valid):
                out[key] = value
        return out

    dataset.attrs = clean(dataset.attrs)
    for var in dataset.variables.values():
        var.attrs = clean(var.attrs)
    return dataset


def build_reference(band: str):
    """Run pyRTE gas optics on the rfmip atmosphere and return (dataset, new_vars).

    `dataset` is the rfmip atmosphere with the gas-optics outputs added.
    `new_vars` is the sorted list of variable names the compute call added --
    these are the golden fields the GT4Py stencils must reproduce.
    """
    gas_optics_file, problem_type = BANDS[band]

    atmosphere = load_example_file(RFMIP_FILES.ATMOSPHERE)
    before = set(atmosphere.data_vars)

    gas_optics = GasOptics(gas_optics_file=gas_optics_file)
    gas_optics.compute(
        atmosphere,
        problem_type=problem_type,
        gas_name_map=RFMIP_GAS_MAPPING,
        add_to_input=True,
    )

    new_vars = sorted(set(atmosphere.data_vars) - before)
    return atmosphere, new_vars


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band", choices=sorted(BANDS), required=True)
    parser.add_argument(
        "--out",
        required=True,
        help="path to write the reference netCDF",
    )
    args = parser.parse_args()

    atmosphere, new_vars = build_reference(args.band)

    if not new_vars:
        raise SystemExit(
            "gas optics added no variables -- check the pyRTE compute API"
        )

    reference = strip_bad_attrs(atmosphere[new_vars])

    print(f"band {args.band}: {len(new_vars)} fields")
    for name in new_vars:
        var = reference[name]
        print(
            f"  {name:28s} dims={tuple(var.dims)} shape={var.shape} "
            f"sum={float(np.asarray(var).sum()):.6e}"
        )

    reference.to_netcdf(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

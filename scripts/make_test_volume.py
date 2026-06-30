#!/usr/bin/env python3
"""Write a small muGrid NetCDF file for testing muEye.

Creates a 64^3 grid with a scalar field ``phi`` holding a solid cube centred in
the domain, plus a second frame with the cube shifted, so the frame slider has
something to scrub.

Run inside the workspace venv (which has muGrid with NetCDF support):

    source ../venv/bin/activate     # from muEye/
    python scripts/make_test_volume.py demo.nc
"""

import sys

import numpy as np

import muGrid

if not getattr(muGrid, "has_netcdf", True):
    sys.exit("This muGrid build has no NetCDF support; cannot write a test file.")


def cube(n, center, half=0.25):
    """Solid axis-aligned cube of half-extent `half` centred at `center`."""
    ax = (np.arange(n) + 0.5) / n
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    cheby = np.maximum.reduce(
        [np.abs(X - center[0]), np.abs(Y - center[1]), np.abs(Z - center[2])]
    )
    return (cheby <= half).astype(np.float64)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "demo.nc"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    fc = muGrid.GlobalFieldCollection((n, n, n))
    phi = fc.real_field("phi", [1])

    file = muGrid.FileIONetCDF(path, open_mode="overwrite")
    file.register_field_collection(fc)

    # Two frames: cube in the centre, then shifted along the diagonal.
    frames = [((0.5, 0.5, 0.5), 0.25), ((0.62, 0.40, 0.55), 0.18)]
    for center, half in frames:
        phi.p[...] = cube(n, center, half).reshape((1, n, n, n))
        file.append_frame().write()

    print(f"Wrote {path}: {n}^3 grid, field 'phi' (cube), {len(frames)} frames.")


if __name__ == "__main__":
    main()

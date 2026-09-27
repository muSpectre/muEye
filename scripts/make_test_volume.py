#!/usr/bin/env python3
"""Write a small muGrid NetCDF file for testing muEye.

Creates a 64^3 grid with a scalar field ``phi`` holding a solid cube centred in
the domain, plus a second frame with the cube shifted, so the frame slider has
something to scrub.

Needs the muGrid Python package with NetCDF support (e.g. the workspace venv
``../muGrid/venv`` with the muGrid build tree on PYTHONPATH):

    python scripts/make_test_volume.py demo.nc
    python scripts/make_test_volume.py demo.nc 32

Options (position-independent):

``--shear S``   also write a ``deformation_gradient`` global attribute (a
                simple shear F_xy = S), which makes muEye render the volume in
                the sheared (non-orthogonal / Bravais) cell C = F * box:

                    python scripts/make_test_volume.py sheared.nc 64 --shear 0.3

``--warp``      also write a 3-component displacement field ``u`` (a bend plus
                a wave, in grid-point units) that muEye can render as deformed
                geometry via the Dataset panel's "Displacement" selector:

                    python scripts/make_test_volume.py --warp warped.nc
"""

import argparse
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Write a small muGrid NetCDF test volume for muEye.")
    parser.add_argument("path", nargs="?", default="demo.nc",
                        help="output file (default: demo.nc)")
    parser.add_argument("n", nargs="?", type=int, default=64,
                        help="grid points per axis (default: 64)")
    parser.add_argument("--shear", type=float, default=0.0, metavar="S",
                        help="write a deformation_gradient attribute with F_xy=S")
    parser.add_argument("--warp", action="store_true",
                        help="also write a 3-component displacement field 'u'")
    args = parser.parse_args(argv)
    if args.n < 2:
        parser.error("n must be at least 2")
    return args


def main():
    args = parse_args()
    path, n, shear, warp = args.path, args.n, args.shear, args.warp

    fc = muGrid.GlobalFieldCollection((n, n, n))
    phi = fc.real_field("phi", [1])
    # A 3-component displacement field muEye can render as deformed geometry: a
    # bend (u_x grows with y) plus a wave in u_y. Values are in grid-point units.
    u = fc.real_field("u", [3]) if warp else None

    file = muGrid.FileIONetCDF(path, open_mode="overwrite")
    file.register_field_collection(fc)

    # Cell geometry (optional): a macroscopic deformation gradient F, written as
    # a flat row-major 3x3 global attribute muEye reads back. This is the exact
    # hook a muGrid-based solver uses to hand its average strain / deformation
    # gradient to muEye — muGrid needs no change, only write_global_attribute.
    # Write it once, up front, before appending frames.
    if shear != 0.0:
        F = np.eye(3)
        F[0, 1] = shear  # simple shear of x by y
        file.write_global_attribute("deformation_gradient", list(F.ravel()))

    if warp:
        ax = (np.arange(n) + 0.5) / n
        X, Y, _ = np.meshgrid(ax, ax, ax, indexing="ij")
        amp = 0.15 * n  # grid-point units
        ux = amp * (Y - 0.5)          # shear/bend: x-shift grows with y
        uy = amp * 0.3 * np.sin(2 * np.pi * X)
        uz = np.zeros_like(ux)
        u.p[...] = np.stack([ux, uy, uz])  # shape (3, n, n, n) = (comp, i, j, k)

    # Two frames: cube in the centre, then shifted along the diagonal.
    frames = [((0.5, 0.5, 0.5), 0.25), ((0.62, 0.40, 0.55), 0.18)]
    for center, half in frames:
        phi.p[...] = cube(n, center, half).reshape((1, n, n, n))
        file.append_frame().write()

    extra = f", deformation_gradient F_xy={shear}" if shear != 0.0 else ""
    extra += ", displacement field 'u'" if warp else ""
    print(f"Wrote {path}: {n}^3 grid, field 'phi' (cube), "
          f"{len(frames)} frames{extra}.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Read a 2D Nek5000 .re2 mesh and reconstruct the target GLL points
for a straight-sided rectangular genbox mesh.

Outputs:
  1) target_gll_points_all.csv
     One row per element-local GLL point (duplicates kept across shared edges)
  2) target_gll_points_unique.csv
     Deduplicated point cloud
  3) target_gll_points_sem_layout.npz
     Arrays x, y, z with SEM layout shape (nelv, lz, ly, lx)

Requirements:
    pip install pymech numpy pandas

Usage:
    python re2_to_gll_points.py box.re2 --lx1 8 --ly1 8

Notes:
- This is intended for 2D straight-sided meshes generated with genbox.
- For curved elements, this script is not sufficient.
- In 2D Nek conventions, we store shape as (nelv, 1, ly, lx).
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from pymech.neksuite import readre2


def gll_nodes(n: int) -> np.ndarray:
    """
    Return the n Gauss-Lobatto-Legendre nodes on [-1, 1].

    n = number of GLL points per element direction.
    """
    if n < 2:
        raise ValueError("Need at least 2 GLL points per direction.")
    if n == 2:
        return np.array([-1.0, 1.0], dtype=np.float64)

    # Interior GLL nodes are the roots of d/dx P_{n-1}(x)
    from numpy.polynomial.legendre import Legendre

    P = Legendre.basis(n - 1)
    interior = np.sort(P.deriv().roots())
    return np.concatenate(([-1.0], interior, [1.0])).astype(np.float64)


def affine_map(r: np.ndarray, a: float, b: float) -> np.ndarray:
    """Map reference coordinate r in [-1,1] to physical interval [a,b]."""
    return 0.5 * (1.0 - r) * a + 0.5 * (1.0 + r) * b


def element_bounds_from_pos(elem) -> tuple[float, float, float, float]:
    """
    Extract axis-aligned bounds from one pymech element.

    For a straight 2D genbox mesh, min/max of pos are enough.
    """
    # elem.pos has coordinate component first: x,y,z
    x_all = np.asarray(elem.pos[0], dtype=np.float64)
    y_all = np.asarray(elem.pos[1], dtype=np.float64)

    x0 = float(np.min(x_all))
    x1 = float(np.max(x_all))
    y0 = float(np.min(y_all))
    y1 = float(np.max(y_all))

    return x0, x1, y0, y1


def reconstruct_sem_arrays_from_re2(re2_file: str, lx1: int, ly1: int):
    """
    Return SEM-layout coordinate arrays:
        x, y, z with shapes (nelv, 1, ly1, lx1)
    """
    data = readre2(re2_file)

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D mesh, but data.ndim = {data.ndim}")

    nelv = data.nel

    # GLL nodes in reference element
    r = gll_nodes(lx1)  # x-like direction
    s = gll_nodes(ly1)  # y-like direction

    x = np.zeros((nelv, 1, ly1, lx1), dtype=np.float64)
    y = np.zeros((nelv, 1, ly1, lx1), dtype=np.float64)
    z = np.zeros((nelv, 1, ly1, lx1), dtype=np.float64)

    for e, elem in enumerate(data.elem):
        x0, x1, y0, y1 = element_bounds_from_pos(elem)

        # Physical 1D GLL coordinates in this element
        xg = affine_map(r, x0, x1)   # shape (lx1,)
        yg = affine_map(s, y0, y1)   # shape (ly1,)

        # Tensor product grid, stored as (nelv, 1, ly, lx)
        X, Y = np.meshgrid(xg, yg, indexing="xy")
        x[e, 0, :, :] = X
        y[e, 0, :, :] = Y
        z[e, 0, :, :] = 0.0

    return x, y, z


def write_outputs(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    nelv, lz, ly, lx = x.shape

    # All element-local points (duplicates kept)
    rows = []
    for e in range(nelv):
        for j in range(ly):
            for i in range(lx):
                rows.append(
                    {
                        "element": e,
                        "k": 0,
                        "j": j,
                        "i": i,
                        "x": x[e, 0, j, i],
                        "y": y[e, 0, j, i],
                        "z": z[e, 0, j, i],
                    }
                )

    df_all = pd.DataFrame(rows)
    df_all.to_csv("target_gll_points_all.csv", index=False)

    # Unique physical points
    df_unique = (
        df_all[["x", "y", "z"]]
        .drop_duplicates()
        .sort_values(["x", "y", "z"], kind="mergesort")
        .reset_index(drop=True)
    )
    df_unique.to_csv("target_gll_points_unique.csv", index=False)

    # Preserve SEM layout for later use with pySEMTools
    np.savez(
        "target_gll_points_sem_layout.npz",
        x=x,
        y=y,
        z=z,
    )

    print("Wrote:")
    print("  target_gll_points_all.csv")
    print("  target_gll_points_unique.csv")
    print("  target_gll_points_sem_layout.npz")
    print()
    print(f"SEM layout shape = {x.shape} = (nelv, lz, ly, lx)")
    print(f"All element-local points: {len(df_all)}")
    print(f"Unique physical points:   {len(df_unique)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("re2_file", help="Path to the .re2 mesh file")
    parser.add_argument("--lx1", type=int, required=True, help="GLL points in x per element")
    parser.add_argument("--ly1", type=int, required=True, help="GLL points in y per element")
    args = parser.parse_args()

    x, y, z = reconstruct_sem_arrays_from_re2(args.re2_file, args.lx1, args.ly1)
    write_outputs(x, y, z)


if __name__ == "__main__":
    main()
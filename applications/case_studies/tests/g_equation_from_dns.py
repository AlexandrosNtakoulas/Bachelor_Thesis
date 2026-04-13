from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from phi.jax.flow import (
    Box,
    CenteredGrid,
    advect,
    channel,
    extrapolation,
    jit_compile,
    math,
    spatial,
)
from skimage import measure

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _reshape_field(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(nx, ny)


def load_snapshot(case_dir: Path, snapshot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points_path = case_dir / "points.hdf5"
    field_path = case_dir / f"structured_fields{snapshot:05d}.hdf5"

    with h5py.File(points_path, "r") as h5f:
        nx = int(h5f.attrs["nx"])
        ny = int(h5f.attrs["ny"])
        x = np.asarray(h5f["x"][:, :, 0], dtype=np.float32)
        y = np.asarray(h5f["y"][:, :, 0], dtype=np.float32)

    with h5py.File(field_path, "r") as h5f:
        u = _reshape_field(h5f["u"][:], nx, ny)
        v = _reshape_field(h5f["v"][:], nx, ny)
        progress_var = _reshape_field(h5f["progress_var"][:], nx, ny)

    return x, y, u, v, progress_var


def load_velocity_snapshot(case_dir: Path, snapshot: int) -> tuple[np.ndarray, np.ndarray]:
    points_path = case_dir / "points.hdf5"
    field_path = case_dir / f"structured_fields{snapshot:05d}.hdf5"

    with h5py.File(points_path, "r") as h5f:
        nx = int(h5f.attrs["nx"])
        ny = int(h5f.attrs["ny"])

    with h5py.File(field_path, "r") as h5f:
        if "u" not in h5f or "v" not in h5f:
            available = ", ".join(sorted(h5f.keys()))
            raise KeyError(
                f"Velocity fields 'u' and 'v' are missing in {field_path.name}. "
                f"Available datasets: [{available}]"
            )
        u = _reshape_field(h5f["u"][:], nx, ny)
        v = _reshape_field(h5f["v"][:], nx, ny)

    return u, v


def available_velocity_snapshots(case_dir: Path) -> list[int]:
    snapshots: list[int] = []
    for path in sorted(case_dir.glob("structured_fields*.hdf5")):
        snap_str = path.stem.removeprefix("structured_fields")
        if not snap_str.isdigit():
            continue
        with h5py.File(path, "r") as h5f:
            if "u" in h5f and "v" in h5f:
                snapshots.append(int(snap_str))
    return snapshots


def select_velocity_snapshots(case_dir: Path, start_snapshot: int, nsteps: int) -> list[int]:
    available = [snap for snap in available_velocity_snapshots(case_dir) if snap >= start_snapshot]
    if len(available) < nsteps:
        available_str = ", ".join(str(s) for s in available[:20]) if available else "none"
        raise ValueError(
            f"Need {nsteps} velocity snapshots at or after {start_snapshot}, but found {len(available)}.\n"
            f"Velocity directory: {case_dir}\n"
            f"Available velocity snapshots: {available_str}"
        )
    return available[:nsteps]


def compute_cell_centered_bounds(x: np.ndarray, y: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be 2D arrays.")
    if x.shape[0] < 2 or y.shape[1] < 2:
        raise ValueError("Need at least 2 cells in x and y to infer grid spacing.")

    dx = float(np.mean(np.diff(x[:, 0])))
    dy = float(np.mean(np.diff(y[0, :])))

    x_bounds = (float(x.min()) - 0.5 * dx, float(x.max()) + 0.5 * dx)
    y_bounds = (float(y.min()) - 0.5 * dy, float(y.max()) + 0.5 * dy)
    return x_bounds, y_bounds


def split_polyline_on_large_jumps(polyline: np.ndarray, *, jump_threshold: float) -> list[np.ndarray]:
    if polyline.shape[0] < 2:
        return []

    step = np.sqrt(np.sum(np.diff(polyline, axis=0) ** 2, axis=1))
    break_idx = np.flatnonzero(step > jump_threshold)

    start = 0
    segments: list[np.ndarray] = []
    for idx in break_idx:
        seg = polyline[start: idx + 1]
        if seg.shape[0] >= 2:
            segments.append(seg)
        start = idx + 1

    seg = polyline[start:]
    if seg.shape[0] >= 2:
        segments.append(seg)

    return segments


def extract_contours_from_progress_var_periodic_y(
    x: np.ndarray,
    y: np.ndarray,
    progress_var: np.ndarray,
    *,
    iso_level: float,
    y_bounds: tuple[float, float],
    min_points: int = 8,
) -> list[np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    progress_var = np.asarray(progress_var, dtype=np.float32)

    nx, ny = progress_var.shape
    y_min, y_max = y_bounds
    y_period = float(y_max - y_min)

    x_coords = x[:, 0].astype(np.float32)
    y_coords = y[0, :].astype(np.float32)

    progress_tiled = np.concatenate([progress_var, progress_var, progress_var], axis=1)
    y_coords_tiled = np.concatenate(
        [y_coords - y_period, y_coords, y_coords + y_period]
    ).astype(np.float32)

    contours_idx = measure.find_contours(
        progress_tiled,
        level=iso_level,
        fully_connected="high",
        positive_orientation="high",
    )

    dx = float(np.mean(np.diff(x_coords)))
    dy = float(np.mean(np.diff(y_coords)))
    jump_threshold = 3.0 * max(abs(dx), abs(dy))

    segments: list[np.ndarray] = []

    for contour_rc in contours_idx:
        if contour_rc.shape[0] < min_points:
            continue

        ix = contour_rc[:, 0]
        iy = contour_rc[:, 1]

        if not np.any((iy >= ny - 0.5) & (iy <= 2 * ny - 0.5)):
            continue

        xc = np.interp(ix, np.arange(nx, dtype=np.float32), x_coords).astype(np.float32)
        yc = np.interp(iy, np.arange(3 * ny, dtype=np.float32), y_coords_tiled).astype(np.float32)

        polyline = np.column_stack([xc, yc]).astype(np.float32)
        pieces = split_polyline_on_large_jumps(polyline, jump_threshold=jump_threshold)
        if not pieces:
            pieces = [polyline]

        for piece in pieces:
            if piece.shape[0] >= 2:
                segments.append(piece)

    if not segments:
        raise ValueError("No valid contour segments were extracted from progress_var.")

    return segments


def min_distance_to_polyline(
    x: np.ndarray,
    y: np.ndarray,
    contour_segments: list[np.ndarray],
    *,
    segment_chunk_size: int = 32,
) -> np.ndarray:
    points = np.column_stack([x.ravel(), y.ravel()]).astype(np.float32, copy=False)
    min_dist2 = np.full(points.shape[0], np.inf, dtype=np.float32)
    eps = np.float32(1.0e-12)

    for contour in contour_segments:
        seg_start = contour[:-1]
        seg_end = contour[1:]

        for start_idx in range(0, seg_start.shape[0], segment_chunk_size):
            end_idx = min(start_idx + segment_chunk_size, seg_start.shape[0])
            a = seg_start[start_idx:end_idx]
            b = seg_end[start_idx:end_idx]

            ab = b - a
            ab2 = np.sum(ab * ab, axis=1)
            ab2 = np.maximum(ab2, eps)

            ap = points[:, None, :] - a[None, :, :]
            t = np.sum(ap * ab[None, :, :], axis=2) / ab2[None, :]
            t = np.clip(t, 0.0, 1.0)

            closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
            delta = points[:, None, :] - closest
            dist2 = np.sum(delta * delta, axis=2)

            min_dist2 = np.minimum(min_dist2, dist2.min(axis=1))

    return np.sqrt(min_dist2).astype(np.float32).reshape(x.shape)


def build_signed_distance_from_progress_var(
    x: np.ndarray,
    y: np.ndarray,
    progress_var: np.ndarray,
    *,
    iso_level: float,
    y_bounds: tuple[float, float],
) -> np.ndarray:
    contour_segments = extract_contours_from_progress_var_periodic_y(
        x,
        y,
        progress_var,
        iso_level=iso_level,
        y_bounds=y_bounds,
    )

    unsigned_distance = min_distance_to_polyline(x, y, contour_segments)
    sign = np.where(progress_var >= iso_level, 1.0, -1.0).astype(np.float32)
    signed_distance = unsigned_distance * sign
    return signed_distance.astype(np.float32)


def make_centered_scalar_grid(values: np.ndarray, x_bounds: tuple[float, float], y_bounds: tuple[float, float]) -> CenteredGrid:
    tensor = math.tensor(values, spatial("x,y"))
    g_bc = extrapolation.combine_sides(
        x=(extrapolation.BOUNDARY, extrapolation.BOUNDARY),
        y=(extrapolation.PERIODIC, extrapolation.PERIODIC),
    )
    return CenteredGrid(
        tensor,
        g_bc,
        bounds=Box["x,y", x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1]],
    )


def make_centered_velocity_grid(
    u: np.ndarray,
    v: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> CenteredGrid:
    tensor = math.stack(
        {
            "x": math.tensor(u, spatial("x,y")),
            "y": math.tensor(v, spatial("x,y")),
        },
        channel("vector"),
    )
    vel_bc = extrapolation.combine_sides(
        x=(extrapolation.BOUNDARY, extrapolation.BOUNDARY),
        y=(extrapolation.PERIODIC, extrapolation.PERIODIC),
    )
    return CenteredGrid(
        tensor,
        vel_bc,
        bounds=Box["x,y", x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1]],
    )


def one_sided_derivatives(phi: jnp.ndarray, dx: float, dy: float) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    phi_xm = jnp.concatenate([phi[:1, :], phi[:-1, :]], axis=0)
    phi_xp = jnp.concatenate([phi[1:, :], phi[-1:, :]], axis=0)

    phi_ym = jnp.concatenate([phi[:, -1:], phi[:, :-1]], axis=1)
    phi_yp = jnp.concatenate([phi[:, 1:], phi[:, :1]], axis=1)

    dmx = (phi - phi_xm) / dx
    dpx = (phi_xp - phi) / dx
    dmy = (phi - phi_ym) / dy
    dpy = (phi_yp - phi) / dy
    return dmx, dpx, dmy, dpy


def godunov_abs_grad(phi: jnp.ndarray, dx: float, dy: float, sd: float) -> jnp.ndarray:
    dmx, dpx, dmy, dpy = one_sided_derivatives(phi, dx, dy)
    a = -sd

    gx_pos = jnp.maximum(jnp.maximum(dmx, 0.0) ** 2, jnp.minimum(dpx, 0.0) ** 2)
    gy_pos = jnp.maximum(jnp.maximum(dmy, 0.0) ** 2, jnp.minimum(dpy, 0.0) ** 2)

    gx_neg = jnp.maximum(jnp.minimum(dmx, 0.0) ** 2, jnp.maximum(dpx, 0.0) ** 2)
    gy_neg = jnp.maximum(jnp.minimum(dmy, 0.0) ** 2, jnp.maximum(dpy, 0.0) ** 2)

    use_pos = a >= 0.0
    gx = jnp.where(use_pos, gx_pos, gx_neg)
    gy = jnp.where(use_pos, gy_pos, gy_neg)

    return jnp.sqrt(gx + gy)


def smoothed_sign(phi0: jnp.ndarray, dx: float, dy: float, eps_factor: float = 1.0) -> jnp.ndarray:
    dx = jnp.asarray(dx, dtype=phi0.dtype)
    dy = jnp.asarray(dy, dtype=phi0.dtype)
    eps = jnp.asarray(eps_factor, dtype=phi0.dtype) * jnp.minimum(dx, dy)
    return phi0 / jnp.sqrt(phi0 * phi0 + eps * eps)

def godunov_abs_grad_reinit(phi: jnp.ndarray, phi0: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Godunov Hamiltonian for reinitialization PDE:
        phi_tau + S(phi0) (|grad phi| - 1) = 0
    """
    dmx, dpx, dmy, dpy = one_sided_derivatives(phi, dx, dy)
    s = smoothed_sign(phi0, dx, dy)

    gx_pos = jnp.maximum(jnp.maximum(dmx, 0.0) ** 2, jnp.minimum(dpx, 0.0) ** 2)
    gy_pos = jnp.maximum(jnp.maximum(dmy, 0.0) ** 2, jnp.minimum(dpy, 0.0) ** 2)

    gx_neg = jnp.maximum(jnp.minimum(dmx, 0.0) ** 2, jnp.maximum(dpx, 0.0) ** 2)
    gy_neg = jnp.maximum(jnp.minimum(dmy, 0.0) ** 2, jnp.maximum(dpy, 0.0) ** 2)

    use_pos = s >= 0.0
    gx = jnp.where(use_pos, gx_pos, gx_neg)
    gy = jnp.where(use_pos, gy_pos, gy_neg)

    return jnp.sqrt(gx + gy)


@jit_compile
def reinitialize_step(
    g_field: CenteredGrid,
    g0_field: CenteredGrid,
    dtau: float,
    dx: float,
    dy: float,
) -> CenteredGrid:
    phi = g_field.values.native(("x", "y"))
    phi0 = g0_field.values.native(("x", "y"))

    s = smoothed_sign(phi0, dx, dy)
    abs_grad = godunov_abs_grad_reinit(phi, phi0, dx, dy)
    phi_new = phi - dtau * s * (abs_grad - 1.0)

    return g_field.with_values(math.tensor(phi_new, spatial("x,y")))


def reinitialize(
    g_field: CenteredGrid,
    dx: float,
    dy: float,
    *,
    n_iters: int = 5,
    dtau_factor: float = 0.3,
) -> CenteredGrid:
    """
    Sussman-type reinitialization.
    Keeps the zero level set approximately fixed while driving |grad G| -> 1.
    """
    dtau = dtau_factor * min(dx, dy)
    g0_field = g_field
    g_re = g_field
    for _ in range(n_iters):
        g_re = reinitialize_step(g_re, g0_field, dtau, dx, dy)
    return g_re


@jit_compile
def g_eq_step(
    g_field: CenteredGrid,
    velocity: CenteredGrid,
    dt: float,
    sd: float,
    dx: float,
    dy: float,
) -> CenteredGrid:
    g_adv = advect.semi_lagrangian(g_field, velocity, dt)
    phi = g_adv.values.native(("x", "y"))
    abs_grad = godunov_abs_grad(phi, dx, dy, sd)
    phi_new = phi + dt * sd * abs_grad
    return g_adv.with_values(math.tensor(phi_new, spatial("x,y")))


def save_plot(
    x: np.ndarray,
    y: np.ndarray,
    g0: np.ndarray,
    g_final: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, data, title in [
        (axes[0], g0, "Initial signed-distance field"),
        (axes[1], g_final, "Final G field"),
    ]:
        mesh = ax.pcolormesh(x, y, data, shading="auto", cmap="coolwarm")
        ax.contour(x, y, data, levels=[0.0], colors="k", linewidths=1.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(mesh, ax=ax)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_animation(
    x: np.ndarray,
    y: np.ndarray,
    frames: list[np.ndarray],
    output_path: Path,
    *,
    fps: int = 20,
) -> None:
    vmin = min(float(np.min(frame)) for frame in frames)
    vmax = max(float(np.max(frame)) for frame in frames)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    mesh = ax.pcolormesh(x, y, frames[0], shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    zero_contour = ax.contour(x, y, frames[0], levels=[0.0], colors="k", linewidths=1.5)

    title = ax.set_title("G-equation evolution: step 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(mesh, ax=ax)

    def update(frame_idx: int):
        nonlocal zero_contour
        mesh.set_array(frames[frame_idx].ravel())
        zero_contour.remove()
        zero_contour = ax.contour(x, y, frames[frame_idx], levels=[0.0], colors="k", linewidths=1.5)
        title.set_text(f"G-equation evolution: step {frame_idx}")
        return [mesh, title, zero_contour]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a signed-distance field from the theta=0.6 progress variable using marching squares "
            "with periodic y handling, evolve the G-equation with constant Sd, and reinitialize after each step."
        )
    )
    parser.add_argument("--snapshot", type=int, default=300)
    parser.add_argument("--phi-tag", default="phi_0.40")
    parser.add_argument("--lat-tag", default="h400x025_ref")
    parser.add_argument("--iso-level", type=float, default=0.6)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--nsteps", type=int, default=2)
    parser.add_argument("--sd", type=float, default=1.0)
    parser.add_argument("--velocity-phi-tag", default=None)
    parser.add_argument("--velocity-lat-tag", default=None)
    parser.add_argument("--fps", type=int, default=1)

    parser.add_argument("--reinit-every", type=int, default=1)
    parser.add_argument("--reinit-iters", type=int, default=5)
    parser.add_argument("--reinit-dtau-factor", type=float, default=0.3)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "fields" / "g_equation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    field_dir = PROJECT_ROOT / "data" / "fields" / "structured_grids" / args.phi_tag / args.lat_tag
    velocity_field_dir = PROJECT_ROOT / "data" / "fields" / "structured_grids" / (
        args.velocity_phi_tag or args.phi_tag
    ) / (args.velocity_lat_tag or args.lat_tag)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, y, _, _, progress_var = load_snapshot(field_dir, args.snapshot)
    x_bounds, y_bounds = compute_cell_centered_bounds(x, y)

    dx = float(np.mean(np.diff(x[:, 0])))
    dy = float(np.mean(np.diff(y[0, :])))

    signed_distance = build_signed_distance_from_progress_var(
        x,
        y,
        progress_var,
        iso_level=args.iso_level,
        y_bounds=y_bounds,
    )

    g_field = make_centered_scalar_grid(signed_distance, x_bounds, y_bounds)

    velocity_snapshots = select_velocity_snapshots(velocity_field_dir, args.snapshot, args.nsteps)
    print(f"Using velocity snapshots: {velocity_snapshots}")

    frames = [signed_distance.copy()]
    for istep, velocity_snapshot in enumerate(velocity_snapshots, start=1):
        try:
            u_step, v_step = load_velocity_snapshot(velocity_field_dir, velocity_snapshot)
        except KeyError as exc:
            available = available_velocity_snapshots(velocity_field_dir)
            available_str = ", ".join(str(s) for s in available[:20]) if available else "none"
            raise KeyError(
                f"{exc}\nRequested velocity snapshot: {velocity_snapshot}\n"
                f"Velocity directory: {velocity_field_dir}\n"
                f"Snapshots containing both u and v: {available_str}"
            ) from exc

        velocity = make_centered_velocity_grid(u_step, v_step, x_bounds, y_bounds)
        g_field = g_eq_step(g_field, velocity, args.dt, args.sd, dx, dy)

        if args.reinit_every > 0 and istep % args.reinit_every == 0:
            g_field = reinitialize(
                g_field,
                dx,
                dy,
                n_iters=args.reinit_iters,
                dtau_factor=args.reinit_dtau_factor,
            )

        frames.append(np.asarray(g_field.values.numpy("x,y"), dtype=np.float32))

    g_final = frames[-1]

    base_name = f"{args.phi_tag}_{args.lat_tag}_snapshot_{args.snapshot:05d}"
    np.save(args.output_dir / f"{base_name}_g_initial.npy", signed_distance)
    np.save(args.output_dir / f"{base_name}_g_final.npy", g_final)

    save_plot(
        x,
        y,
        signed_distance,
        g_final,
        args.output_dir / f"{base_name}_g_evolution.png",
    )
    save_animation(
        x,
        y,
        frames,
        args.output_dir / f"{base_name}_g_evolution.gif",
        fps=args.fps,
    )

    print(f"Saved initial field to {args.output_dir / f'{base_name}_g_initial.npy'}")
    print(f"Saved final field to   {args.output_dir / f'{base_name}_g_final.npy'}")
    print(f"Saved figure to        {args.output_dir / f'{base_name}_g_evolution.png'}")
    print(f"Saved animation to     {args.output_dir / f'{base_name}_g_evolution.gif'}")


if __name__ == "__main__":
    main()
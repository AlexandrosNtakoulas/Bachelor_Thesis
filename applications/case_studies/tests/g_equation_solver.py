import h5py
import numpy as np
from pathlib import Path

from skimage import measure
from phi.jax.flow import (
    Box,
    CenteredGrid,
    advect,
    channel,
    extrapolation,
    spatial,
    Field,
    field
)
from phi.jax.flow import math as phimath
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import jax.numpy as jnp


def _reshape_field(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(nx, ny)



class GSolver():
    def __init__(self):
        self.CASE_NAME = None
        self.case_dir  = None

        self.g_field   = None
        self.vel_field = None
        self.x         = None
        self.y         = None
        self.x_bounds  = None
        self.y_bounds  = None
        self.dx        = None
        self.dy        = None
        self.time      = None

    def initialize(self, CASE_NAME,  initial_snapshot: int, isoline: float):
        """
        Initialize simulation case
        """
        # Set simluation case
        self.CASE_NAME = CASE_NAME
        self._set_case_dir()
        self.time = initial_snapshot
        # Load initial conditions for progress variable and velocity
        self.x, self.y, u, v, theta = self._load_snapshot(initial_snapshot)
        # Set domain bounds
        self.x_bounds = (self.x.min(), self.x.max())
        self.y_bounds = (self.y.min(), self.y.max())
        # Set dx,dy
        self.dx = float(self.x[1, 0] - self.x[0, 0])
        self.dy = float(self.y[0, 1] - self.y[0, 0])
        # Generate signed distance field
        G0 = self._generate_signed_distance_field(theta, isoline)
        # Generate Centered fields
        self.g_field = self._make_scalar_centered_grid(G0)
        self.vel_field = self._make_vector_centered_grid(u,v)

    def solve(
        self,
        n_steps: int,
        dt: float,
        *,
        reinit_every: int = 1,
        reinit_iters: int = 5,
        reinit_dtau_factor: float = 0.3,
    ):
        """
        Solves the G-equation for n_steps with time step dt.
        """
        for istep in range(1, n_steps + 1):
            self._update_velocity_field(int(self.time))
            self.g_field = self._step(dt, self.g_field, self.vel_field)

            if reinit_every > 0 and istep % reinit_every == 0:
                self.g_field = self._reinitialize(
                    self.g_field,
                    self.dx,
                    self.dy,
                    n_iters=reinit_iters,
                    dtau_factor=reinit_dtau_factor,
                )

            self.time += dt

    def _step(
        self,
        dt: float,
        g_field: Field,
        v_field: Field,
    ) -> Field:
        """
        Performs a single time step in the G-equation evolution
        """
        g_adv = advect.semi_lagrangian(g_field, v_field, dt)

        grad = field.spatial_gradient(g_adv, at="center")
        grad_norm = field.vec_length(grad)

        sd = self._calculate_sd(g_adv)   # or constant field / scalar
        g_new = g_adv + dt * sd * grad_norm

        return g_new
    
    def _one_sided_derivatives(
        self,
        phi: jnp.ndarray,
        dx: float,
        dy: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        First-order one-sided derivatives.

        Boundary treatment:
        - x: zero-normal-gradient style replication at left/right boundaries
        - y: periodic
        """
        # x-direction: replicate boundary values
        phi_xm = jnp.concatenate([phi[:1, :], phi[:-1, :]], axis=0)
        phi_xp = jnp.concatenate([phi[1:, :], phi[-1:, :]], axis=0)

        # y-direction: periodic
        phi_ym = jnp.concatenate([phi[:, -1:], phi[:, :-1]], axis=1)
        phi_yp = jnp.concatenate([phi[:, 1:], phi[:, :1]], axis=1)

        dmx = (phi - phi_xm) / dx   # D_x^-
        dpx = (phi_xp - phi) / dx   # D_x^+
        dmy = (phi - phi_ym) / dy   # D_y^-
        dpy = (phi_yp - phi) / dy   # D_y^+

        return dmx, dpx, dmy, dpy
    
    def _upwind_advection(
        self,
        dmx: jnp.ndarray,
        dpx: jnp.ndarray,
        dmy: jnp.ndarray,
        dpy: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        First-order upwind discretization of u G_x + v G_y.
        """
        u_pos = jnp.maximum(u, 0.0)
        u_neg = jnp.minimum(u, 0.0)

        v_pos = jnp.maximum(v, 0.0)
        v_neg = jnp.minimum(v, 0.0)

        adv_x = u_pos * dmx + u_neg * dpx
        adv_y = v_pos * dmy + v_neg * dpy

        return adv_x + adv_y
    
    def _generate_signed_distance_field(self, field ,iso):
        """
        Generate a signed distance field based on the current state of g_field
        """
        level_set_idx = self._extract_isoline(field, iso)
        level_set_xy = self._contours_to_physical(level_set_idx)
        # Calculate distance of each point to isoline
        unsigned_distance = self._min_distance_to_level_set(level_set_xy)
        # Add a positive sign if above target isoa and negative if below
        sign = np.where(field >= iso, 1.0, -1.0).astype(np.float32)
        signed_distance = unsigned_distance * sign
        return signed_distance
        
    def _extract_isoline(self, field, iso_level) -> list[np.ndarray]:
        """
        Extracts the isoline from the field at the specified level
        """
        contours_idx = measure.find_contours(
            image = field,
            level=iso_level,
            fully_connected="high",
            positive_orientation="high",
        )
        return contours_idx

    def _min_distance_to_level_set(
        self,
        contours: list[np.ndarray],
        *,
        segment_chunk_size: int = 32,
    ) -> np.ndarray:
        """
        Computes the minimum distance of each grid point to all contour segments.
        """
        if not contours:
            raise ValueError("No contour found for the requested isoline.")

        points = np.column_stack([self.x.ravel(), self.y.ravel()]).astype(np.float32, copy=False)
        min_dist2 = np.full(points.shape[0], np.inf, dtype=np.float32)
        eps = np.float32(1.0e-12)

        for contour in contours:
            contour = np.asarray(contour, dtype=np.float32)
            if contour.shape[0] < 2:
                continue

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

        if not np.isfinite(min_dist2).any():
            raise ValueError("Contours were found, but none contained valid line segments.")

        return np.sqrt(min_dist2).astype(np.float32).reshape(self.x.shape)
    
    def _contours_to_physical(self, contours_idx: list[np.ndarray]) -> list[np.ndarray]:
        """
        Converts contour coordinates from index space to physical space using interpolation
        """
        x1d = self.x[:, 0].astype(np.float32)
        y1d = self.y[0, :].astype(np.float32)

        contours_phys = []
        for contour in contours_idx:
            row = contour[:, 0]
            col = contour[:, 1]

            x_phys = np.interp(row, np.arange(len(x1d), dtype=np.float32), x1d)
            y_phys = np.interp(col, np.arange(len(y1d), dtype=np.float32), y1d)

            contours_phys.append(np.column_stack([x_phys, y_phys]).astype(np.float32))

        return contours_phys
    
    def _load_snapshot(self, snapshot: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads the field snapshot from the dataset
        """
        points_path = self.case_dir / "points.hdf5"
        field_path = self.case_dir / f"structured_fields{snapshot:05d}.hdf5"

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
    
    def _update_velocity_field(self, snapshot):
        """
        Update the velocity field based on the current state of g_field
        """
        _,_,u,v,_ = self._load_snapshot(snapshot)
        print(f"Reading velocity field from {snapshot}")
        self.vel_field = self._make_vector_centered_grid(u,v)

    def _make_scalar_centered_grid(
        self, 
        values, 
    ) -> Field:
        """
        Create a centered grid for g_field and velocity
        """
        tensor = phimath.tensor(values, spatial("x,y"))
        g_bc = extrapolation.combine_sides(
            x=(extrapolation.BOUNDARY, extrapolation.BOUNDARY),
            y=(extrapolation.PERIODIC, extrapolation.PERIODIC),
        )
        return CenteredGrid(
            tensor,
            g_bc,
            bounds=Box["x,y", self.x_bounds[0]:self.x_bounds[1], self.y_bounds[0]:self.y_bounds[1]],
        )
    
    def _make_vector_centered_grid(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> Field:
        tensor = phimath.stack(
            {
                "x": phimath.tensor(u, spatial("x,y")),
                "y": phimath.tensor(v, spatial("x,y")),
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
            bounds=Box["x,y", self.x_bounds[0]:self.x_bounds[1], self.y_bounds[0]:self.y_bounds[1]],
        )

    def _set_case_dir(self):
        """
        Set the case directory based on the CASE_NAME
        """
        self.case_dir = Path(f"data/fields/structured_grids/{self.CASE_NAME}")

    def _calculate_sd(self, phi: np.ndarray) -> np.ndarray:
        # For now constant value
        return 0.2* np.ones_like(phi)
    
    def _calculate_normal_vector(self, g_field, eps = 1e-06) -> Field:
        grad = field.spatial_gradient(g_field, at="center")
        grad_norm = field.vec_length(grad)
        normal = grad / (grad_norm + eps)
        return normal
    
    def _calculate_curvature(self, g_field) -> Field:
        normal = self._calculate_normal_vector(g_field)
        curv = field.divergence(normal)
        return curv
    
    # START CHECK
        
    def _smoothed_sign(self, phi0: jnp.ndarray, dx: float, dy: float, eps_factor: float = 1.0) -> jnp.ndarray:
        dx = jnp.asarray(dx, dtype=phi0.dtype)
        dy = jnp.asarray(dy, dtype=phi0.dtype)
        eps = jnp.asarray(eps_factor, dtype=phi0.dtype) * jnp.minimum(dx, dy)
        return phi0 / jnp.sqrt(phi0 * phi0 + eps * eps)

    def _godunov_abs_grad_reinit(self, phi: jnp.ndarray, phi0: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
        """
        Godunov Hamiltonian for reinitialization PDE:
            phi_tau + S(phi0) (|grad phi| - 1) = 0
        """
        dmx, dpx, dmy, dpy = self._one_sided_derivatives(phi, dx, dy)
        s = self._smoothed_sign(phi0, dx, dy)

        gx_pos = jnp.maximum(jnp.maximum(dmx, 0.0) ** 2, jnp.minimum(dpx, 0.0) ** 2)
        gy_pos = jnp.maximum(jnp.maximum(dmy, 0.0) ** 2, jnp.minimum(dpy, 0.0) ** 2)

        gx_neg = jnp.maximum(jnp.minimum(dmx, 0.0) ** 2, jnp.maximum(dpx, 0.0) ** 2)
        gy_neg = jnp.maximum(jnp.minimum(dmy, 0.0) ** 2, jnp.maximum(dpy, 0.0) ** 2)

        use_pos = s >= 0.0
        gx = jnp.where(use_pos, gx_pos, gx_neg)
        gy = jnp.where(use_pos, gy_pos, gy_neg)

        return jnp.sqrt(gx + gy)

    def _reinitialize_step(
        self,
        g_field: Field,
        g0_field: Field,
        dtau: float,
        dx: float,
        dy: float,
    ) -> Field:
        phi = g_field.values.native(("x", "y"))
        phi0 = g0_field.values.native(("x", "y"))

        s = self._smoothed_sign(phi0, dx, dy)
        abs_grad = self._godunov_abs_grad_reinit(phi, phi0, dx, dy)
        phi_new = phi - dtau * s * (abs_grad - 1.0)

        return g_field.with_values(phimath.tensor(phi_new, spatial("x,y")))


    def _reinitialize(
        self,
        g_field: Field,
        dx: float,
        dy: float,
        *,
        n_iters: int = 5,
        dtau_factor: float = 0.3,
    ) -> Field:
        """
        Sussman-type reinitialization.
        Keeps the zero level set approximately fixed while driving |grad G| -> 1.
        """
        dtau = dtau_factor * min(dx, dy)
        g0_field = g_field
        g_re = g_field
        for _ in range(n_iters):
            g_re = self._reinitialize_step(g_re, g0_field, dtau, dx, dy)
        return g_re
    ### END CHECK

    def print_case_info(self):
        print(f"Case name: {self.CASE_NAME}")
        print(f"X-bounds: {self.x_bounds}")
        print(f"Y-bounds: {self.y_bounds}")
        print(f"Time: {self.time}")




def field_to_numpy(data: Field | np.ndarray) -> np.ndarray:
    if isinstance(data, np.ndarray):
        return data.astype(np.float32, copy=False)
    return np.asarray(data.values.numpy("x,y"), dtype=np.float32)


def save_plot(
    x: np.ndarray,
    y: np.ndarray,
    g0: Field | np.ndarray,
    g_final: Field | np.ndarray,
    output_path: Path,
) -> None:
    g0_np = field_to_numpy(g0)
    g_final_np = field_to_numpy(g_final)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    for ax, data, title in [
        (axes[0], g0_np, "Initial signed-distance field"),
        (axes[1], g_final_np, "Final G field"),
    ]:
        mesh = ax.pcolormesh(x, y, data, shading="auto", cmap="coolwarm")
        ax.contour(x, y, data, levels=[0.0], colors="k", linewidths=1.5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        fig.colorbar(mesh, ax=ax)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_animation(
    x: np.ndarray,
    y: np.ndarray,
    frames: list[Field | np.ndarray],
    output_path: Path,
    *,
    fps: int = 20,
) -> None:
    frames_np = [field_to_numpy(frame) for frame in frames]

    vmin = min(float(np.min(frame)) for frame in frames_np)
    vmax = max(float(np.max(frame)) for frame in frames_np)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    mesh = ax.pcolormesh(
        x,
        y,
        frames_np[0],
        shading="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )
    contour_set = ax.contour(x, y, frames_np[0], levels=[0.0], colors="k", linewidths=1.5)

    title = ax.set_title("G-equation evolution: step 0")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(mesh, ax=ax)

    def update(frame_idx: int):
        nonlocal contour_set

        frame = frames_np[frame_idx]

        mesh.set_array(frame[:-1, :-1].ravel())

        for coll in contour_set.collections:
            coll.remove()
        contour_set = ax.contour(x, y, frame, levels=[0.0], colors="k", linewidths=1.5)

        title.set_text(f"G-equation evolution: step {frame_idx}")
        return [mesh, title, *contour_set.collections]

    anim = FuncAnimation(fig, update, frames=len(frames_np), interval=1000 / fps, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

def main():
    solver = GSolver()
    print("Initializing")
    solver.initialize("phi_0.40/h400x025_ref", 300, 0.6)
    solver.print_case_info()
    print("Solving")
    g0 = solver.g_field
    solver.solve(n_steps= 2, dt = 1)
    g_final = solver.g_field
    print("Plotting")
    save_plot(
        solver.x,
        solver.y,
        g0,
        g_final,
        Path("g_evolution.png"),
    )
    save_animation(
        solver.x,
        solver.y,
        frames=[g0, g_final],
        output_path=Path("g_evolution.gif"),
        fps=1,
    )
    print("Done")

if __name__ == "__main__":
    main()


# Which numerical schemes should I use? Godunov Upwind?
# How often should I reinitialize? Should I do it with the PDE? Sussman type method
# Should I solve the equation on the entire field or in a bound around the interface?
# Should I be integrating with smaller timestep than the DNS data

"""
Curently using:
- Semi-Lagrangian advection for the G-equation evolution
- Central differences for spatial gradients (gradG)
- Sd = 0.2 everywhere (constant normal speed)
- A polyline to represent the zero-level set
- Sussman-type reinitialization every step to maintain the signed-distance property of g_field
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpi4py import MPI
from typing import Optional, List, Tuple

from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import Field
from pysemtools.io.ppymech.neksuite import pynekread

import cantera as ct
from .utils import _unwrap_scalar

# ----------------------------------------------------
# SEMDataset: load + access Nek data
# ----------------------------------------------------
class SEMDataset:
    def __init__(self, fname: str, comm: Optional[MPI.Comm] = None, gname: Optional[str] = None):
        self.comm = comm
        self.msh = Mesh(comm, create_connectivity=False)
        self.fld = Field(comm)

        # read coordinates/mesh from gname (geometry file)
        pynekread(gname, comm, msh=self.msh, fld=self.fld, overwrite_fld=False)

        # read actual field data from fname (time snapshot)
        pynekread(fname, comm, msh=self.msh, fld=self.fld, overwrite_fld=True)

        # Build derivative operators -> look more into this
        self.coef = Coef(self.msh, comm)

        # Cache coords -> delete?
        self.X, self.Y, self.Z = self.msh.x, self.msh.y, self.msh.z

        # Velocity
        vel = self.fld.fields.get("vel", None)
        self.u, self.v = vel[0], vel[1]
        self.w = vel[2] if len(vel) > 2 else np.zeros_like(self.u)


        # Temperature
        self.T = _unwrap_scalar(self.fld.fields.get("temp", None))

        # Scalars
        self.scalars = self.fld.fields.get("scal", None)

    def grad2d(self, F: np.ndarray):
        dFdx = self.coef.dudxyz(F, self.coef.drdx, self.coef.dsdx)
        dFdy = self.coef.dudxyz(F, self.coef.drdy, self.coef.dsdy)
        return dFdx, dFdy

    def hess2d(self, F: np.ndarray):
        dFdx, dFdy = self.grad2d(F)
        d2xx = self.coef.dudxyz(dFdx, self.coef.drdx, self.coef.dsdx)
        d2xy = self.coef.dudxyz(dFdx, self.coef.drdy, self.coef.dsdy)
        d2yy = self.coef.dudxyz(dFdy, self.coef.drdy, self.coef.dsdy)
        return d2xx, d2xy, d2yy

    def get_scalar_by_index(self, idx: int):
        if self.scalars is None:
            raise KeyError("No 'scal' array in this file.")
        return self.scalars[idx]

# ----------------------------------------------------
# FlameFront2D: progress variable, front extraction
# ----------------------------------------------------
class FlameFront2D:
    def __init__(self, ds: SEMDataset):
        self.ds = ds
        self.df = None

    @staticmethod
    def progress_variable_T(t:np.ndarray =None, t_u:float =None, t_b:float =None):
        den = t_b - t_u if t_b > t_u else 1.0
        return np.clip((t - t_u) / den, a_min=0.0, a_max=1.0)

    @staticmethod
    def band_mask(c: np.ndarray , c_level:float =0.38, tol:float =0.01):
        return (c > (c_level - tol)) & (c < (c_level + tol))

    def heat_release_mask(self, threshold_factor:float =0.15, heat_release_field=None):
        """Return mask where heat release > threshold_factor * max_heat_release"""
        max_hrr = np.max(heat_release_field)
        return heat_release_field > (threshold_factor * max_hrr)

    def make_front_dataframe(
        self,
        scalar_name_map: Optional[List[str]] = None,
        c_level:float =0.38,
        tol:float =0.01,
        include_first_vel_derivs:bool =True,
        include_second_vel_derivs:bool =True,
        include_T_derivs:bool =True,
        include_curvature_derivs:bool =True,
        sample_mode:str ="progress",      # "progress" for progress variable or "heat_release" for heat release rate
        hrr_factor:float =0.15,           #  only used if sample_mode="heat_release"
    ) -> None:

        X, Y, u, v, t = self.ds.X, self.ds.Y, self.ds.u, self.ds.v, self.ds.T

        t_u = np.percentile(t, q=2)
        t_b = np.percentile(t, q=98)
        c = self.progress_variable_T(t,t_u,t_b)

        # Choose mask based on sampling mode
        if sample_mode == "progress":
            mask = self.band_mask(c, c_level, tol)
        elif sample_mode == "heat_release":
            mask = self.heat_release_mask(threshold_factor=hrr_factor)
        else:
            raise ValueError("sample_mode must be 'progress' or 'heat_release'")

        data = {
            "x": X[mask],
            "y": Y[mask],
            "u": u[mask],
            "v": v[mask],
            "T": t[mask],
            "c": c[mask]
        }
        if include_first_vel_derivs:
            dudx, dudy = self.ds.grad2d(u)
            dvdx, dvdy = self.ds.grad2d(v)
            data.update({
                "dudx": dudx[mask],
                "dudy": dudy[mask],
                "dvdx": dvdx[mask],
                "dvdy": dvdy[mask]
            })
        if include_T_derivs:
            dTdx, dTdy = self.ds.grad2d(t)
            d2T_xx, d2T_xy, d2T_yy = self.ds.hess2d(t)
            data.update({
                "dTdx": dTdx[mask],
                "dTdy": dTdy[mask],
                "d2T_dx2": d2T_xx[mask],
                "d2T_dxdy": d2T_xy[mask],
                "d2T_dy2": d2T_yy[mask],
            })
        if include_second_vel_derivs:
            d2u_xx, d2u_xy, d2u_yy = self.ds.hess2d(u)
            d2v_xx, d2v_xy, d2v_yy = self.ds.hess2d(v)
            data.update({
                "d2u_dx2": d2u_xx[mask], "d2u_dxdy": d2u_xy[mask], "d2u_dy2": d2u_yy[mask],
                "d2v_dx2": d2v_xx[mask], "d2v_dxdy": d2v_xy[mask], "d2v_dy2": d2v_yy[mask],
            })

        if scalar_name_map and self.ds.scalars is not None:
            for i, name in enumerate(scalar_name_map):
                arr = self.ds.get_scalar_by_index(i)
                data[name] = arr[mask]

        #########################################################
        # To be changed once curvature is included in the data
        if include_curvature_derivs:
            if scalar_name_map and "curvature" in scalar_name_map:
                idx = scalar_name_map.index("curvature")
                curv = self.ds.get_scalar_by_index(idx)
            else:
                curv = self.ds.get_scalar_by_index(10)

            dcurv_dx = self.ds.coef.dudxyz(curv, self.ds.coef.drdx, self.ds.coef.dsdx)
            dcurv_dy = self.ds.coef.dudxyz(curv, self.ds.coef.drdy, self.ds.coef.dsdy)
            d2curv_dx2 = self.ds.coef.dudxyz(dcurv_dx, self.ds.coef.drdx, self.ds.coef.dsdx)
            d2curv_dxdy = self.ds.coef.dudxyz(dcurv_dx, self.ds.coef.drdy, self.ds.coef.dsdy)
            d2curv_dy2 = self.ds.coef.dudxyz(dcurv_dy, self.ds.coef.drdy, self.ds.coef.dsdy)

            data.update({
                "dcurv_dx": dcurv_dx[mask],
                "dcurv_dy": dcurv_dy[mask],
                "d2curv_dx2": d2curv_dx2[mask],
                "d2curv_dxdy": d2curv_dxdy[mask],
                "d2curv_dy2": d2curv_dy2[mask],
            })
        #########################################################

        self.df = pd.DataFrame(data)
        return

    def add_Le_to_dataset(self, species_list: list, eq_ratio:float =0.5, t_ref:float = 300, pref:float =1.0):
        Y_matrix = np.stack([self.df[name].values for name in species_list], axis=1)
        chem = ChemistryFeatures(
            mechanism= "chem.yaml",
            phi = eq_ratio,
            t_ref = t_ref,
            p_ref = pref
        )
        omegas, Le_D = chem.compute_features(self.df["T"].values, Y_matrix)
        for k, sp in enumerate(chem.gas.species_names):
            self.df[f"omega_{sp}"] = omegas[:, k]
        self.df["Le_def"] = Le_D
        return




class ChemistryFeatures:
    def __init__(self, mechanism:str ="chem.yaml", phi:float =1.0, t_ref:float =300.0, p_ref:float =1.0):
        self.gas = ct.Solution(mechanism)
        self.gas.transport_model = "Mix"
        self.phi = phi
        self.t_ref = t_ref
        self.p_ref = p_ref * ct.one_atm

    def compute_features(self, t:np.ndarray =None, y_matrix:np.ndarray =None) ->Tuple[np.ndarray, np.ndarray]:

        gas = self.gas
        omegas = []
        Le_D = []

        for i in range(y_matrix.shape[0]):
            gas.TPY = t[i] * self.t_ref, self.p_ref, y_matrix[i, :]
            omegas.append(gas.net_production_rates)
            # deficient reactant index
            if self.phi < 1.0:
                i_def = gas.species_index("H2")
            else:
                i_def = gas.species_index("O2")

            alpha = gas.thermal_conductivity / (gas.density * gas.cp)
            Le_D.append(alpha / gas.mix_diff_coeffs[i_def])

        return np.array(omegas), np.array(Le_D)

# ----------------------------------------------------
# Plot2D: quick visualization helpers
# ----------------------------------------------------
class Plot2D:

    @staticmethod
    def plot_field(msh, field, mode="sem", plot_name = None,levels=100, cmap="RdBu_r", vmin=None, vmax=None):
        """
        Plot a scalar field on a pySEMTools mesh.

        Parameters
        ----------
        msh : Mesh
            pySEMTools Mesh object (provides x,y arrays).
        field : ndarray or list
            Field data (e.g. fld.registry["u"]).
        mode : {"sem","tri"}
            sem = element-wise pcolormesh (preserve SEM structure).
            tri = tricontourf on flattened points.
        """


        def _to_numpy(a):
            if isinstance(a, list):
                a = a[0]
            if hasattr(a, "detach") and hasattr(a, "cpu"):
                a = a.detach().cpu().numpy()
            return np.asarray(a)

        F = _to_numpy(field)
        X = _to_numpy(msh.x)
        Y = _to_numpy(msh.y)

        if vmin is None: vmin = float(np.nanmin(F))
        if vmax is None: vmax = float(np.nanmax(F))

        fig, ax = plt.subplots(figsize=(6,4), dpi=160)

        if mode == "sem":
            for e in range(X.shape[0]):
                pc = ax.pcolormesh(
                    X[e,0,:,:], Y[e,0,:,:], F[e,0,:,:],
                    shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax
                )
            cbar = fig.colorbar(pc, ax=ax)
        else:
            x, y, f = X.ravel(), Y.ravel(), F.ravel()
            m = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
            tc = ax.tricontourf(x[m], y[m], f[m], levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(tc, ax=ax)

        cbar.set_label("field value")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title(f"Field plot of {plot_name}({'SEM patches' if mode=='sem' else 'triangulated'})")
        plt.show()

    @staticmethod
    def print_pearson(df: pd.DataFrame, cols: Optional[List[str]] = None):
        num = df[cols] if cols is not None else df.select_dtypes(include=[np.number])
        print(num.corr(method="pearson").round(3))


    @staticmethod
    def heat_map(df: pd.DataFrame, cols: list[str], max_points: int = 8000):
        from itertools import product
        """
        Pairwise scatter/histogram matrix with optional coloring by progress variable c.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset to visualize.
        cols : list of str
            List of column names to plot against each other.
        max_points : int
            Limit number of points (random sample) to keep plots responsive.
        """
        # downsample if necessary
        if len(df) > max_points:
            dfp = df.sample(max_points, random_state=0)
        else:
            dfp = df

        k = len(cols)
        fig, axes = plt.subplots(k, k, figsize=(2.6 * k, 2.6 * k), dpi=140)

        cvals = dfp["c"].values if "c" in dfp.columns else None
        sc = None  # for colorbar handle

        for i, j in product(range(k), range(k)):
            ax = axes[i, j]
            xi, yj = cols[j], cols[i]
            if i == j:
                ax.hist(dfp[xi].values, bins=40, alpha=0.85)
                ax.set_ylabel("")
            else:
                if cvals is None:
                    ax.scatter(dfp[xi].values, dfp[yj].values, s=6, alpha=0.4)
                else:
                    sc = ax.scatter(dfp[xi].values, dfp[yj].values, s=6, alpha=0.5,
                                    c=cvals, cmap="viridis")

            # tidy ticks
            if i < k - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(xi)
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(yj)

        # single colorbar if c is present
        if sc is not None:
            cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
            cb = plt.colorbar(sc, cax=cax)
            cb.set_label("progress variable c")

        plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List

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

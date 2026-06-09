"""
PCA From Scratch — Core Implementation
=======================================
Principal Component Analysis implemented entirely from first principles
using only NumPy. No sklearn, no scipy PCA — just linear algebra.

Mathematical foundation:
  1. Standardize: X_std = (X - μ) / σ
  2. Covariance: C = (1/(n-1)) * X_std.T @ X_std
  3. Eigen-decomposition: C = V Λ V^T
  4. Sort eigenpairs by descending eigenvalue
  5. Project: X_proj = X_std @ V_k   (V_k = top-k eigenvectors)
  6. Reconstruct: X_rec = X_proj @ V_k.T  (then un-standardize)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Union
import warnings

from .utils import (
    standardize,
    covariance_matrix,
    sort_eigenpairs,
    reconstruction_error,
)
from .visualization import (
    plot_explained_variance,
    plot_cumulative_variance,
    plot_2d_projection,
    plot_3d_projection,
    plot_eigenvectors,
    plot_original_data,
)


class PCAFromScratch:
    """
    Principal Component Analysis — implemented from mathematical first principles.

    This class follows the scikit-learn estimator API (fit / transform /
    fit_transform / inverse_transform) so it feels familiar, while every
    computation is done by hand with NumPy.

    Parameters
    ----------
    n_components : int or float or None
        Number of principal components to keep.
        - int  → keep exactly that many components
        - float (0, 1) → keep enough components to explain that fraction
          of variance (e.g. 0.95 keeps components explaining 95 % variance)
        - None → keep all components
    whiten : bool, default False
        Divide projected coordinates by sqrt(eigenvalue) so every component
        has unit variance. Useful before feeding into a downstream model.

    Attributes (available after fit)
    ----------
    components_ : ndarray, shape (n_components, n_features)
        Top-k eigenvectors (principal axes), stored as *rows*.
    explained_variance_ : ndarray, shape (n_components,)
        Eigenvalues of the covariance matrix for each retained component.
    explained_variance_ratio_ : ndarray, shape (n_components,)
        Fraction of total variance explained by each component.
    cumulative_variance_ratio_ : ndarray, shape (n_components,)
        Cumulative explained variance.
    mean_ : ndarray, shape (n_features,)
        Per-feature mean used for standardisation.
    std_ : ndarray, shape (n_features,)
        Per-feature standard deviation used for standardisation.
    n_samples_ : int
        Number of training samples.
    n_features_ : int
        Number of original features.
    eigenvalues_all_ : ndarray, shape (n_features,)
        All eigenvalues (sorted descending) — useful for scree plots.
    eigenvectors_all_ : ndarray, shape (n_features, n_features)
        All eigenvectors (columns) corresponding to eigenvalues_all_.
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        whiten: bool = False,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten

        # Filled by fit()
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.n_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None
        self.eigenvalues_all_: Optional[np.ndarray] = None
        self.eigenvectors_all_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "PCAFromScratch":
        """
        Fit PCA on the training data X.

        Steps
        -----
        1. Validate input.
        2. Standardise each feature (mean=0, std=1).
        3. Compute the sample covariance matrix C (shape: p×p).
        4. Eigen-decompose C → eigenvalues λ, eigenvectors V.
        5. Sort by descending eigenvalue.
        6. Resolve the number of components k.
        7. Retain the top-k eigenvectors.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        self
        """
        X = self._validate(X)
        n, p = X.shape
        self.n_samples_, self.n_features_ = n, p

        # ── Step 2: standardise ──────────────────────────────────────
        X_std, self.mean_, self.std_ = standardize(X)

        # ── Step 3: covariance matrix ────────────────────────────────
        # C[i,j] = cov(feature_i, feature_j)
        # Shape: (p, p)   — one entry per feature pair.
        C = covariance_matrix(X_std)

        # ── Step 4: eigen-decomposition ──────────────────────────────
        # numpy.linalg.eig returns complex values for non-symmetric
        # matrices; our covariance matrix *is* symmetric by construction,
        # so eigenvalues are real.  We cast to real to be safe.
        eigenvalues, eigenvectors = np.linalg.eig(C)
        eigenvalues = eigenvalues.real          # shape (p,)
        eigenvectors = eigenvectors.real        # shape (p, p) — columns are vectors

        # ── Step 5: sort descending ──────────────────────────────────
        eigenvalues, eigenvectors = sort_eigenpairs(eigenvalues, eigenvectors)

        self.eigenvalues_all_ = eigenvalues
        self.eigenvectors_all_ = eigenvectors   # columns = eigenvectors

        # Total variance = sum of all eigenvalues (each eigenvalue equals
        # the variance of the data along that principal direction).
        total_variance = eigenvalues.sum()

        # ── Step 6: choose k ─────────────────────────────────────────
        k = self._resolve_n_components(eigenvalues, total_variance)

        # ── Step 7: retain top-k ─────────────────────────────────────
        # eigenvectors_all_ columns → we take the first k columns.
        # Store as rows (scikit-learn convention) for easier indexing.
        self.components_ = eigenvectors[:, :k].T   # shape (k, p)

        self.explained_variance_ = eigenvalues[:k]
        self.explained_variance_ratio_ = eigenvalues[:k] / total_variance
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X into the principal-component subspace.

        X_proj = X_std @ V_k

        where V_k is the matrix of top-k eigenvectors (shape p×k).

        If whiten=True, we additionally divide each component i by
        sqrt(λ_i) so all components have unit variance.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        X_projected : ndarray, shape (n_samples, n_components)
        """
        self._check_is_fitted()
        X = self._validate(X)

        # Standardise with the *training* mean and std
        X_std = (X - self.mean_) / self.std_

        # Project: each row of X_std is a sample; we multiply by V_k
        # (shape p×k) to get k-dimensional coordinates.
        # components_ rows are eigenvectors → V_k = components_.T
        V_k = self.components_.T          # shape (p, k)
        X_proj = X_std @ V_k              # shape (n, k)

        if self.whiten:
            # Divide by sqrt of eigenvalue so component variances = 1
            X_proj = X_proj / np.sqrt(self.explained_variance_)

        return X_proj

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Convenience: fit on X, then return its projection."""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_proj: np.ndarray) -> np.ndarray:
        """
        Reconstruct the *standardised* feature space from projections,
        then un-standardise to recover approximate original scale.

        X_rec_std = X_proj @ V_k.T
        X_rec     = X_rec_std * σ + μ

        The reconstruction is exact only if k == n_features; otherwise
        we incur a reconstruction error proportional to the explained
        variance *not* captured by the top-k components.

        Parameters
        ----------
        X_proj : ndarray, shape (n_samples, n_components)

        Returns
        -------
        X_reconstructed : ndarray, shape (n_samples, n_features)
        """
        self._check_is_fitted()

        if self.whiten:
            # Undo whitening before back-projection
            X_proj = X_proj * np.sqrt(self.explained_variance_)

        V_k = self.components_.T          # shape (p, k)
        X_rec_std = X_proj @ V_k.T        # shape (n, p)

        # Un-standardise
        X_rec = X_rec_std * self.std_ + self.mean_
        return X_rec

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def explained_variance(self, verbose: bool = True) -> dict:
        """
        Return (and optionally print) a summary of the explained variance.

        Returns
        -------
        dict with keys:
          'eigenvalues', 'explained_variance_ratio',
          'cumulative_variance_ratio', 'n_components'
        """
        self._check_is_fitted()

        summary = {
            "n_components": len(self.explained_variance_),
            "eigenvalues": self.explained_variance_,
            "explained_variance_ratio": self.explained_variance_ratio_,
            "cumulative_variance_ratio": self.cumulative_variance_ratio_,
        }

        if verbose:
            print("=" * 55)
            print(f"PCA — Explained Variance Summary  (k={summary['n_components']})")
            print("=" * 55)
            print(f"{'PC':>4}  {'Eigenvalue':>12}  {'Var %':>8}  {'Cumul %':>9}")
            print("-" * 40)
            for i, (ev, evr, cum) in enumerate(
                zip(
                    self.explained_variance_,
                    self.explained_variance_ratio_,
                    self.cumulative_variance_ratio_,
                ),
                start=1,
            ):
                print(f"PC{i:>2}  {ev:>12.4f}  {evr*100:>7.2f}%  {cum*100:>8.2f}%")
            print("=" * 55)

        return summary

    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Mean squared reconstruction error on dataset X.

        MSE = mean((X - inverse_transform(transform(X)))**2)

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)

        Returns
        -------
        float  — mean squared error
        """
        self._check_is_fitted()
        X = self._validate(X)
        X_rec = self.inverse_transform(self.transform(X))
        return reconstruction_error(X, X_rec)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_variance(
        self,
        figsize: tuple = (14, 5),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Side-by-side bar + cumulative-line plots of explained variance.
        Shows *all* components so users can see the full scree.
        """
        self._check_is_fitted()

        all_ratios = self.eigenvalues_all_ / self.eigenvalues_all_.sum()
        cum_ratios = np.cumsum(all_ratios)

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        plot_explained_variance(axes[0], all_ratios, n_selected=len(self.explained_variance_))
        plot_cumulative_variance(axes[1], cum_ratios, n_selected=len(self.explained_variance_))

        fig.suptitle("PCA — Explained Variance", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_projection(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        figsize: tuple = (7, 6),
        save_path: Optional[str] = None,
        title: str = "PCA Projection",
    ) -> plt.Figure:
        """
        Scatter plot of the projected data (PC1 vs PC2).
        If labels are provided each class gets its own colour.
        """
        self._check_is_fitted()
        X = self._validate(X)
        X_proj = self.transform(X)

        k = X_proj.shape[1]
        if k < 2:
            raise ValueError("Need at least 2 components to plot a 2-D projection.")

        fig, ax = plt.subplots(figsize=figsize)
        plot_2d_projection(
            ax,
            X_proj,
            labels=labels,
            evr=self.explained_variance_ratio_,
            title=title,
        )
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_eigenvectors(
        self,
        feature_names: Optional[list] = None,
        figsize: tuple = (10, 4),
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Heatmap of the top-k eigenvectors (loading matrix).
        Rows = principal components, Columns = original features.
        """
        self._check_is_fitted()
        fig, ax = plt.subplots(figsize=figsize)
        plot_eigenvectors(ax, self.components_, feature_names=feature_names)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_full_dashboard(
        self,
        X: np.ndarray,
        labels: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        save_path: Optional[str] = None,
        title: str = "PCA Dashboard",
    ) -> plt.Figure:
        """
        Comprehensive 4-panel dashboard:
          [top-left]  2-D PCA scatter
          [top-right] explained variance bar
          [bot-left]  cumulative variance line
          [bot-right] eigenvector heatmap
        """
        self._check_is_fitted()
        X = self._validate(X)
        X_proj = self.transform(X)

        all_ratios = self.eigenvalues_all_ / self.eigenvalues_all_.sum()
        cum_ratios = np.cumsum(all_ratios)

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

        ax_proj = fig.add_subplot(gs[0, 0])
        ax_bar  = fig.add_subplot(gs[0, 1])
        ax_cum  = fig.add_subplot(gs[1, 0])
        ax_heat = fig.add_subplot(gs[1, 1])

        plot_2d_projection(ax_proj, X_proj, labels=labels,
                           evr=self.explained_variance_ratio_, title="PC1 vs PC2")
        plot_explained_variance(ax_bar, all_ratios,
                                n_selected=len(self.explained_variance_))
        plot_cumulative_variance(ax_cum, cum_ratios,
                                 n_selected=len(self.explained_variance_))
        plot_eigenvectors(ax_heat, self.components_,
                          feature_names=feature_names)

        fig.suptitle(title, fontsize=16, fontweight="bold")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_n_components(
        self, eigenvalues: np.ndarray, total_variance: float
    ) -> int:
        """Translate self.n_components into a concrete integer k."""
        p = len(eigenvalues)

        if self.n_components is None:
            return p

        if isinstance(self.n_components, float):
            if not 0 < self.n_components <= 1:
                raise ValueError("float n_components must be in (0, 1].")
            # Minimum k so that cumulative variance ≥ threshold
            cum = np.cumsum(eigenvalues) / total_variance
            k = int(np.searchsorted(cum, self.n_components) + 1)
            return min(k, p)

        if isinstance(self.n_components, int):
            if self.n_components < 1 or self.n_components > p:
                raise ValueError(
                    f"n_components must be between 1 and n_features={p}."
                )
            return self.n_components

        raise TypeError(
            f"n_components must be int, float, or None; got {type(self.n_components)}."
        )

    @staticmethod
    def _validate(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (got {X.ndim}-D).")
        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values.")
        return X

    def _check_is_fitted(self) -> None:
        if self.components_ is None:
            raise RuntimeError("Call fit() before using this method.")

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_transformed(
        self, X: np.ndarray, path: str, labels: Optional[np.ndarray] = None
    ) -> None:
        """
        Project X and save the result to a CSV file.

        Parameters
        ----------
        X     : original data, shape (n_samples, n_features)
        path  : destination CSV path
        labels: optional class labels (added as last column)
        """
        import pandas as pd

        X_proj = self.transform(X)
        k = X_proj.shape[1]
        cols = [f"PC{i+1}" for i in range(k)]
        df = pd.DataFrame(X_proj, columns=cols)
        if labels is not None:
            df["label"] = labels
        df.to_csv(path, index=False)
        print(f"Saved transformed data → {path}")

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self.components_ is not None
        return (
            f"PCAFromScratch("
            f"n_components={self.n_components}, "
            f"whiten={self.whiten}, "
            f"fitted={fitted})"
        )

"""
PCA From Scratch — Utility Functions
=====================================
Pure NumPy helpers that form the mathematical backbone of the PCA pipeline.
Each function is a single responsibility unit with clear docstrings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(
    path: str,
    label_col: Optional[str] = None,
    drop_cols: Optional[list] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], list]:
    """
    Load a numerical CSV dataset, optionally extracting a label column.

    Parameters
    ----------
    path      : filesystem path to the CSV file
    label_col : name of the column containing class labels (string)
                If None, no labels are extracted.
    drop_cols : list of column names to ignore entirely

    Returns
    -------
    X            : ndarray, shape (n_samples, n_features)  — feature matrix
    labels       : ndarray or None — class labels
    feature_names: list of str    — names of the retained feature columns
    """
    df = pd.read_csv(path)

    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    labels: Optional[np.ndarray] = None
    if label_col and label_col in df.columns:
        labels = df[label_col].values
        df = df.drop(columns=[label_col])

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    feature_names = list(df.columns)
    X = df.values.astype(float)

    return X, labels, feature_names


# ──────────────────────────────────────────────────────────────────────────────
# Standardisation
# ──────────────────────────────────────────────────────────────────────────────

def standardize(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score standardise each feature column independently.

    Formula  →  X_std[:, j] = (X[:, j] − μ_j) / σ_j

    Why standardise?
    PCA finds directions of maximum *variance*. If feature A is measured
    in kilometres (range 0–1000) and feature B in millimetres (0–1), the
    covariance matrix will be dominated by feature A purely because of its
    scale. Standardising levels the playing field so every feature starts
    with variance 1.

    Parameters
    ----------
    X    : ndarray, shape (n_samples, n_features)
    mean : pre-computed column means  (use training mean at test time)
    std  : pre-computed column stds   (use training std  at test time)

    Returns
    -------
    X_std : ndarray, shape (n_samples, n_features) — standardised
    mean  : ndarray, shape (n_features,)
    std   : ndarray, shape (n_features,)
    """
    if mean is None:
        mean = X.mean(axis=0)          # μ_j = (1/n) Σ x_{ij}
    if std is None:
        std = X.std(axis=0, ddof=1)    # σ_j = sqrt( (1/(n-1)) Σ (x_{ij}-μ_j)² )

    # Guard against zero-variance features (constant columns)
    std_safe = np.where(std == 0, 1.0, std)
    if np.any(std == 0):
        import warnings
        warnings.warn(
            "One or more features have zero variance and will not be scaled.",
            UserWarning,
        )

    X_std = (X - mean) / std_safe
    return X_std, mean, std


# ──────────────────────────────────────────────────────────────────────────────
# Covariance matrix
# ──────────────────────────────────────────────────────────────────────────────

def covariance_matrix(X_std: np.ndarray) -> np.ndarray:
    """
    Compute the sample covariance matrix of the (already standardised) data.

    Formula
    -------
      C = (1 / (n − 1)) · X_std.T @ X_std

    Shape  : (n_features, n_features)  — one entry per feature pair.

    Mathematical intuition
    ----------------------
    C[i, j] measures how much features i and j vary *together*:
      • C[i, j] > 0  → they tend to increase/decrease in tandem
      • C[i, j] < 0  → when one rises the other falls
      • C[i, i]      → variance of feature i  (equals 1 if X is standardised)

    The covariance matrix is symmetric (C = Cᵀ) and positive semi-definite,
    which guarantees real, non-negative eigenvalues.

    Parameters
    ----------
    X_std : ndarray, shape (n_samples, n_features) — standardised data

    Returns
    -------
    C : ndarray, shape (n_features, n_features)
    """
    n = X_std.shape[0]
    # (n_features, n_samples) @ (n_samples, n_features) → (n_features, n_features)
    C = (X_std.T @ X_std) / (n - 1)
    return C


# ──────────────────────────────────────────────────────────────────────────────
# Eigenpair sorting
# ──────────────────────────────────────────────────────────────────────────────

def sort_eigenpairs(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort eigenvalues in descending order and rearrange eigenvectors to match.

    The first principal component is defined as the direction of *maximum*
    variance (largest eigenvalue). Sorting ensures PC1 always captures more
    variance than PC2, and so on.

    Parameters
    ----------
    eigenvalues  : ndarray, shape (p,)   — one eigenvalue per feature
    eigenvectors : ndarray, shape (p, p) — columns are eigenvectors

    Returns
    -------
    eigenvalues_sorted  : ndarray, shape (p,)
    eigenvectors_sorted : ndarray, shape (p, p)  — columns reordered
    """
    # argsort returns ascending order → reverse with [::-1] for descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx]


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction error
# ──────────────────────────────────────────────────────────────────────────────

def reconstruction_error(X_original: np.ndarray, X_reconstructed: np.ndarray) -> float:
    """
    Mean squared error between original and reconstructed data.

    MSE = (1/N) Σᵢ ||x_i − x̂_i||²

    A lower MSE means the chosen k components faithfully represent the data.
    When k == n_features the error is (approximately) zero.

    Parameters
    ----------
    X_original      : ndarray, shape (n_samples, n_features)
    X_reconstructed : ndarray, shape (n_samples, n_features)

    Returns
    -------
    float — mean squared reconstruction error
    """
    diff = X_original - X_reconstructed
    # Squared Frobenius norm divided by number of elements
    return float(np.mean(diff ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# Automatic component selection
# ──────────────────────────────────────────────────────────────────────────────

def select_n_components_by_variance(
    eigenvalues: np.ndarray,
    threshold: float = 0.95,
) -> int:
    """
    Return the minimum number of components needed to explain *threshold*
    fraction of total variance.

    Parameters
    ----------
    eigenvalues : ndarray, shape (p,) — sorted descending
    threshold   : float in (0, 1], e.g. 0.95 means 95 %

    Returns
    -------
    k : int
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1].")

    total = eigenvalues.sum()
    cumulative = np.cumsum(eigenvalues) / total
    # searchsorted finds the index where cumulative first reaches threshold
    k = int(np.searchsorted(cumulative, threshold)) + 1
    return min(k, len(eigenvalues))


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic dataset generator
# ──────────────────────────────────────────────────────────────────────────────

def make_synthetic_dataset(
    n_samples: int = 300,
    n_features: int = 5,
    n_informative: int = 2,
    n_classes: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic Gaussian dataset with known cluster structure.
    Useful for demonstrating that PCA recovers the low-dimensional structure.

    Parameters
    ----------
    n_samples    : total number of rows
    n_features   : number of columns (dimensions)
    n_informative: number of dimensions with real variance signal
    n_classes    : number of Gaussian clusters
    random_state : random seed for reproducibility

    Returns
    -------
    X      : ndarray, shape (n_samples, n_features)
    labels : ndarray, shape (n_samples,)  — integer class index 0..n_classes-1
    """
    rng = np.random.default_rng(random_state)
    n_per_class = n_samples // n_classes

    # Random cluster centres in the informative subspace
    centres = rng.uniform(-5, 5, size=(n_classes, n_informative))

    X_parts, label_parts = [], []
    for c_idx, centre in enumerate(centres):
        # Samples in informative dimensions
        x_inf = rng.normal(loc=centre, scale=1.0, size=(n_per_class, n_informative))
        # Noise dimensions carry no useful signal
        x_noise = rng.normal(scale=0.3, size=(n_per_class, n_features - n_informative))
        X_parts.append(np.hstack([x_inf, x_noise]))
        label_parts.append(np.full(n_per_class, c_idx, dtype=int))

    X = np.vstack(X_parts)
    labels = np.concatenate(label_parts)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], labels[perm]

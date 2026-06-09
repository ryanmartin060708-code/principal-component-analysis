"""
PCA From Scratch
================
Principal Component Analysis implemented entirely from mathematical first
principles — no scikit-learn, no scipy PCA.  Only NumPy + Matplotlib.

Quick start
-----------
>>> from src import PCAFromScratch
>>> pca = PCAFromScratch(n_components=2)
>>> X_proj = pca.fit_transform(X)
>>> pca.explained_variance()
>>> pca.plot_variance()
"""

from .pca import PCAFromScratch
from .utils import (
    load_csv,
    standardize,
    covariance_matrix,
    sort_eigenpairs,
    reconstruction_error,
    select_n_components_by_variance,
    make_synthetic_dataset,
)
from .visualization import (
    plot_2d_projection,
    plot_3d_projection,
    plot_explained_variance,
    plot_cumulative_variance,
    plot_eigenvectors,
    plot_original_data,
    plot_before_after,
)

__all__ = [
    "PCAFromScratch",
    "load_csv",
    "standardize",
    "covariance_matrix",
    "sort_eigenpairs",
    "reconstruction_error",
    "select_n_components_by_variance",
    "make_synthetic_dataset",
    "plot_2d_projection",
    "plot_3d_projection",
    "plot_explained_variance",
    "plot_cumulative_variance",
    "plot_eigenvectors",
    "plot_original_data",
    "plot_before_after",
]

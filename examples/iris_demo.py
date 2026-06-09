"""
examples/iris_demo.py
=====================
Demonstrate PCA From Scratch on the Iris dataset.

The Iris dataset has 4 features (sepal length, sepal width, petal length,
petal width) measured on 150 flowers across 3 species.  PCA maps them to
2-D while keeping most of the variance, and the 3 species become clearly
separable in the projection.

Run from project root:
    python examples/iris_demo.py
"""

import os
import sys
import urllib.request
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless — saves plots to disk instead of opening a window
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import PCAFromScratch, load_csv

# ── 1. Download Iris CSV if not cached ────────────────────────────────────────
IRIS_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
IRIS_PATH = os.path.join(ROOT, "datasets", "iris.csv")

def download_iris() -> None:
    """Fetch the raw UCI Iris file and add a header row."""
    os.makedirs(os.path.dirname(IRIS_PATH), exist_ok=True)
    if os.path.exists(IRIS_PATH):
        return
    print("Downloading Iris dataset …")
    tmp_path = IRIS_PATH + ".tmp"
    urllib.request.urlretrieve(IRIS_URL, tmp_path)
    header = "sepal_length,sepal_width,petal_length,petal_width,species\n"
    with open(tmp_path) as f:
        rows = [r for r in f.readlines() if r.strip()]   # drop trailing blank line
    with open(IRIS_PATH, "w") as f:
        f.write(header)
        f.writelines(rows)
    os.remove(tmp_path)
    print(f"Saved → {IRIS_PATH}")

download_iris()

# ── 2. Load data ──────────────────────────────────────────────────────────────
X, labels, feature_names = load_csv(IRIS_PATH, label_col="species")
print(f"\nIris dataset: {X.shape[0]} samples × {X.shape[1]} features")
print(f"Feature names : {feature_names}")
print(f"Classes       : {sorted(set(labels))}")

# ── 3. Fit PCA with 2 components ──────────────────────────────────────────────
pca2 = PCAFromScratch(n_components=2)
X_2d = pca2.fit_transform(X)
pca2.explained_variance()

# ── 4. Also fit with 3 components for 3-D demo ────────────────────────────────
pca3 = PCAFromScratch(n_components=3)
X_3d = pca3.fit_transform(X)

# ── 5. Reconstruction error comparison ────────────────────────────────────────
print("\nReconstruction Error Comparison")
print("-" * 40)
for k in [1, 2, 3, 4]:
    pca_k = PCAFromScratch(n_components=k)
    pca_k.fit(X)
    err = pca_k.reconstruction_error(X)
    var = pca_k.cumulative_variance_ratio_[-1] * 100
    print(f"  k={k}  MSE={err:.4f}   cumulative variance={var:.1f}%")

# ── 6. Save plots ─────────────────────────────────────────────────────────────
IMAGES = os.path.join(ROOT, "images")
os.makedirs(IMAGES, exist_ok=True)

# 6a. Full dashboard
fig_dash = pca2.plot_full_dashboard(
    X, labels=labels, feature_names=feature_names,
    save_path=os.path.join(IMAGES, "iris_dashboard.png"),
    title="Iris Dataset — PCA Dashboard",
)
plt.close(fig_dash)
print("\nSaved → images/iris_dashboard.png")

# 6b. Variance plots
fig_var = pca2.plot_variance(save_path=os.path.join(IMAGES, "iris_variance.png"))
plt.close(fig_var)
print("Saved → images/iris_variance.png")

# 6c. Eigenvector loadings
fig_ev = pca2.plot_eigenvectors(
    feature_names=feature_names,
    save_path=os.path.join(IMAGES, "iris_loadings.png"),
)
plt.close(fig_ev)
print("Saved → images/iris_loadings.png")

# 6d. Before / after comparison
from src.visualization import plot_before_after
fig_ba = plot_before_after(
    X, X_2d,
    labels=labels,
    feature_names=feature_names,
    evr=pca2.explained_variance_ratio_,
    save_path=os.path.join(IMAGES, "iris_before_after.png"),
)
plt.close(fig_ba)
print("Saved → images/iris_before_after.png")

# 6e. 3-D projection
from src.visualization import plot_3d_projection
fig_3d = plot_3d_projection(
    X_3d, labels=labels,
    evr=pca3.explained_variance_ratio_,
    title="Iris — 3D PCA Projection",
    save_path=os.path.join(IMAGES, "iris_3d.png"),
)
plt.close(fig_3d)
print("Saved → images/iris_3d.png")

# ── 7. Save transformed data ──────────────────────────────────────────────────
pca2.save_transformed(
    X,
    path=os.path.join(ROOT, "datasets", "iris_pca2.csv"),
    labels=labels,
)

print("\n✓ Iris demo complete.")

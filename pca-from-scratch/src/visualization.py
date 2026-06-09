"""
PCA From Scratch — Visualization Module
========================================
Publication-quality Matplotlib plots for every stage of the PCA pipeline.
All functions accept an Axes object so they can be embedded in any layout.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
from typing import Optional, List


# ── palette ───────────────────────────────────────────────────────────────────
# Accessible, high-contrast palette (works on screen and in print)
_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]
_HIGHLIGHT = "#E03E3E"   # selected-component accent
_GRID_ALPHA = 0.25


def _label_colors(labels: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Map integer/string labels → per-row hex colours."""
    if labels is None:
        return None
    unique = list(dict.fromkeys(labels))   # preserves insertion order
    lut = {u: _PALETTE[i % len(_PALETTE)] for i, u in enumerate(unique)}
    return np.array([lut[l] for l in labels])


# ──────────────────────────────────────────────────────────────────────────────
# 2-D projection scatter
# ──────────────────────────────────────────────────────────────────────────────

def plot_2d_projection(
    ax: plt.Axes,
    X_proj: np.ndarray,
    labels: Optional[np.ndarray] = None,
    evr: Optional[np.ndarray] = None,
    title: str = "PCA — 2D Projection",
    alpha: float = 0.75,
    s: int = 40,
) -> None:
    """
    Scatter plot of the first two principal components.

    Parameters
    ----------
    ax     : Matplotlib Axes to draw on
    X_proj : ndarray, shape (n_samples, ≥2) — projected coordinates
    labels : optional class labels for colouring
    evr    : explained variance ratio array (used to annotate axes)
    title  : plot title
    alpha  : marker transparency
    s      : marker size
    """
    pc1 = X_proj[:, 0]
    pc2 = X_proj[:, 1]

    if labels is not None:
        unique_labels = list(dict.fromkeys(labels))
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(
                pc1[mask], pc2[mask],
                c=_PALETTE[i % len(_PALETTE)],
                label=str(lbl),
                alpha=alpha, s=s, edgecolors="white", linewidths=0.4,
            )
        ax.legend(title="Class", fontsize=8, title_fontsize=9,
                  framealpha=0.85, edgecolor="#cccccc")
    else:
        ax.scatter(pc1, pc2, c=_PALETTE[0], alpha=alpha, s=s,
                   edgecolors="white", linewidths=0.4)

    xlabel = "PC 1"
    ylabel = "PC 2"
    if evr is not None and len(evr) >= 2:
        xlabel += f"  ({evr[0]*100:.1f} %)"
        ylabel += f"  ({evr[1]*100:.1f} %)"

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, alpha=_GRID_ALPHA, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# 3-D projection scatter
# ──────────────────────────────────────────────────────────────────────────────

def plot_3d_projection(
    X_proj: np.ndarray,
    labels: Optional[np.ndarray] = None,
    evr: Optional[np.ndarray] = None,
    title: str = "PCA — 3D Projection",
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    3-D scatter of the first three principal components.
    Returns a Figure (creates its own axes with projection='3d').
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the projection

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    pc1, pc2, pc3 = X_proj[:, 0], X_proj[:, 1], X_proj[:, 2]

    if labels is not None:
        unique_labels = list(dict.fromkeys(labels))
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(
                pc1[mask], pc2[mask], pc3[mask],
                c=_PALETTE[i % len(_PALETTE)],
                label=str(lbl), alpha=0.75, s=30, edgecolors="white", linewidths=0.3,
            )
        ax.legend(title="Class", fontsize=8, title_fontsize=9)
    else:
        ax.scatter(pc1, pc2, pc3, c=_PALETTE[0], alpha=0.75, s=30,
                   edgecolors="white", linewidths=0.3)

    def _ax_label(prefix, idx):
        base = f"{prefix} {idx+1}"
        if evr is not None and len(evr) > idx:
            base += f"  ({evr[idx]*100:.1f} %)"
        return base

    ax.set_xlabel(_ax_label("PC", 0), fontsize=9)
    ax.set_ylabel(_ax_label("PC", 1), fontsize=9)
    ax.set_zlabel(_ax_label("PC", 2), fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Explained variance bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_explained_variance(
    ax: plt.Axes,
    explained_variance_ratio: np.ndarray,
    n_selected: Optional[int] = None,
    title: str = "Explained Variance per Component",
) -> None:
    """
    Horizontal bar chart of explained variance for every component.
    Bars corresponding to selected components are highlighted.

    Parameters
    ----------
    ax                     : Matplotlib Axes
    explained_variance_ratio: ndarray, shape (p,)  — ALL components
    n_selected             : how many are retained (highlighted in red)
    title                  : chart title
    """
    p = len(explained_variance_ratio)
    x = np.arange(1, p + 1)
    colors = [
        _HIGHLIGHT if (n_selected and i < n_selected) else "#a8c4e0"
        for i in range(p)
    ]

    bars = ax.bar(x, explained_variance_ratio * 100, color=colors,
                  edgecolor="white", linewidth=0.6, zorder=2)

    # Value labels on bars
    for bar, val in zip(bars, explained_variance_ratio * 100):
        if val > 2:   # only label if bar is tall enough to read
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=7, color="#333333",
            )

    ax.set_xlabel("Principal Component", fontsize=10)
    ax.set_ylabel("Explained Variance (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x], fontsize=8)
    ax.set_ylim(0, max(explained_variance_ratio * 100) * 1.2)
    ax.grid(axis="y", alpha=_GRID_ALPHA, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if n_selected:
        legend_elements = [
            mpatches.Patch(color=_HIGHLIGHT, label=f"Selected (k={n_selected})"),
            mpatches.Patch(color="#a8c4e0", label="Discarded"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, framealpha=0.85)


# ──────────────────────────────────────────────────────────────────────────────
# Cumulative variance line chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_cumulative_variance(
    ax: plt.Axes,
    cumulative_variance_ratio: np.ndarray,
    n_selected: Optional[int] = None,
    thresholds: Optional[list] = None,
    title: str = "Cumulative Explained Variance",
) -> None:
    """
    Step-line plot of cumulative explained variance.
    Horizontal reference lines at common variance thresholds (80, 90, 95 %).

    Parameters
    ----------
    ax                       : Matplotlib Axes
    cumulative_variance_ratio: ndarray, shape (p,) — ALL components
    n_selected               : draw a vertical rule at this component count
    thresholds               : list of fractions to draw horizontal rules (default [0.8, 0.9, 0.95])
    title                    : chart title
    """
    if thresholds is None:
        thresholds = [0.80, 0.90, 0.95]

    p = len(cumulative_variance_ratio)
    x = np.arange(1, p + 1)

    ax.plot(x, cumulative_variance_ratio * 100,
            marker="o", markersize=5, color=_PALETTE[0],
            linewidth=2, zorder=3, label="Cumulative variance")
    ax.fill_between(x, cumulative_variance_ratio * 100, alpha=0.12,
                    color=_PALETTE[0], zorder=2)

    # Reference lines
    for thr in thresholds:
        ax.axhline(thr * 100, color="#aaaaaa", linestyle="--",
                   linewidth=1, alpha=0.8, zorder=1)
        ax.text(
            p * 0.02, thr * 100 + 0.5,
            f"{int(thr*100)} %",
            fontsize=7, color="#666666", va="bottom",
        )

    # Vertical rule for selected k
    if n_selected:
        ax.axvline(n_selected, color=_HIGHLIGHT, linestyle=":",
                   linewidth=1.5, zorder=4, label=f"k = {n_selected}")
        ax.legend(fontsize=8, framealpha=0.85)

    ax.set_xlabel("Number of Components", fontsize=10)
    ax.set_ylabel("Cumulative Variance (%)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x], fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=_GRID_ALPHA, linestyle="--", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# Eigenvector heatmap (loading matrix)
# ──────────────────────────────────────────────────────────────────────────────

def plot_eigenvectors(
    ax: plt.Axes,
    components: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "Principal Component Loadings",
    cmap: str = "RdBu_r",
) -> None:
    """
    Heatmap of the loading matrix (rows = PCs, columns = original features).
    Cell colour encodes the signed contribution of each feature to each PC:
      • Deep red  → large positive loading
      • Deep blue → large negative loading
      • White     → near-zero (feature barely contributes to this PC)

    Parameters
    ----------
    ax            : Matplotlib Axes
    components    : ndarray, shape (k, p) — rows are eigenvectors
    feature_names : list of p feature name strings
    title         : chart title
    cmap          : diverging colormap name
    """
    k, p = components.shape
    if feature_names is None:
        feature_names = [f"F{i+1}" for i in range(p)]

    # Symmetric colour scale
    vmax = np.abs(components).max()

    im = ax.imshow(
        components,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
    )

    # Axis labels
    ax.set_xticks(np.arange(p))
    ax.set_xticklabels(feature_names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(k))
    ax.set_yticklabels([f"PC{i+1}" for i in range(k)], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)

    # Cell annotations
    for row in range(k):
        for col in range(p):
            val = components[row, col]
            text_color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(col, row, f"{val:.2f}",
                    ha="center", va="center", fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="Loading")


# ──────────────────────────────────────────────────────────────────────────────
# Original data scatter (up to 2-D)
# ──────────────────────────────────────────────────────────────────────────────

def plot_original_data(
    ax: plt.Axes,
    X: np.ndarray,
    labels: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    title: str = "Original Data (first 2 features)",
) -> None:
    """
    Scatter plot of the first two features of the raw data.
    Useful as a 'before PCA' reference panel.
    """
    f1 = X[:, 0]
    f2 = X[:, 1] if X.shape[1] > 1 else np.zeros(len(X))

    if labels is not None:
        unique_labels = list(dict.fromkeys(labels))
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            ax.scatter(f1[mask], f2[mask],
                       c=_PALETTE[i % len(_PALETTE)],
                       label=str(lbl), alpha=0.7, s=35,
                       edgecolors="white", linewidths=0.4)
        ax.legend(title="Class", fontsize=8, title_fontsize=9, framealpha=0.85)
    else:
        ax.scatter(f1, f2, c=_PALETTE[0], alpha=0.7, s=35,
                   edgecolors="white", linewidths=0.4)

    fn = feature_names or [f"Feature {i+1}" for i in range(X.shape[1])]
    ax.set_xlabel(fn[0], fontsize=10)
    ax.set_ylabel(fn[1] if X.shape[1] > 1 else "—", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.grid(True, alpha=_GRID_ALPHA, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ──────────────────────────────────────────────────────────────────────────────
# Before / After comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_before_after(
    X: np.ndarray,
    X_proj: np.ndarray,
    labels: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    evr: Optional[np.ndarray] = None,
    figsize: tuple = (13, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side 'before PCA' (original features 1 & 2) and
    'after PCA' (PC1 & PC2) scatter plots.
    """
    fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=figsize)

    plot_original_data(ax_before, X, labels=labels,
                       feature_names=feature_names,
                       title="Before PCA  (raw features 1 & 2)")
    plot_2d_projection(ax_after, X_proj, labels=labels,
                       evr=evr, title="After PCA  (PC1 & PC2)")

    fig.suptitle("Dimensionality Reduction: Before vs After PCA",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

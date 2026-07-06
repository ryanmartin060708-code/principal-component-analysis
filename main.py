"""
main.py — Command-Line Interface for PCA From Scratch
======================================================
Usage examples:

  # Run on a CSV file, keep 2 components
  python main.py --input datasets/sample.csv --components 2

  # Auto-select components that explain 95 % variance
  python main.py --input datasets/sample.csv --variance 0.95

  # Use whitening, save transformed CSV, specify label column
  python main.py --input datasets/iris.csv --components 2 \\
                 --label species --whiten --save-csv out.csv

  # Run built-in synthetic demo (no --input needed)
  python main.py --demo
"""

import argparse
import os
import sys
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src import PCAFromScratch, load_csv, make_synthetic_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA From Scratch — dimensionality reduction in pure NumPy",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Input
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--label", "-l", type=str, default=None,
        help="Name of the label/class column in the CSV (optional).",
    )

    # Component selection (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--components", "-k", type=int, default=None,
        help="Number of principal components to retain.",
    )
    group.add_argument(
        "--variance", "-v", type=float, default=None,
        help="Keep enough components to explain this fraction of variance (0–1).",
    )

    # Options
    parser.add_argument(
        "--whiten", action="store_true",
        help="Apply whitening so every PC has unit variance.",
    )
    parser.add_argument(
        "--save-csv", type=str, default=None,
        help="Save the transformed (projected) data to this CSV path.",
    )
    parser.add_argument(
        "--save-plots", type=str, default=None,
        help="Directory to save generated plots (default: images/).",
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run a built-in synthetic dataset demo (ignores --input).",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip all plot generation.",
    )

    return parser.parse_args()


def run_demo() -> None:
    """Quick smoke-test with a synthetic clustered dataset."""
    print("=" * 55)
    print("PCA From Scratch — Synthetic Dataset Demo")
    print("=" * 55)

    X, labels = make_synthetic_dataset(
        n_samples=300, n_features=6, n_informative=2, n_classes=3, random_state=0
    )
    print(f"Dataset: {X.shape[0]} samples × {X.shape[1]} features, 3 classes")

    pca = PCAFromScratch(n_components=2, whiten=False)
    X_proj = pca.fit_transform(X)
    pca.explained_variance()

    err = pca.reconstruction_error(X)
    print(f"\nReconstruction MSE (k=2): {err:.4f}")

    images_dir = os.path.join(ROOT, "images")
    os.makedirs(images_dir, exist_ok=True)

    fig = pca.plot_full_dashboard(
        X, labels=labels,
        save_path=os.path.join(images_dir, "demo_dashboard.png"),
        title="Synthetic Dataset — PCA Dashboard",
    )
    plt.close(fig)
    print(f"\nSaved → {images_dir}/demo_dashboard.png")
    print("✓ Demo complete.")


def main() -> None:
    args = parse_args()

    if args.demo:
        run_demo()
        return

    if args.input is None:
        print("Error: provide --input <file.csv> or use --demo")
        print("       python main.py --help  for full usage")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: file not found → {args.input}")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"\nLoading: {args.input}")
    X, labels, feature_names = load_csv(args.input, label_col=args.label)
    print(f"Shape  : {X.shape[0]} samples × {X.shape[1]} features")
    if labels is not None:
        print(f"Classes: {sorted(set(labels))}")
    print(f"Features: {feature_names}")

    # ── Resolve n_components ──────────────────────────────────────────────────
    if args.variance is not None:
        n_components = args.variance      # float → threshold mode
    elif args.components is not None:
        n_components = args.components    # int
    else:
        n_components = 2                  # sensible default
        print(f"\nNo --components or --variance given; defaulting to k=2")

    # ── Fit PCA ───────────────────────────────────────────────────────────────
    pca = PCAFromScratch(n_components=n_components, whiten=args.whiten)
    pca.fit(X)

    print()
    pca.explained_variance()

    err = pca.reconstruction_error(X)
    print(f"\nReconstruction MSE : {err:.6f}")

    # ── Save transformed CSV ──────────────────────────────────────────────────
    if args.save_csv:
        pca.save_transformed(X, path=args.save_csv, labels=labels)

    # ── Plots ──────────────────────────────────────────────────────────────────
    if not args.no_plots:
        images_dir = args.save_plots or os.path.join(ROOT, "images")
        os.makedirs(images_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(args.input))[0]

        fig_dash = pca.plot_full_dashboard(
            X,
            labels=labels,
            feature_names=feature_names,
            save_path=os.path.join(images_dir, f"{base}_dashboard.png"),
            title=f"{base} — PCA Dashboard",
        )
        plt.close(fig_dash)

        fig_var = pca.plot_variance(
            save_path=os.path.join(images_dir, f"{base}_variance.png"),
        )
        plt.close(fig_var)

        fig_ev = pca.plot_eigenvectors(
            feature_names=feature_names,
            save_path=os.path.join(images_dir, f"{base}_loadings.png"),
        )
        plt.close(fig_ev)

        print(f"\nPlots saved → {images_dir}/")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()

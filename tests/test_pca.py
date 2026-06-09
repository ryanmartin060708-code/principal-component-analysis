"""
tests/test_pca.py
==================
pytest test suite for PCA From Scratch.

Run from project root:
    pytest tests/ -v

Coverage:
  - Standardisation correctness
  - Covariance matrix shape and symmetry
  - Eigenpair sorting
  - Fit/transform/inverse_transform dimensions
  - Explained variance sums to 1
  - Reconstruction error (exact at k=p, lower at k<p)
  - float n_components (variance threshold)
  - Whiten flag
  - Edge cases (single component, all components, constant column warning)
"""

import warnings
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pca import PCAFromScratch
from src.utils import (
    standardize,
    covariance_matrix,
    sort_eigenpairs,
    reconstruction_error,
    select_n_components_by_variance,
    make_synthetic_dataset,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_X():
    """Deterministic 50×4 dataset for quick tests."""
    rng = np.random.default_rng(0)
    return rng.normal(size=(50, 4))

@pytest.fixture
def iris_like():
    """Synthetic 150×4 Gaussian dataset with 3 classes."""
    X, labels = make_synthetic_dataset(n_samples=150, n_features=4,
                                       n_informative=2, n_classes=3,
                                       random_state=7)
    return X, labels


# ── Standardisation ───────────────────────────────────────────────────────────

class TestStandardize:
    def test_zero_mean(self, small_X):
        X_std, mu, _ = standardize(small_X)
        np.testing.assert_allclose(X_std.mean(axis=0), 0, atol=1e-10)

    def test_unit_std(self, small_X):
        X_std, _, _ = standardize(small_X)
        np.testing.assert_allclose(X_std.std(axis=0, ddof=1), 1, atol=1e-10)

    def test_returns_training_params(self, small_X):
        _, mu, sigma = standardize(small_X)
        assert mu.shape == (small_X.shape[1],)
        assert sigma.shape == (small_X.shape[1],)

    def test_constant_column_warning(self):
        X = np.ones((20, 3))
        X[:, 1] = np.arange(20, dtype=float)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            standardize(X)
            assert any(issubclass(warning.category, UserWarning) for warning in w)


# ── Covariance matrix ─────────────────────────────────────────────────────────

class TestCovarianceMatrix:
    def test_shape(self, small_X):
        X_std, _, _ = standardize(small_X)
        C = covariance_matrix(X_std)
        p = small_X.shape[1]
        assert C.shape == (p, p)

    def test_symmetry(self, small_X):
        X_std, _, _ = standardize(small_X)
        C = covariance_matrix(X_std)
        np.testing.assert_allclose(C, C.T, atol=1e-12)

    def test_diagonal_is_approx_one_for_standardised(self, small_X):
        """After standardisation each feature has variance ≈ 1 (ddof=1)."""
        X_std, _, _ = standardize(small_X)
        C = covariance_matrix(X_std)
        np.testing.assert_allclose(np.diag(C), 1.0, atol=0.15)

    def test_matches_numpy(self, small_X):
        X_std, _, _ = standardize(small_X)
        C_ours = covariance_matrix(X_std)
        C_numpy = np.cov(X_std.T, ddof=1)
        np.testing.assert_allclose(C_ours, C_numpy, atol=1e-10)


# ── Eigenpair sorting ─────────────────────────────────────────────────────────

class TestSortEigenpairs:
    def test_descending_order(self):
        eigenvalues = np.array([1.0, 4.0, 2.0, 0.5])
        eigenvectors = np.eye(4)
        ev_sorted, _ = sort_eigenpairs(eigenvalues, eigenvectors)
        assert list(ev_sorted) == sorted(list(eigenvalues), reverse=True)

    def test_eigenvectors_follow(self):
        eigenvalues = np.array([3.0, 1.0, 2.0])
        # Make eigenvectors distinguishable
        eigenvectors = np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]], dtype=float)
        ev_sorted, vec_sorted = sort_eigenpairs(eigenvalues, eigenvectors)
        # Largest eigenvalue (3.0) was originally at index 0 → still first
        assert ev_sorted[0] == 3.0
        np.testing.assert_array_equal(vec_sorted[:, 0], eigenvectors[:, 0])
        # Second largest (2.0) was at index 2
        assert ev_sorted[1] == 2.0
        np.testing.assert_array_equal(vec_sorted[:, 1], eigenvectors[:, 2])


# ── PCAFromScratch fit / transform ────────────────────────────────────────────

class TestPCAFitTransform:
    def test_fit_transform_shape(self, small_X):
        pca = PCAFromScratch(n_components=2)
        X_proj = pca.fit_transform(small_X)
        assert X_proj.shape == (small_X.shape[0], 2)

    def test_fit_then_transform_shape(self, small_X):
        pca = PCAFromScratch(n_components=3)
        pca.fit(small_X)
        X_proj = pca.transform(small_X)
        assert X_proj.shape == (small_X.shape[0], 3)

    def test_components_shape(self, small_X):
        k = 3
        pca = PCAFromScratch(n_components=k)
        pca.fit(small_X)
        assert pca.components_.shape == (k, small_X.shape[1])

    def test_all_components_when_none(self, small_X):
        pca = PCAFromScratch(n_components=None)
        X_proj = pca.fit_transform(small_X)
        assert X_proj.shape == small_X.shape

    def test_projected_mean_near_zero(self, small_X):
        """Projections should be zero-centred (we standardised first)."""
        pca = PCAFromScratch(n_components=2)
        X_proj = pca.fit_transform(small_X)
        np.testing.assert_allclose(X_proj.mean(axis=0), 0, atol=1e-10)

    def test_error_before_fit(self, small_X):
        pca = PCAFromScratch(n_components=2)
        with pytest.raises(RuntimeError):
            pca.transform(small_X)

    def test_invalid_n_components_too_large(self, small_X):
        pca = PCAFromScratch(n_components=100)
        with pytest.raises(ValueError):
            pca.fit(small_X)

    def test_invalid_input_1d(self):
        pca = PCAFromScratch(n_components=1)
        with pytest.raises(ValueError):
            pca.fit(np.array([1.0, 2.0, 3.0]))


# ── inverse_transform ─────────────────────────────────────────────────────────

class TestInverseTransform:
    def test_shape(self, small_X):
        pca = PCAFromScratch(n_components=2)
        X_rec = pca.fit(small_X).inverse_transform(pca.transform(small_X))
        assert X_rec.shape == small_X.shape

    def test_perfect_reconstruction_at_full_rank(self, small_X):
        """k = p → reconstruction error ≈ 0 (up to floating-point noise)."""
        p = small_X.shape[1]
        pca = PCAFromScratch(n_components=p)
        pca.fit(small_X)
        X_rec = pca.inverse_transform(pca.transform(small_X))
        np.testing.assert_allclose(X_rec, small_X, atol=1e-8)

    def test_reduced_reconstruction_is_worse(self, small_X):
        """Lower k → higher reconstruction error."""
        errors = []
        for k in [1, 2, 3, 4]:
            pca = PCAFromScratch(n_components=k)
            pca.fit(small_X)
            errors.append(pca.reconstruction_error(small_X))
        # Error should be non-increasing as k grows
        assert all(errors[i] >= errors[i+1] - 1e-12 for i in range(len(errors)-1))


# ── Explained variance ────────────────────────────────────────────────────────

class TestExplainedVariance:
    def test_sum_to_one(self, small_X):
        """All-component ratios must sum to 1."""
        pca = PCAFromScratch(n_components=None)
        pca.fit(small_X)
        total = pca.explained_variance_ratio_.sum()
        np.testing.assert_allclose(total, 1.0, atol=1e-10)

    def test_ratios_descending(self, small_X):
        pca = PCAFromScratch(n_components=None)
        pca.fit(small_X)
        evr = pca.explained_variance_ratio_
        assert all(evr[i] >= evr[i+1] - 1e-12 for i in range(len(evr)-1))

    def test_cumulative_monotone(self, small_X):
        pca = PCAFromScratch(n_components=None)
        pca.fit(small_X)
        cum = pca.cumulative_variance_ratio_
        assert all(cum[i] <= cum[i+1] + 1e-12 for i in range(len(cum)-1))

    def test_float_n_components(self, small_X):
        """n_components=0.95 should yield ≥95 % cumulative variance."""
        pca = PCAFromScratch(n_components=0.95)
        pca.fit(small_X)
        assert pca.cumulative_variance_ratio_[-1] >= 0.95 - 1e-10

    def test_summary_dict_keys(self, small_X):
        pca = PCAFromScratch(n_components=2)
        pca.fit(small_X)
        summary = pca.explained_variance(verbose=False)
        assert set(summary.keys()) == {
            "n_components", "eigenvalues",
            "explained_variance_ratio", "cumulative_variance_ratio",
        }


# ── Whitening ─────────────────────────────────────────────────────────────────

class TestWhiten:
    def test_whitened_unit_variance(self, small_X):
        """After whitening, each PC column should have variance ≈ 1."""
        pca = PCAFromScratch(n_components=2, whiten=True)
        X_proj = pca.fit_transform(small_X)
        var = X_proj.var(axis=0, ddof=1)
        np.testing.assert_allclose(var, 1.0, atol=0.2)

    def test_inverse_whitened(self, small_X):
        """inverse_transform should approximately recover original data."""
        p = small_X.shape[1]
        pca = PCAFromScratch(n_components=p, whiten=True)
        pca.fit(small_X)
        X_rec = pca.inverse_transform(pca.transform(small_X))
        np.testing.assert_allclose(X_rec, small_X, atol=1e-6)


# ── Utility: select_n_components_by_variance ──────────────────────────────────

class TestSelectNComponents:
    def test_returns_int(self):
        ev = np.array([3.0, 2.0, 1.0, 0.5])
        k = select_n_components_by_variance(ev, threshold=0.90)
        assert isinstance(k, int)

    def test_sufficient_variance(self):
        ev = np.array([3.0, 2.0, 1.0, 0.5])
        total = ev.sum()
        for thr in [0.5, 0.8, 0.95, 1.0]:
            k = select_n_components_by_variance(ev, threshold=thr)
            cum = np.cumsum(ev[:k]) / total
            assert cum[-1] >= thr - 1e-10

    def test_invalid_threshold(self):
        ev = np.array([3.0, 1.0])
        with pytest.raises(ValueError):
            select_n_components_by_variance(ev, threshold=1.5)


# ── Reconstruction error utility ──────────────────────────────────────────────

class TestReconstructionError:
    def test_zero_for_identical(self, small_X):
        err = reconstruction_error(small_X, small_X)
        assert err == pytest.approx(0.0, abs=1e-12)

    def test_positive_for_different(self, small_X):
        err = reconstruction_error(small_X, small_X + 1.0)
        assert err > 0

    def test_symmetric_perturbation(self, small_X):
        """MSE( X, X+δ ) == MSE( X, X-δ )"""
        delta = np.ones_like(small_X) * 0.5
        err_pos = reconstruction_error(small_X, small_X + delta)
        err_neg = reconstruction_error(small_X, small_X - delta)
        assert err_pos == pytest.approx(err_neg, rel=1e-10)


# ── Synthetic dataset ─────────────────────────────────────────────────────────

class TestMakeSyntheticDataset:
    def test_shape(self):
        X, labels = make_synthetic_dataset(n_samples=90, n_features=5, n_classes=3)
        assert X.shape[0] == 90
        assert X.shape[1] == 5
        assert len(labels) == 90

    def test_class_count(self):
        _, labels = make_synthetic_dataset(n_samples=90, n_classes=3)
        assert len(set(labels)) == 3


# ── repr ──────────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_unfitted(self):
        pca = PCAFromScratch(n_components=2)
        assert "fitted=False" in repr(pca)

    def test_repr_fitted(self, small_X):
        pca = PCAFromScratch(n_components=2)
        pca.fit(small_X)
        assert "fitted=True" in repr(pca)

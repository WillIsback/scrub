"""Tests for the model training pipeline."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.trainer import Trainer, DEFAULT_C_PARAMS


class TestTrainerInit:
    """Trainer construction and configuration."""

    def test_default_params(self):
        t = Trainer()
        assert len(t.c_params) == len(DEFAULT_C_PARAMS)
        assert t.cv_folds == 5
        assert t.class_weight == "balanced"

    def test_custom_params(self):
        t = Trainer(cv_folds=3, c_params=[0.1, 1.0])
        assert t.cv_folds == 3
        assert t.c_params == [0.1, 1.0]


class TestTrainerFeatureMatrix:
    """Feature matrix construction from data rows."""

    def test_builds_feature_matrix(self):
        t = Trainer()
        rows = [
            {"value": "abc123", "line": "x = abc123", "filename": ".env"},
            {"value": "aaaaaa", "line": "y = aaaaaa", "filename": ".py"},
        ]
        X, y = t.build_feature_matrix(rows)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape == (2, 32)  # 2 samples, 32 features
        assert y.shape == (2,)

    def test_label_column(self):
        t = Trainer()
        rows = [
            {"value": "abc", "line": "x", "filename": "", "label": 1},
            {"value": "xyz", "line": "y", "filename": "", "label": 0},
        ]
        X, y = t.build_feature_matrix(rows)
        assert list(y) == [1, 0]

    def test_feature_values_match_expected(self):
        """First feature (entropy) of 'aaaaa' should be 0."""
        t = Trainer()
        rows = [{"value": "aaaaa", "line": "x", "filename": ""}]
        X, _ = t.build_feature_matrix(rows)
        assert X[0, 0] == 0.0  # entropy of uniform string


class TestTrainerCrossValidation:
    """Cross-validation training."""

    def test_cv_returns_results(self):
        t = Trainer(cv_folds=2)
        # Create enough data for 2-fold CV
        rows = [
            {"value": "real_secret_123", "line": "password = 'real_secret_123'", "filename": ".env", "label": 1},
            {"value": "not_a_secret", "line": "x = 'not_a_secret'", "filename": ".py", "label": 0},
            {"value": "another_secret_456", "line": "token = 'another_secret_456'", "filename": ".env", "label": 1},
            {"value": "benign_string", "line": "name = 'benign_string'", "filename": ".txt", "label": 0},
        ]
        results = t.cross_validate(rows)
        assert "cv_scores" in results
        assert "best_C" in results
        assert "coefficients" in results
        assert "intercept" in results
        assert len(results["cv_scores"]) == t.cv_folds

    def test_cv_metrics_contain_expected_keys(self):
        t = Trainer(cv_folds=2)
        rows = [
            {"value": f"secret_{i}", "line": f"x = secret_{i}", "filename": ".env", "label": i % 2}
            for i in range(10)
        ]
        results = t.cross_validate(rows)
        first_fold = results["cv_scores"][0]
        assert "precision" in first_fold
        assert "recall" in first_fold
        assert "f1" in first_fold
        assert "mcc" in first_fold
        assert "accuracy" in first_fold
        assert "log_loss" in first_fold


class TestTrainerExport:
    """Model export to Rust-compatible format."""

    def test_export_contains_coefficients(self):
        t = Trainer(cv_folds=2)
        rows = [
            {"value": f"token_{i}xyz", "line": "auth = 'token_{i}xyz'", "filename": ".env", "label": 1 if i < 5 else 0}
            for i in range(10)
        ]
        results = t.cross_validate(rows)
        exported = t.export_rust(results)
        assert "pub const COEFFS" in exported
        assert "pub const BIAS" in exported
        assert "pub const N_FEATURES" in exported
        assert "pub const FEATURE_NAMES" in exported

    def test_export_has_correct_feature_count(self):
        t = Trainer(cv_folds=2)
        rows = [
            {"value": f"token_{i}xyz", "line": "auth = 'token_{i}xyz'", "filename": ".env", "label": 1 if i < 5 else 0}
            for i in range(10)
        ]
        results = t.cross_validate(rows)
        exported = t.export_rust(results)
        assert "N_FEATURES: usize = 32" in exported

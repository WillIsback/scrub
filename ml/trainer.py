"""Training pipeline for logistic regression credential classifier.

Trains on the issue reports dataset with stratified K-fold CV.
Exports coefficients to Rust source code.
"""

import json
import math
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from ml.features import FEATURE_COUNT, FEATURE_NAMES, compute_all_features

# Default regularization parameters to sweep
DEFAULT_C_PARAMS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


class Trainer:
    """Logistic regression trainer with cross-validation.

    Parameters
    ----------
    cv_folds : int
        Number of cross-validation folds (default 5).
    c_params : list[float]
        Regularization strength values to try (inverse of lambda).
        Smaller = stronger regularization.
    class_weight : str or dict
        Passed to sklearn's LogisticRegression.
    random_state : int
        For reproducibility.
    """

    def __init__(
        self,
        cv_folds: int = 5,
        c_params: list[float] | None = None,
        class_weight: str = "balanced",
        random_state: int = 42,
    ):
        self.cv_folds = cv_folds
        self.c_params = c_params or DEFAULT_C_PARAMS
        self.class_weight = class_weight
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Feature matrix construction
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self, rows: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, y) from a list of dicts with keys:
        value, line, filename[, label].
        """
        n = len(rows)
        X = np.zeros((n, FEATURE_COUNT), dtype=np.float64)
        y = np.zeros(n, dtype=np.int64)

        for i, row in enumerate(rows):
            X[i] = compute_all_features(
                value=row.get("value", ""),
                line=row.get("line", ""),
                filename=row.get("filename", ""),
            )
            y[i] = row.get("label", 0)

        return X, y

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Run stratified K-fold CV over all C params.

        Returns the best model (by mean F1) with per-fold metrics.
        """
        X, y = self.build_feature_matrix(rows)
        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        best_f1 = -1.0
        best_result: dict[str, Any] = {}

        for C in self.c_params:
            fold_scores: list[dict[str, float]] = []
            fold_coeffs: list[np.ndarray] = []

            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = LogisticRegression(
                    C=C,
                    l1_ratio=0,  # L2 penalty
                    solver="lbfgs",
                    class_weight=self.class_weight,
                    max_iter=5000,
                    random_state=self.random_state,
                )
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]

                fold_scores.append({
                    "precision": precision_score(y_val, y_pred, zero_division=0),
                    "recall": recall_score(y_val, y_pred, zero_division=0),
                    "f1": f1_score(y_val, y_pred, zero_division=0),
                    "mcc": matthews_corrcoef(y_val, y_pred),
                    "accuracy": accuracy_score(y_val, y_pred),
                    "log_loss": log_loss(y_val, y_prob),
                })
                fold_coeffs.append(model.coef_.flatten())

            # Average metrics across folds
            mean_f1 = np.mean([s["f1"] for s in fold_scores])
            mean_precision = np.mean([s["precision"] for s in fold_scores])
            mean_recall = np.mean([s["recall"] for s in fold_scores])
            mean_mcc = np.mean([s["mcc"] for s in fold_scores])

            # Train final model on all data for this C
            final_model = LogisticRegression(
                C=C,
                l1_ratio=0,  # L2 penalty
                solver="lbfgs",
                class_weight=self.class_weight,
                max_iter=5000,
                random_state=self.random_state,
            )
            final_model.fit(X, y)

            result = {
                "C": C,
                "mean_f1": mean_f1,
                "mean_precision": mean_precision,
                "mean_recall": mean_recall,
                "mean_mcc": mean_mcc,
                "cv_scores": fold_scores,
                "coefficients": final_model.coef_.flatten().tolist(),
                "intercept": float(final_model.intercept_[0]),
                "fold_coefficients": [c.tolist() for c in fold_coeffs],
            }

            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_result = result
                best_result["best_C"] = C

        # Compute coefficient stability (std across folds)
        if best_result.get("fold_coefficients"):
            coeffs_arr = np.array(best_result["fold_coefficients"])
            best_result["coeff_std"] = coeffs_arr.std(axis=0).tolist()
        else:
            best_result["coeff_std"] = []

        # Loss curve: track training loss across iterations for best model
        best_result["loss_curve"] = self._compute_loss_curve(X, y, best_result)

        # Feature importance (coefficient magnitude)
        coeffs = np.array(best_result["coefficients"])
        best_result["feature_importance"] = sorted(
            zip(FEATURE_NAMES, coeffs.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return best_result

    @staticmethod
    def _compute_loss_curve(
        X: np.ndarray, y: np.ndarray, result: dict[str, Any]
    ) -> list[float]:
        """Re-train and capture loss at each iteration."""
        model = LogisticRegression(
            C=result["C"],
            l1_ratio=0,  # L2 penalty
            solver="saga",  # saga supports warm_start
            class_weight="balanced",
            max_iter=1,
            warm_start=True,
            random_state=42,
        )
        losses: list[float] = []
        n_iterations = 100
        for _ in range(n_iterations):
            model.fit(X, y)
            prob = model.predict_proba(X)[:, 1]
            losses.append(float(log_loss(y, prob)))
            # Check convergence
            if len(losses) > 2 and abs(losses[-1] - losses[-2]) < 1e-6:
                break
        return losses

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_rust(self, result: dict[str, Any]) -> str:
        """Export the best model as Rust source code."""
        coeffs = result["coefficients"]
        bias = result["intercept"]
        n_features = len(coeffs)

        # Format coefficients for Rust
        coeff_lines = []
        for i in range(0, n_features, 4):
            chunk = coeffs[i : i + 4]
            line = ", ".join(f"{c:.10f}" for c in chunk)
            coeff_lines.append(f"    {line},")

        coeffs_str = "\n".join(coeff_lines)

        # Format feature names
        names_str = ",\n".join(f'    "{name}"' for name in FEATURE_NAMES)

        return f"""// Auto-generated by ml/trainer.py — do not edit manually.
// Generated at: {__import__('datetime').datetime.now().isoformat()}
// Best C: {result.get('best_C', result['C'])}
// CV Mean F1: {result['mean_f1']:.4f}
// CV Mean Precision: {result['mean_precision']:.4f}
// CV Mean Recall: {result['mean_recall']:.4f}
// CV Mean MCC: {result['mean_mcc']:.4f}

/// Number of features in the model.
pub const N_FEATURES: usize = {n_features};

/// Logistic regression coefficients (weights), one per feature.
pub const COEFFS: [f64; N_FEATURES] = [
{coeffs_str}
];

/// Logistic regression bias (intercept).
pub const BIAS: f64 = {bias:.10f};

/// Names of each feature (in order, matches COEFFS).
pub const FEATURE_NAMES: [&str; N_FEATURES] = [
{names_str}
];
"""

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    @staticmethod
    def predict_proba(features: np.ndarray, coeffs: list[float], bias: float) -> float:
        """Sigmoid(w·x + b)."""
        logit = np.dot(features, coeffs) + bias
        return 1.0 / (1.0 + math.exp(-logit))

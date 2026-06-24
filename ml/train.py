#!/usr/bin/env python3
"""Train the logistic regression credential classifier.

Usage:
    python ml/train.py                          # train on issue reports dataset
    python ml/train.py --export                 # train + export Rust source
    python ml/train.py --cv-folds 5             # custom folds
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.trainer import Trainer

DATA_DIR = Path("ml/data/Secret-Leak-Detection-Issue-Report/Data")
OUTPUT_DIR = Path("ml/output")


def load_data(split: str) -> list[dict]:
    """Load a CSV split and convert to row dicts."""
    path = DATA_DIR / f"{split}.csv"
    print(f"  Loading {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)...")
    df = pd.read_csv(path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "value": str(row.get("candidate_string", "")),
            "line": str(row.get("text", "")),
            "filename": "",
            "label": int(row.get("label", 0)),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train credential classifier")
    parser.add_argument("--cv-folds", type=int, default=5, help="CV folds")
    parser.add_argument("--export", action="store_true", help="Export Rust source")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading training data...")
    train_rows = load_data("train")
    val_rows = load_data("val")

    # Combine train + val for CV (test is held out)
    all_train = train_rows + val_rows
    n_pos = sum(1 for r in all_train if r["label"] == 1)
    n_neg = len(all_train) - n_pos
    print(f"\nTraining set: {len(all_train)} samples ({n_pos} positive, {n_neg} negative)")
    print(f"  Positive ratio: {n_pos / len(all_train) * 100:.1f}%")
    print(f"  Class weight: balanced (neg:{n_neg / n_pos:.1f} : pos:1)")

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------
    print(f"\nRunning {args.cv_folds}-fold stratified CV...")
    t0 = time.time()
    trainer = Trainer(cv_folds=args.cv_folds)
    results = trainer.cross_validate(all_train)
    elapsed = time.time() - t0

    print(f"\n  Best C: {results['best_C']}")
    print(f"  Mean F1:       {results['mean_f1']:.4f}")
    print(f"  Mean Precision: {results['mean_precision']:.4f}")
    print(f"  Mean Recall:    {results['mean_recall']:.4f}")
    print(f"  Mean MCC:       {results['mean_mcc']:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Per-fold breakdown
    print("\n  Per-fold scores:")
    for i, s in enumerate(results["cv_scores"]):
        print(f"    Fold {i+1}: P={s['precision']:.3f} R={s['recall']:.3f} "
              f"F1={s['f1']:.3f} MCC={s['mcc']:.3f} loss={s['log_loss']:.3f}")

    # Feature importance
    print("\n  Top 10 features by |coefficient|:")
    for name, coeff in results["feature_importance"][:10]:
        print(f"    {name:30s} {coeff:+.4f}")

    # Loss curve
    loss = results["loss_curve"]
    print(f"\n  Loss curve: {len(loss)} iterations, final loss={loss[-1]:.4f}")

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    print("\nEvaluating on test set...")
    test_rows = load_data("test")
    X_test, y_test = trainer.build_feature_matrix(test_rows)

    # Re-train on all training data with best C
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss as sk_log_loss,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    X_all, y_all = trainer.build_feature_matrix(all_train)
    final_model = LogisticRegression(
        C=results["best_C"],
        l1_ratio=0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    final_model.fit(X_all, y_all)

    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "accuracy": accuracy_score(y_test, y_pred),
        "log_loss": sk_log_loss(y_test, y_prob),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "n_test": len(y_test),
        "n_positive": int(y_test.sum()),
    }
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  MCC:       {test_metrics['mcc']:.4f}")
    print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "best_C": results["best_C"],
        "cv": {
            "mean_f1": results["mean_f1"],
            "mean_precision": results["mean_precision"],
            "mean_recall": results["mean_recall"],
            "mean_mcc": results["mean_mcc"],
            "per_fold": results["cv_scores"],
        },
        "test": test_metrics,
        "coefficients": results["coefficients"],
        "intercept": results["intercept"],
        "feature_importance": [
            {"name": n, "coeff": c}
            for n, c in results["feature_importance"]
        ],
        "coeff_std": results["coeff_std"],
        "loss_curve": results["loss_curve"],
    }

    output_path = OUTPUT_DIR / "training_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ------------------------------------------------------------------
    # Export Rust source
    # ------------------------------------------------------------------
    if args.export:
        rust_code = trainer.export_rust(results)
        rust_path = OUTPUT_DIR / "model.rs"
        with open(rust_path, "w") as f:
            f.write(rust_code)
        print(f"Rust source exported to {rust_path}")
        print(f"  N_FEATURES: 24")
        print(f"  COEFFS size: {len(results['coefficients'])} floats")
        print(f"  File size: {rust_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()

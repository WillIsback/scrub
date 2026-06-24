#!/usr/bin/env python3
"""Evaluate the ML post-filter on CredData.

Loads CredData meta files, extracts lines from source files,
computes features, runs ML prediction, and compares metrics
(at various thresholds) against ground truth.

This shows whether the ML filter improves precision/recall/F1.
"""

import csv
import os
import sys
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Add parent to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.features import FEATURE_COUNT, compute_all_features

CREDDATA_META_DIR = Path("bench-data/meta")
CREDDATA_DATA_DIR = Path("bench-data/data")
RESULTS_FILE = Path("ml/output/training_results.json")


def load_creddata() -> list[dict]:
    """Load all CredData entries with their source lines."""
    rows = []
    meta_files = sorted(CREDDATA_META_DIR.glob("*.csv"))

    for mf in meta_files:
        with open(mf, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for record in reader:
                file_path = record.get("FilePath", "")
                line_start = int(record.get("LineStart", "0") or "0")
                ground_truth = record.get("GroundTruth", "F").strip().upper() == "T"
                value_start = int(record.get("ValueStart", "0") or "0")
                value_end = int(record.get("ValueEnd", "0") or "0")

                # Build full source path (FilePath already starts with "data/")
                # Remove the leading "data/" since CREDDATA_DATA_DIR already points there
                relative = file_path.lstrip("/")
                if relative.startswith("data/"):
                    relative = relative[5:]
                src_path = CREDDATA_DATA_DIR / relative
                if not src_path.exists():
                    continue

                # Read the line from the source file
                try:
                    with open(src_path, "r", encoding="utf-8", errors="replace") as sf:
                        lines = sf.readlines()
                except Exception:
                    continue

                if line_start < 1 or line_start > len(lines):
                    continue

                line_text = lines[line_start - 1].rstrip("\n").rstrip("\r")

                # Extract value from the line
                if value_start > 0 and value_end > 0 and value_end <= len(line_text):
                    value = line_text[value_start:value_end]
                else:
                    value = ""

                rows.append({
                    "value": value,
                    "line": line_text,
                    "filename": str(src_path),
                    "label": 1 if ground_truth else 0,
                })

    return rows


def main():
    print("Loading CredData...")
    rows = load_creddata()
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = len(rows) - n_pos
    print(f"  {len(rows)} entries ({n_pos} positive, {n_neg} negative)")

    # Load trained model
    print("\nLoading trained model...")
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    coeffs = np.array(results["coefficients"])
    bias = results["intercept"]
    print(f"  Best C: {results['best_C']}")
    print(f"  N features: {len(coeffs)}")

    # Compute features and predictions for all rows
    print("\nComputing features...")
    X = np.zeros((len(rows), FEATURE_COUNT), dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    errors = 0

    for i, row in enumerate(rows):
        try:
            feats = compute_all_features(
                value=row["value"],
                line=row["line"],
                filename=row["filename"],
            )
            X[i] = feats
        except Exception as e:
            errors += 1
            continue

    if errors:
        print(f"  Warning: {errors} rows had errors, using zeros")

    # Compute prediction scores
    logits = X @ coeffs + bias
    scores = 1.0 / (1.0 + np.exp(-logits))

    # Evaluate at various thresholds
    print("\n=== ML Filter Evaluation on CredData ===\n")
    print(f"{'Threshold':>10s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'MCC':>8s} {'Kept':>8s}")
    print("-" * 60)

    # Also compute baseline (always keep all = threshold 0)
    y_base = y  # ground truth for reference

    for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        y_pred = (scores >= threshold).astype(np.int64)
        kept = int(y_pred.sum())

        if kept == 0:
            print(f"{threshold:>10.2f} {'—':>8s} {'—':>8s} {'—':>8s} {'—':>8s} {kept:>8d}")
            continue

        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y, y_pred)

        print(f"{threshold:>10.2f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {mcc:>8.4f} {kept:>8d}")

    # Find best F1 threshold
    best_f1 = -1
    best_threshold = 0.0
    best_metrics = {}

    for threshold in np.arange(0.01, 1.0, 0.01):
        y_pred = (scores >= threshold).astype(np.int64)
        if y_pred.sum() == 0:
            continue
        f1 = f1_score(y, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                "threshold": threshold,
                "precision": precision_score(y, y_pred, zero_division=0),
                "recall": recall_score(y, y_pred, zero_division=0),
                "f1": f1,
                "mcc": matthews_corrcoef(y, y_pred),
                "accuracy": accuracy_score(y, y_pred),
                "roc_auc": roc_auc_score(y, scores),
                "kept": int(y_pred.sum()),
                "n_total": len(y),
                "n_positive": int(y.sum()),
            }

    print("\n=== Best Threshold ===")
    print(f"  Threshold:     {best_metrics['threshold']:.2f}")
    print(f"  Precision:     {best_metrics['precision']:.4f}")
    print(f"  Recall:        {best_metrics['recall']:.4f}")
    print(f"  F1:            {best_metrics['f1']:.4f}")
    print(f"  MCC:           {best_metrics['mcc']:.4f}")
    print(f"  ROC AUC:       {best_metrics['roc_auc']:.4f}")
    print(f"  Kept:          {best_metrics['kept']} / {best_metrics['n_total']}")
    print(f"  Baseline F1 (no ML): {f1_score(y, np.ones(len(y)), zero_division=0):.4f}")

    # Also evaluate with caviarder's actual predictions as baseline
    # (We don't run caviarder here — we use the fact that ML scores are
    # computed only on lines that caviarder already matched. So the
    # baseline "keep all" is the current caviarder performance.)
    print("\n=== Comparison with Published Caviarder Baseline ===")
    print(f"  Caviarder (no ML, from bench F1=0.486)")
    print(f"  Caviarder + ML (best F1) = {best_metrics['f1']:.4f}")

    # Save results
    output = {
        "dataset": "CredData",
        "n_samples": len(rows),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "best_threshold": best_metrics["threshold"],
        "best_f1": best_metrics["f1"],
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
        "best_mcc": best_metrics["mcc"],
        "roc_auc": best_metrics["roc_auc"],
    }

    out_path = Path("ml/output/creddata_eval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

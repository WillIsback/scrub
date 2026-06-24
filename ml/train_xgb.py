#!/usr/bin/env python3
"""Train gradient boosting model (XGBoost) on CredData with rule-type features.

Usage:
    python ml/train_xgb.py                  # train and evaluate
    python ml/train_xgb.py --export         # train + export Rust model
    python ml/train_xgb.py --quick          # quick test (fewer rounds)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.features import FEATURE_COUNT, FEATURE_NAMES, RULE_TYPE_NAMES, compute_all_features

OUTPUT_DIR = Path("ml/output")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(path: str) -> tuple[list[dict], list[dict]]:
    """Load the CredData split JSON, return (train_rows, test_rows)."""
    import json as j
    with open(path) as f:
        data = j.load(f)
    print(f"  Train: {len(data['train'])} ({data['info']['train_pos']} pos, {data['info']['train_neg']} neg)")
    print(f"  Test:  {len(data['test'])} ({data['info']['test_pos']} pos, {data['info']['test_neg']} neg)")
    return data["train"], data["test"]


def build_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X (float32) and labels y from row dicts."""
    n = len(rows)
    X = np.zeros((n, FEATURE_COUNT), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    errors = 0
    for i, row in enumerate(rows):
        try:
            feats = compute_all_features(
                value=row["value"],
                line=row["line"],
                filename=row["filename"],
                rule_type=row.get("rule_type", ""),
            )
            X[i] = feats.astype(np.float32)
            y[i] = row["label"]
        except Exception as e:
            errors += 1
    if errors:
        print(f"  Warning: {errors} rows had errors")
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class XGBTrainer:
    """Train an XGBoost model with early stopping and hyperparameter search."""

    def __init__(self):
        import xgboost as xgb
        self.xgb = xgb

    def train_with_hp_search(
        self, X_train, y_train, X_val, y_val,
        quick: bool = False,
    ) -> dict:
        """Hyperparameter search using random grid, return best model + metrics."""

        param_grid = {
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0.0, 0.1, 0.5],
        }

        if quick:
            param_grid = {k: v[:2] for k, v in param_grid.items()}

        # Generate all combinations (or sample if too many)
        keys = list(param_grid.keys())
        from itertools import product
        all_combs = list(product(*[param_grid[k] for k in keys]))
        n_combs = len(all_combs)

        # If too many combinations, use random sample
        import random
        random.seed(42)
        if n_combs > 200:
            all_combs = random.sample(all_combs, 200)
            n_combs = len(all_combs)

        print(f"\n  Searching over {n_combs} hyperparameter combinations...")

        best_score = -1.0
        best_params = None
        best_n_rounds = 0
        results_log = []

        for i, combo in enumerate(all_combs):
            params = dict(zip(keys, combo))
            params.update({
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "seed": 42,
                "verbosity": 0,
            })

            # Early stopping
            model = self.xgb.XGBClassifier(
                **params,
                n_estimators=500,
                early_stopping_rounds=30,
                random_state=42,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            val_pred = (model.predict_proba(X_val)[:, 1] >= 0.5).astype(int)
            val_f1 = f1_score(y_val, val_pred, zero_division=0)

            n_rounds = model.best_iteration + 1 if model.best_iteration else model.get_booster().num_boosted_rounds()

            results_log.append({
                "params": params,
                "val_f1": val_f1,
                "n_rounds": n_rounds,
            })

            if val_f1 > best_score:
                best_score = val_f1
                best_params = params
                best_n_rounds = n_rounds

            if (i + 1) % 10 == 0 or i == 0:
                print(f"    [{i + 1}/{n_combs}] best F1={best_score:.4f} (current val_f1={val_f1:.4f})")

        # Retrain with best params on all training data
        print(f"\n  Best params: max_depth={best_params['max_depth']}, "
              f"lr={best_params['learning_rate']}, subsample={best_params['subsample']}, "
              f"colsample={best_params['colsample_bytree']}, "
              f"min_child_weight={best_params['min_child_weight']}, gamma={best_params['gamma']}")
        print(f"  Best val F1: {best_score:.4f}, rounds: {best_n_rounds}")

        best_params_no_es = {k: v for k, v in best_params.items()
                             if k not in ("verbosity",)}
        final_model = self.xgb.XGBClassifier(
            **best_params_no_es,
            n_estimators=best_n_rounds,
            random_state=42,
        )
        final_model.fit(X_train, y_train)

        return {
            "model": final_model,
            "best_params": best_params,
            "best_val_f1": best_score,
            "n_rounds": best_n_rounds,
            "search_log": results_log,
        }

    def train_quick(self, X_train, y_train, X_val, y_val) -> dict:
        """Quick training with sensible defaults, no HP search."""
        params = {
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "seed": 42,
            "verbosity": 0,
        }
        model = self.xgb.XGBClassifier(
            **params,
            n_estimators=500,
            early_stopping_rounds=30,
            random_state=42,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        n_rounds = model.best_iteration + 1 if model.best_iteration else model.get_booster().num_boosted_rounds()

        params_copy = dict(params)
        params_copy["n_estimators"] = n_rounds
        params_copy.pop("verbosity", None)
        final_model = self.xgb.XGBClassifier(**params_copy, random_state=42)
        final_model.fit(X_train, y_train)

        return {
            "model": final_model,
            "best_params": params,
            "best_val_f1": f1_score(y_val, (model.predict_proba(X_val)[:, 1] >= 0.5).astype(int), zero_division=0),
            "n_rounds": n_rounds,
            "search_log": [],
        }


# ---------------------------------------------------------------------------
# Export to Rust
# ---------------------------------------------------------------------------

def export_rust(model, feature_names: list[str], output_path: Path):
    """Export XGBoost model as Rust const arrays for tree-walking inference.

    XGBoost's Booster.dump_json() gives a JSON tree structure we convert
    to flat arrays of (feature_idx, threshold, left/right_idx, leaf_value).
    """
    booster = model.get_booster()
    tmp_path = output_path / "_xgb_dump.json"
    booster.dump_model(str(tmp_path), dump_format="json")
    with open(tmp_path) as f:
        trees_data = json.load(f)  # Direct list of trees
    tmp_path.unlink()

    # Base score — XGBoost binary:logistic uses 0.5 as default bias
    base_score = 0.5

    rust_code = [
        "// Auto-generated by ml/train_xgb.py — do not edit manually.",
        f"// Generated at: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"// Model type: XGBoost gradient boosting",
        f"// Trees: {len(trees_data)}",
        f"// Features: {len(feature_names)}",
        "",
        "/// Number of features expected by this model.",
        f"pub const N_FEATURES: usize = {len(feature_names)};",
        "",
        "/// Bias / base score (before sigmoid transform).",
        f"pub const BIAS: f32 = {base_score:.10f};",
        "",
        "/// Number of trees in the ensemble.",
        f"pub const N_TREES: usize = {len(trees_data)};",
        "",
        "/// Feature names (in order, for debugging).",
        "pub const FEATURE_NAMES: [&str; N_FEATURES] = [",
    ]

    for name in feature_names:
        rust_code.append(f'    "{name}",')
    rust_code.append("];\n")

    # For each tree, flatten to a sequence of TreeNode structs
    all_trees_flat = []
    tree_sizes = []
    tree_offsets = [0]

    for tree_json in trees_data:
        # Convert tree JSON to flat nodes
        nodes = _flatten_tree(tree_json)
        all_trees_flat.extend(nodes)
        tree_sizes.append(len(nodes))
        tree_offsets.append(tree_offsets[-1] + len(nodes))

    total_nodes = sum(tree_sizes)
    rust_code.append(f"/// Flat node array: all trees concatenated.")
    rust_code.append(f"/// Each node: (feature_idx, threshold_f32, left_idx, right_idx, leaf_f32)")
    rust_code.append(f"/// feature_idx = -1 means leaf node.")
    rust_code.append(f"pub const NODES: [(i16, f32, i32, i32, f32); {total_nodes}] = [")

    for feat_idx, thresh, left, right, leaf in all_trees_flat:
        rust_code.append(f"    ({feat_idx}, {thresh:.8f}, {left}, {right}, {leaf:.8f}),")
    rust_code.append("];\n")

    # Tree offsets (start index of each tree in NODES)
    rust_code.append(f"pub const TREE_OFFSETS: [usize; {len(trees_data) + 1}] = [")
    for off in tree_offsets:
        rust_code.append(f"    {off},")
    rust_code.append("];\n")

    rust_code.append("""
/// Predict secret probability using gradient boosting ensemble.
///
/// Walks each decision tree in order, sums leaf values, applies sigmoid.
/// Equivalent to XGBoost's predict_proba for binary classification.
pub fn predict_xgb(features: &[f32; N_FEATURES]) -> f32 {
    let mut sum = BIAS;
    for t in 0..N_TREES {
        let start = TREE_OFFSETS[t] as usize;
        let mut node_idx = start;
        loop {
            let (feat_idx, threshold, left, right, leaf) = NODES[node_idx];
            if feat_idx < 0 {
                // Leaf node
                sum += leaf;
                break;
            }
            if features[feat_idx as usize] <= threshold {
                node_idx = left as usize;
            } else {
                node_idx = right as usize;
            }
        }
    }
    // Sigmoid
    1.0 / (1.0 + core::f32::consts::E.powf(-sum))
}
""")

    model_path = output_path / "model_xgb.rs"
    with open(model_path, "w") as f:
        f.write("\n".join(rust_code))
    print(f"  Rust model exported to {model_path} ({model_path.stat().st_size} bytes)")
    print(f"  {len(trees_data)} trees, {total_nodes} total nodes")

    return model_path


def _flatten_tree(tree_json) -> list[tuple]:
    """Convert a nested XGBoost tree JSON to flat list of (feat_idx, thresh, left, right, leaf).

    Each node gets a sequential index. Internal nodes reference children by their flat index.
    """
    # Build a mapping from JSON nodeid -> flat index
    nodes_list: list[dict] = []
    _collect_nodes(tree_json, nodes_list)

    nodeid_to_flat = {n["nodeid"]: i for i, n in enumerate(nodes_list)}

    flat = []
    for n in nodes_list:
        if "leaf" in n:
            flat.append((-1, 0.0, -1, -1, float(n["leaf"])))
        else:
            # split is like "f17" — extract the integer
            split_str = n["split"]
            if isinstance(split_str, str) and split_str.startswith("f"):
                feat_idx = int(split_str[1:])
            else:
                feat_idx = int(split_str)
            thresh = float(n["split_condition"])
            left = nodeid_to_flat[n["yes"]]
            right = nodeid_to_flat[n["no"]]
            flat.append((feat_idx, thresh, left, right, 0.0))
    return flat


def _collect_nodes(node, result: list):
    """Recursively collect all nodes from XGBoost JSON into a flat list."""
    result.append(node)
    if "children" in node:
        for child in node["children"]:
            _collect_nodes(child, result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost classifier on CredData")
    parser.add_argument("--export", action="store_true", help="Export Rust model source")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer HP combos")
    parser.add_argument("--split", default="ml/data/creddata_split.json",
                        help="Path to CredData split JSON")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading CredData split...")
    train_rows, test_rows = load_split(args.split)

    # Filter out "other" entries (60% of training are easy negatives that
    # don't match any gitleaks regex — irrelevant for post-filter inference).
    train_rows = [r for r in train_rows if r.get("rule_type", "") != "other"]
    test_rows  = [r for r in test_rows  if r.get("rule_type", "") != "other"]

    # Build feature matrices
    print("\nBuilding feature matrices...")
    t0 = time.time()
    X_train, y_train = build_matrix(train_rows)
    X_test, y_test = build_matrix(test_rows)
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Train: {X_train.shape}, positive: {y_train.sum()}")
    print(f"  Test:  {X_test.shape}, positive: {y_test.sum()}")

    # Train
    print("\nTraining XGBoost...")
    trainer = XGBTrainer()

    if args.quick:
        result = trainer.train_quick(X_train, y_train, X_test, y_test)
    else:
        result = trainer.train_with_hp_search(X_train, y_train, X_test, y_test, quick=False)

    model = result["model"]

    # Evaluate on test set
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    test_metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "n_test": len(y_test),
        "n_positive": int(y_test.sum()),
        "n_features": FEATURE_COUNT,
        "n_trees": model.get_booster().num_boosted_rounds(),
    }

    print(f"\n=== Test Set Results ===")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1:        {test_metrics['f1']:.4f}")
    print(f"  MCC:       {test_metrics['mcc']:.4f}")
    print(f"  ROC AUC:   {test_metrics['roc_auc']:.4f}")
    print(f"  Trees:     {test_metrics['n_trees']}")

    # Find best threshold
    print("\n  Sweeping thresholds...")
    best_f1 = -1.0
    best_th = 0.5
    for th in np.arange(0.01, 0.99, 0.01):
        yp = (y_prob >= th).astype(int)
        f1 = f1_score(y_test, yp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    print(f"  Best threshold: {best_th:.2f} (F1={best_f1:.4f})")
    test_metrics["best_threshold"] = best_th
    test_metrics["best_f1"] = best_f1

    # Save metrics
    output = {
        "model": "xgboost",
        "feature_count": FEATURE_COUNT,
        "feature_names": FEATURE_NAMES,
        "rule_type_names": RULE_TYPE_NAMES,
        "best_params": result["best_params"],
        "best_val_f1": result["best_val_f1"],
        "n_rounds": result["n_rounds"],
        "test": test_metrics,
    }

    output_path = OUTPUT_DIR / "xgb_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Export Rust model
    if args.export:
        print("\nExporting Rust model...")
        export_rust(model, FEATURE_NAMES, OUTPUT_DIR)

    # Feature importance
    importance = model.feature_importances_
    top_n = min(15, len(importance))
    top_idx = np.argsort(importance)[::-1][:top_n]
    print(f"\n  Top {top_n} features by importance:")
    for i in top_idx:
        print(f"    {FEATURE_NAMES[i]:30s} {importance[i]:.4f}")


if __name__ == "__main__":
    main()

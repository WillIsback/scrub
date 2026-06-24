# ML Post-Filter for caviarder

**Date:** 2026-06-24
**Status:** Approved

## Overview

Add an optional logistic regression post-filter to caviarder. After regex matching, each candidate line is scored by a tiny ML model. If the score falls below a threshold, the alert is suppressed — reducing false positives while preserving real detections.

## Motivation

caviarder currently achieves 55.9% precision / 43.0% recall (F1 0.486) on CredData metadata. The main source of false positives is regex patterns that match credential-like strings (UUIDs, hashes, hex strings, random tokens) that aren't actual credentials. A lightweight ML model can learn to distinguish "looks like a credential" from "looks like a real secret" using structural features.

## Design

### Architecture

```
Line input
     │
     ▼
┌─────────────────┐  match?  ┌──────────────────┐  ≥threshold  ┌──────────┐
│  7 regex rules  │ ────────→│  Feature Extract  │ ───────────→│  Report   │
│  (caviarder)    │          │  (~25 features)   │             │   alert   │
└─────────────────┘          └────────┬──────────┘             └──────────┘
                                      │
                                      ▼
                              ┌──────────────────┐  <threshold
                              │  Logistic Reg    │ ───────────→  ❌ Suppress
                              │  sigmoid(w·x+b)  │
                              └──────────────────┘
```

### Components

1. **Feature Extractor** — computes ~25 numerical features from a matched line
2. **Logistic Regression** — `sigmoid(dot(weights, features) + bias)`, nanosecond inference
3. **Threshold Gate** — configurable threshold (CLI flag), default = disabled (pure regex)

### Feature Set (~25 features)

All features are computed from the matched value and its surrounding line context.

**Value-level features:**
- `entropy` — Shannon entropy of the matched value
- `log_len` — log(length of value)
- `digit_ratio` — proportion of digits [0-9]
- `upper_ratio` — proportion of uppercase letters [A-Z]
- `lower_ratio` — proportion of lowercase letters [a-z]
- `special_ratio` — proportion of non-alphanumeric characters
- `max_consecutive_repeat` — longest run of identical characters
- `has_padding` — ends with `=` or `==` (base64 padding)
- `has_colon`, `has_slash`, `has_dot`, `has_hyphen`, `has_underscore`, `has_equals` — binary flags
- `is_hex_only` — only contains [0-9a-fA-F]

**Line-level features:**
- `line_log_len` — log(length of full line)
- `keyword_count` — count of credential-related keywords (password, token, key, secret, etc.)
- `is_assignment` — binary, contains `=` (assignment context)
- `has_quotes` — binary, value is in quotes
- `has_function_call` — binary, value inside parentheses
- `has_comment` — binary, line contains comment marker

**File-level features:**
- `is_source_file` — binary (.py, .js, .rs, etc.)
- `is_config_file` — binary (.env, .cfg, .ini, .json, .yaml, .xml)

**Rule-level features:**
- Rule type — one-hot encoded (7 categories → 6 binary features, 1 baseline)

### Training Pipeline (Python, scikit-learn)

**Data:** Issue reports dataset (Zenodo 10.5281/zenodo.17430335) — 54,148 instances, 5,881 true secrets. Labeled instances from GitHub issues. No overlap with CredData (our evaluation dataset).

**Methodology:**
- 5-fold stratified cross-validation (preserve class distribution)
- Class weighting: `class_weight='balanced'` (inverse frequency weighting)
- Regularization sweep: L2 penalty, C ∈ {0.001, 0.01, 0.1, 1, 10}
- Tracking: log-loss per fold, precision/recall/F1/MCC per fold, ROC curve
- Overfitting check: training vs validation loss curves, coefficient stability across folds

**Output:** Coefficients exported as a Rust `const` array + bias term.

### Inference (Rust)

```rust
fn predict_proba(features: &[f64; N_FEATURES]) -> f64 {
    let logit: f64 = COEFFS.iter()
        .zip(features.iter())
        .map(|(w, x)| w * x)
        .sum::<f64>() + BIAS;
    1.0 / (1.0 + (-logit).exp())
}
```

- No external dependencies
- Model size: ~30 floats × 8 bytes = ~240 bytes (+ bias)
- Inference time: nanoseconds on CPU
- Feature computation: lightweight string operations, no allocations beyond feature array

### Integration

- **CLI flag:** `--ml-threshold <float>` (default: 0.0 = disabled)
- **Feature gate:** model coefficients compiled in via `include_bytes!` or `const` array
- **Backward compatible:** existing configs produce identical output (disabled by default)

### Evaluation

| Metric | Expected |
|--------|----------|
| Test set | CredData (67K lines, unseen) |
| Baseline | caviarder pure regex: 55.9% precision, 43.0% recall, 0.486 F1 |
| Target | ↑ precision (fewer FPs), maintain or ↑ recall |

## Constraints

- No GPU memory — model runs on CPU, ~200 bytes
- No new Rust dependencies for inference
- Training uses separate dataset from evaluation (no data leakage)
- All training code in Python, in a dedicated directory (`ml/`)
- Model must be reproducible (fixed seed, tracked config)

## Out of Scope

- Deep learning models (BERT, LSTM, transformers)
- Real-time training or online learning
- Multi-class classification (binary only: "keep" or "suppress")
- Automatic threshold tuning

## Files

| Path | Purpose |
|------|---------|
| `ml/` | Python training pipeline |
| `ml/train.py` | Main training script |
| `ml/features.py` | Feature extraction functions |
| `ml/model.py` | Model definition and training logic |
| `ml/export.py` | Export coefficients to Rust source |
| `ml/requirements.txt` | Python dependencies |
| `ml/tests/` | Tests for feature extraction |
| `src/ml.rs` | Rust inference module |
| `src/lib.rs` | Integration point (optional gate) |

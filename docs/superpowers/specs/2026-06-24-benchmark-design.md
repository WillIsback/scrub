# caviarder — Benchmark & Confusion Matrix Design

**Date**: 2026-06-24
**Status**: Approved design

## Overview

Add a benchmark suite to caviarder that measures both **performance** (throughput, per-rule cost)
and **accuracy** (precision, recall, F1 via confusion matrix) against a labeled dataset of real-world
source code secrets.

## Dataset: Samsung/CredData

**[CredData](https://github.com/Samsung/CredData)** by Samsung is an open-source (Apache-2.0) dataset
of credentials extracted from 297 public GitHub repositories.

| Metric | Value |
|--------|-------|
| License | Apache-2.0 — no agreement required |
| Labeled lines | 73 842 (4 583 True, 69 259 False) |
| Total LOC scanned | ~19 million |
| Languages | 49+ (Text, Go, JS, Python, Java, Ruby, YAML, etc.) |
| Categories | API Key, Token, Password, Private Key, Generic Secret, etc. |
| Obfuscation | Real secrets replaced with synthetic values (same regex pattern, same length) |
| Baseline (gitleaks) | Precision 52.5%, Recall 24.4%, F1 0.33 |

The dataset is hosted on GitHub and downloaded via a provided `download_data.py` script.
Credentials are obfuscated so no real secrets are exposed — the synthetic values match the
same regex patterns and entropy characteristics as the originals.

### Setup

```bash
git clone https://github.com/Samsung/CredData.git _creddata
cd _creddata
python download_data.py   # generates data/ and meta/ directories
```

A script `scripts/setup-bench-data.sh` will automate this and place the dataset
under `bench-data/` (gitignored).

## Benchmark Structure

### Files

```
benches/
  throughput.rs         # Criterion benchmark — raw throughput
  per_rule.rs           # Criterion benchmark — per-rule timing
confusion/
  main.rs               # Binary — confusion matrix against CredData
scripts/
  setup-bench-data.sh   # Download & prepare CredData
```

### Cargo.toml additions

```toml
[dev-dependencies]
criterion = { version = "4", features = ["html_reports"] }

[[bench]]
name = "throughput"
harness = false

[[bench]]
name = "per_rule"
harness = false

[[bin]]
name = "cav-bench-confusion"
path = "confusion/main.rs"
```

### .gitignore additions

```
bench-data/
_creddata/
```

## Benchmark Details

### Throughput (`cargo bench --bench throughput`)

Measures how fast caviarder can process raw text.

- **Input**: Generate a large text file (10 MB) containing a mix of clean lines and embedded secrets (API keys, tokens, passwords)
- **What it measures**: MB/s processed, time to first result, time to completion
- **Variables**: File size (1 MB, 10 MB, 50 MB), secret density (1%, 5%, 10%)
- **Criterion config**: Measure over 10+ samples, warm-up 3 s, measure 10 s

### Per-Rule (`cargo bench --bench per_rule`)

Measures the cost of each gitleaks rule individually.

- **Input**: For each of the 220+ rules, a text file containing only matches of that rule
- **What it measures**: μs per rule, μs per match
- **Output**: Ranking of slowest rules, total time for all rules
- **Criterion config**: Same as throughput

### Confusion Matrix (`cargo run --bin cav-bench-confusion`)

A standalone binary that loads the CredData dataset and compares caviarder's decisions against ground truth.

### Metadata format

Each `meta/*.csv` file contains rows with the following columns (relevant subset):

| Column | Description |
|--------|-------------|
| `FilePath` | Path to the source file under `bench-data/data/` |
| `LineStart` | Line number (1-indexed) |
| `LineEnd` | End line (same as start for single-line) |
| `GroundTruth` | `T` (True), `F` (False), or `X` (Unknown) — `F` and `X` both treated as False |
| `Category` | Credential category (e.g. `Secret:Token`) |

### Detection logic

1. Load metadata CSV from `bench-data/meta/*.csv`
2. For each entry, read the source line from `bench-data/data/` at row `LineStart`
3. Extract the raw line text
4. Run `Redactor::redact()` on the line
5. Compare original line vs redacted output:
   - **If different** (something was redacted) → caviarder **detected** something
   - **If identical** (nothing changed) → caviarder **did not detect** anything
6. Cross with ground truth label to classify TP / FP / FN / TN

| Prediction \ Truth | True (T) | False (F/X) |
|-------------------|----------|-------------|
| Redacted | TP | FP |
| Not redacted | FN | TN |

**Output:**

```
=== Confusion Matrix (CredData) ===
# Instances:  73 842
# True:       4 583
# False:     69 259

Metrics:
  Precision:   xx.x%
  Recall:      xx.x%
  F1:          xx.xxx
  Accuracy:    xx.x%

Baseline (gitleaks on CredData):
  Precision:   52.6%
  Recall:      24.4%
  F1:          0.334
```

If `bench-data/` is not present, the binary prints a clear error message and exits with code 0.

## Usage

```bash
# 1. Download dataset (one-time)
./scripts/setup-bench-data.sh

# 2. Run benchmarks
cargo bench --bench throughput
cargo bench --bench per_rule

# 3. Run confusion matrix
cargo run --bin cav-bench-confusion
```

## Non-Goals

- Measuring memory allocation precisely (would require heap profiling tools)
- Benchmarking against non-source-code datasets (logs, binary files)
- Comparison with other secret detection tools (only caviarder vs CredData ground truth)
- Running benchmarks in CI (dataset is too large)

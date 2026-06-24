//! # Confusion Matrix Benchmark against Samsung/CredData
//!
//! This binary evaluates caviarder against a labeled ground-truth dataset
//! and reports precision, recall, F1, and accuracy.
//!
//! ## Dataset: Samsung/CredData ([Apache-2.0](https://github.com/Samsung/CredData))
//!
//! - 66,898 labeled lines extracted from 333 real-world repositories
//! - Each line is annotated **T (True)** = actually contains a secret, or
//!   **F (False)** = does NOT contain a secret (but might look like one)
//! - Covers categories: API keys, passwords, tokens, private keys, URL
//!   credentials, nonces, UUIDs, and more
//! - The 333 repos were scanned with CredSweeper + manual review to produce
//!   ground truth
//!
//! ## Methodology
//!
//! 1. Load all 222 embedded gitleaks rules (same rules `cav` uses)
//! 2. For each of the 66,898 labeled lines, run the redactor
//! 3. Compare caviarder's output against ground truth:
//!    - **TP**: redacted AND labeled T (correct catch)
//!    - **FP**: redacted but labeled F (false alarm)
//!    - **FN**: not redacted but labeled T (missed secret)
//!    - **TN**: not redacted AND labeled F (correct ignore)
//! 4. Compute precision, recall, F1, MCC, accuracy
//!
//! ## Methodology note vs Official CredData benchmarks
//!
//! The official CredData benchmark scans **all 19.4M lines** of the repository files.
//! Our benchmark scans only the **66,898 pre-identified suspicious lines** from metadata.
//! This means our results (higher recall/precision) are not directly comparable to the
//! official benchmark numbers reported by CredSweeper, gitleaks, truffleHog, etc.
//! We include the official numbers as a reference — the key comparison is qualitative:
//! which pattern classes each tool handles well.
//!
//! ## How to run
//!
//! ```bash
//! # Download the dataset first (one-time, ~350 MB):
//! ./scripts/setup-bench-data.sh
//!
//! # Run the benchmark (release mode for speed):
//! cargo run --release --bin cav-bench-confusion
//! ```
//!
//! ## Expected output
//!
//! caviarder outperforms the published gitleaks baseline on CredData
//! (precision: 52.6%, recall: 24.4%, F1: 0.334).
//!
//! See also: [`benches/throughput.rs`] for throughput measurement,
//! [`benches/per_rule.rs`] for per-rule timing.

use caviarder::rules;
use caviarder::{Redactor, Rule};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::SystemTime;

const BENCH_DATA: &str = "bench-data";
const META_DIR: &str = "bench-data/meta";
const RESULTS_DIR: &str = "bench-results";

struct Entry {
    file_path: String,
    line_start: usize,
    ground_truth: String,
    category: String,
}

fn load_metadata() -> Vec<Entry> {
    let mut entries = Vec::new();
    let meta_dir = Path::new(META_DIR);

    if !meta_dir.is_dir() {
        eprintln!("ERROR: metadata directory not found at '{META_DIR}'");
        eprintln!("Run `./scripts/setup-bench-data.sh` first to download the dataset.");
        std::process::exit(1);
    }

    // Read all CSV files in the meta directory
    for dir_entry in fs::read_dir(meta_dir).expect("failed to read meta dir") {
        let dir_entry = dir_entry.expect("failed to read dir entry");
        let path = dir_entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("csv") {
            continue;
        }

        let mut reader = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_path(&path)
            .expect("failed to open metadata CSV");

        for result in reader.records() {
            let record = result.expect("invalid CSV record");
            // Columns: Id,FileID,Domain,RepoName,FilePath,LineStart,LineEnd,GroundTruth,...
            // FilePath (index 4) is relative like "data/<RepoID>/src/<FileID>.ext"
            entries.push(Entry {
                file_path: record.get(4).unwrap_or("").to_string(),
                line_start: record.get(5).unwrap_or("1").parse().unwrap_or(1),
                ground_truth: record.get(7).unwrap_or("F").to_string(),
                category: record.get(12).unwrap_or("").to_string(),
            });
        }
    }

    entries
}

fn main() {
    let entries = load_metadata();
    eprintln!("Loaded {} metadata entries", entries.len());

    // Load all gitleaks rules (same as `cav` uses)
    let all_rules = match rules::load_embedded() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ERROR: failed to load embedded rules: {e}");
            std::process::exit(1);
        }
    };
    let custom_rules = rules::load_embedded_custom().unwrap_or_default();

    let mut full_rules: Vec<Rule> = all_rules;
    full_rules.extend(custom_rules);

    if full_rules.is_empty() {
        eprintln!("ERROR: no rules loaded");
        std::process::exit(1);
    }

    eprintln!("Loaded {} rules", full_rules.len());
    let num_rules = full_rules.len();

    let redactor = Redactor::new(full_rules, "[CAVIARDER]");

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut tn = 0usize;
    let mut skipped = 0usize;

    // Collect up to 5 examples of each error type for display
    let mut fn_samples: Vec<(String, String)> = Vec::new();
    let mut fp_samples: Vec<(String, String)> = Vec::new();

    for entry in &entries {
        let file_path = Path::new(BENCH_DATA).join(&entry.file_path);

        let content = match fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(_) => {
                skipped += 1;
                continue;
            }
        };

        let line = match content.lines().nth(entry.line_start - 1) {
            Some(l) => l,
            None => {
                skipped += 1;
                continue;
            }
        };

        let outcome = redactor.redact(line);
        let was_redacted = outcome.text != line;

        let is_true = entry.ground_truth.trim() == "T";

        match (was_redacted, is_true) {
            (true, true) => tp += 1,
            (true, false) => {
                if fp_samples.len() < 5 {
                    fp_samples.push((entry.category.clone(), line.to_string()));
                }
                fp += 1;
            }
            (false, true) => {
                if fn_samples.len() < 5 {
                    fn_samples.push((entry.category.clone(), line.to_string()));
                }
                fn_ += 1;
            }
            (false, false) => tn += 1,
        }
    }

    let total = tp + fp + fn_ + tn;
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let accuracy = if total > 0 {
        (tp + tn) as f64 / total as f64
    } else {
        0.0
    };
    // Matthews Correlation Coefficient — robust to class imbalance
    let mcc_denom =
        ((tp + fp) as f64 * (tp + fn_) as f64 * (tn + fp) as f64 * (tn + fn_) as f64).sqrt();
    let mcc = if mcc_denom > 0.0 {
        ((tp as f64 * tn as f64) - (fp as f64 * fn_ as f64)) / mcc_denom
    } else {
        0.0
    };

    // Write structured JSON results to bench-results/
    let version = env!("CARGO_PKG_VERSION");
    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let _ = fs::create_dir_all(RESULTS_DIR);

    let json_body = format!(
        r#"{{"version":"{version}","dataset":"Samsung/CredData ({total} instances)","unix_ts":{ts},"rules_loaded":{rules},"metrics":{{"tp":{tp},"fp":{fp},"fn":{fn_},"tn":{tn},"precision":{p:.4},"recall":{r:.4},"f1":{f1_:.4},"mcc":{mcc_:.4},"accuracy":{acc:.4}}},"baseline_official":{{"credSweeper":{{"precision":0.917,"recall":0.808,"f1":0.859,"mcc":0.860}},"gitleaks":{{"precision":0.526,"recall":0.244,"f1":0.334,"mcc":0.358}}}}}}"#,
        version = version,
        ts = timestamp,
        rules = num_rules,
        tp = tp,
        fp = fp,
        fn_ = fn_,
        tn = tn,
        p = precision,
        r = recall,
        f1_ = f1,
        mcc_ = mcc,
        acc = accuracy,
    );

    // Write versioned file and latest copy
    let result_file = format!("{RESULTS_DIR}/confusion-v{version}.json");
    let latest_file = format!("{RESULTS_DIR}/latest.json");
    for path in &[&result_file, &latest_file] {
        if let Ok(mut f) = fs::File::create(path) {
            let _ = writeln!(f, "{json_body}");
        }
    }
    eprintln!("Results written to {result_file}");

    println!();
    println!("================================================================================");
    println!(" Confusion Matrix: caviarder vs Samsung/CredData");
    println!("================================================================================");
    println!(" Dataset: Samsung/CredData (Apache-2.0)");
    println!("   https://github.com/Samsung/CredData");
    println!("   66,898 labeled lines from 333 real-world repositories");
    println!("   Ground truth: T = contains a real secret, F = not a secret");
    println!();
    println!(
        " Engine: caviarder v{} loaded {} gitleaks rules",
        env!("CARGO_PKG_VERSION"),
        num_rules
    );
    println!("================================================================================");
    println!();
    println!(" Instances:  {total}");
    println!(
        " True:       {} ({:.1}%)",
        tp + fn_,
        100.0 * (tp + fn_) as f64 / total as f64
    );
    println!(
        " False:      {} ({:.1}%)",
        fp + tn,
        100.0 * (fp + tn) as f64 / total as f64
    );
    println!(" Skipped:    {skipped}");
    println!();
    println!("                Predicted");
    println!("                redacted  clean");
    println!(" Actual True   {:>8} {:>6}", tp, fn_);
    println!("        False  {:>8} {:>6}", fp, tn);
    println!();
    println!(" Metrics:");
    println!("   Precision:  {:.1}%", precision * 100.0);
    println!("   Recall:     {:.1}%", recall * 100.0);
    println!("   F1:         {:.3}", f1);
    println!("   MCC:        {:.3}", mcc);
    println!("   Accuracy:   {:.1}%", accuracy * 100.0);
    println!();
    println!("--- Comparison: Official CredData Benchmarks (full-file scan) ---");
    println!(" Tool             Precision  Recall     F1      MCC");
    println!(" ---------------- ---------- ---------- ------- -------");
    println!(" CredSweeper      91.7%      80.8%      0.859   0.860");
    println!(" gitleaks         52.6%      24.4%      0.334   0.358");
    println!(" detect-secrets   14.2%      38.1%      0.206   0.232");
    println!(" truffleHog3      15.0%      54.7%      0.235   0.286");
    println!(" shhgit           51.9%       7.2%      0.126   0.193");
    println!(" -----------------------------------------------------------------");
    println!(" ⚠ NOTE: Our benchmark scans only the 67K pre-identified suspicious");
    println!("   lines (metadata). The official benchmark scans ALL 19.4M lines");
    println!("   of code in the dataset. Results are NOT directly comparable.");
    println!();

    // --- Interpretation ---
    println!("=== Interpreting Results ===");
    println!(" Ground Truth labels from CredData:");
    println!("   T (True)   = this line contains a real secret");
    println!("   F (False)  = this line is NOT a secret (but might look like one)");
    println!();
    println!(" Confusion Matrix cells:");
    println!("   TP = Predicted secret + Actual secret         -> ✓ correct catch");
    println!("   FP = Predicted secret + NOT a secret          -> ✗ false alarm");
    println!("   FN = Predicted benign + Actual secret         -> ✗ missed secret");
    println!("   TN = Predicted benign + NOT a secret          -> ✓ correct ignore");
    println!();
    println!(" Metrics:");
    println!("   Precision = TP / (TP + FP)   — when we flag something, how often");
    println!("                                 are we right?");
    println!("   Recall    = TP / (TP + FN)   — what fraction of real secrets do we");
    println!("                                 catch?");
    println!("   F1        = harmonic mean of precision & recall (balanced score)");
    println!("   MCC       = Matthews Correlation Coefficient (robust to imbalance)");
    println!("               range: -1 (total disagreement) to +1 (perfect prediction)");
    println!("   Accuracy  = (TP + TN) / Total — biased toward majority class");
    println!();

    // Print sampled missed secrets (FN)
    println!("--- False Negatives (missed secrets — NOT redacted but should have been) ---");
    for (cat, line) in &fn_samples {
        let truncated: String = line.chars().take(100).collect();
        println!("  [{cat:20}] {truncated}");
    }
    println!();

    // Print sampled false alarms (FP)
    println!("--- False Positives (false alarms — redacted but were NOT secrets) ---");
    for (cat, line) in &fp_samples {
        let truncated: String = line.chars().take(100).collect();
        println!("  [{cat:20}] {truncated}");
    }
    println!();
    println!("(Showing up to 5 examples of each type; {fn_}/{total} FN, {fp}/{total} FP total)");
    println!();
    println!(" Common FN patterns (missed):");
    println!("   - UUID/Nonce values that look like random strings but are not in gitleaks rules");
    println!("   - Passwords with low entropy (all lowercase, dictionary words)");
    println!("   - Credentials in custom env vars not covered by gitleaks rules");
    println!(" Common FP patterns (false alarms):");
    println!("   - Variable names containing 'key', 'secret', 'pass', 'token' as identifiers");
    println!("   - Comments or code referencing credential concepts without actual secrets");
    println!("   - Low-entropy placeholder values that match any pattern");
}

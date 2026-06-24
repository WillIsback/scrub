use caviarder::rules;
use caviarder::{Redactor, Rule};
use std::fs;
use std::path::Path;

const BENCH_DATA: &str = "bench-data";
const META_DIR: &str = "bench-data/meta";

struct Entry {
    file_path: String,
    line_start: usize,
    ground_truth: String,
}

fn load_metadata() -> Vec<Entry> {
    let mut entries = Vec::new();
    let meta_dir = Path::new(META_DIR);

    if !meta_dir.is_dir() {
        eprintln!("ERROR: metadata directory not found at '{META_DIR}'");
        eprintln!("Run `./scripts/setup-bench-data.sh` first to download the dataset.");
        std::process::exit(0);
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

    let redactor = Redactor::new(full_rules, "[CAVIARDER]");

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut tn = 0usize;
    let mut skipped = 0usize;

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
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
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

    println!();
    println!("=== Confusion Matrix (CredData) ===");
    println!(" Instances:  {total}");
    println!(" True:       {} ({:.1}%)", tp + fn_, 100.0 * (tp + fn_) as f64 / total as f64);
    println!(" False:      {} ({:.1}%)", fp + tn, 100.0 * (fp + tn) as f64 / total as f64);
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
    println!("   Accuracy:   {:.1}%", accuracy * 100.0);
    println!();
    println!(" Baseline (gitleaks on CredData):");
    println!("   Precision:  52.6%");
    println!("   Recall:     24.4%");
    println!("   F1:         0.334");
}

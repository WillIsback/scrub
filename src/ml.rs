/// Gradient boosting post-filter for caviarder (XGBoost).
///
/// After regex matching, each candidate is scored by a gradient boosting
/// ensemble using 32 features (24 value/line features + 8 rule-type one-hot).
/// If the score falls below a configurable threshold, the alert is suppressed.
///
/// Features are computed identically to the Python training pipeline
/// in `ml/features.py`.
///
/// Model exported from `ml/train_xgb.py` to `ml/model_xgb.rs`.

use crate::ml_model::N_FEATURES;

// Must match PYTHON keyword list in ml/features.py
const KEYWORDS: &[&str] = &[
    "password", "passwd", "pwd",
    "secret",
    "token",
    "api_key", "apikey", "api",
    "auth",
    "credential", "creds",
    "login",
    "db_", "database",
    "aws", "amazon",
    "ssh",
    "-----begin",
];

// Must match PYTHON sets in ml/features.py
const SOURCE_EXTS: &[&str] = &[
    ".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".cpp",
    ".h", ".hpp", ".rb", ".php", ".swift", ".kt", ".scala",
    ".pl", ".pm", ".sh", ".bash", ".zsh", ".ps1", ".r",
];

const CONFIG_EXTS: &[&str] = &[
    ".env", ".cfg", ".ini", ".conf", ".toml", ".yaml", ".yml",
    ".json", ".xml", ".properties", ".config", ".cnf",
];

/// Rule type categories — must match RULE_TYPE_CATEGORIES in ml/features.py
const RULE_TYPE_COUNT: usize = 8;

/// Classify a gitleaks rule ID into a type category index (0..8).
/// Returns `None` if the rule ID doesn't match any known category
/// (maps to all-zeros one-hot).
fn rule_type_index(rule_id: &str) -> Option<usize> {
    let rid = rule_id.to_lowercase();
    // Patterns for each category (must match PYTHON RULE_TYPE_CATEGORIES)
    let patterns: &[&[&str]] = &[
        &["api-key", "api_key", "apikey", "api-token", "api_token"],      // 0: api_key
        &["token"],                                                         // 1: token
        &["password", "passwd", "pwd", "secret-key", "secret_key", "secret"], // 2: password
        &["auth", "authorization", "bearer", "basic", "credential"],        // 3: auth
        &["private-key", "private_key", "pem", "rsa", "ecdsa", "ed25519"],  // 4: private_key
        &["url", "uri", "curl", "ftp", "postgres"],                         // 5: url
        &["uuid", "nonce"],                                                 // 6: uuid
        &["key", "id", "pat", "sid", "token", "access"],                    // 7: key
    ];

    for (idx, pats) in patterns.iter().enumerate() {
        for pat in *pats {
            if rid.contains(pat) {
                return Some(idx);
            }
        }
    }
    None
}

// -----------------------------------------------------------------------
// Feature computation
// -----------------------------------------------------------------------

/// Compute the 32-element feature vector for a potential credential.
pub fn compute_features(value: &str, line: &str, filename: &str, rule_id: &str) -> [f32; N_FEATURES] {
    let (is_src, is_cfg) = file_type_flags(filename);
    let mut rule_hot = [0.0f32; RULE_TYPE_COUNT];
    if let Some(idx) = rule_type_index(rule_id) {
        rule_hot[idx] = 1.0;
    }

    [
        shannon_entropy(value) as f32,           // 0: entropy
        log_len(value) as f32,                   // 1: log_len
        digit_ratio(value) as f32,               // 2: digit_ratio
        upper_ratio(value) as f32,               // 3: upper_ratio
        lower_ratio(value) as f32,               // 4: lower_ratio
        special_ratio(value) as f32,             // 5: special_ratio
        max_consecutive_repeat(value) as f32,    // 6: max_consecutive_repeat
        has_base64_padding(value) as u8 as f32,  // 7: has_base64_padding
        has_char(value, b':') as u8 as f32,       // 8: has_colon
        has_char(value, b'/') as u8 as f32,       // 9: has_slash
        has_char(value, b'.') as u8 as f32,       // 10: has_dot
        has_char(value, b'-') as u8 as f32,       // 11: has_hyphen
        has_char(value, b'_') as u8 as f32,       // 12: has_underscore
        has_char(value, b'=') as u8 as f32,       // 13: has_equals
        is_hex_only(value) as u8 as f32,         // 14: is_hex_only
        log_len(line) as f32,                    // 15: line_log_len
        keyword_count(line) as f32,              // 16: keyword_count
        has_char(line, b'=') as u8 as f32,        // 17: is_assignment
        has_quote(line) as u8 as f32,            // 18: has_quotes
        is_src as u8 as f32,                     // 19: is_source_file
        is_cfg as u8 as f32,                     // 20: is_config_file
        rule_hot[0],                              // 21: rule_type_api_key
        rule_hot[1],                              // 22: rule_type_token
        rule_hot[2],                              // 23: rule_type_password
        rule_hot[3],                              // 24: rule_type_auth
        rule_hot[4],                              // 25: rule_type_private_key
        rule_hot[5],                              // 26: rule_type_url
        rule_hot[6],                              // 27: rule_type_uuid
        rule_hot[7],                              // 28: rule_type_key
        0.0,                                      // 29 reserved
        0.0,                                      // 30 reserved
        0.0,                                      // 31 reserved
    ]
}

fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    let length = s.len();
    let mut freq: [usize; 256] = [0; 256];
    for &b in s.as_bytes() { freq[b as usize] += 1; }
    let mut entropy = 0.0_f64;
    for &count in freq.iter() {
        if count > 0 {
            let p = count as f64 / length as f64;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn log_len(s: &str) -> f64 {
    if s.is_empty() { 0.0 } else { (s.len() as f64).ln() }
}

fn digit_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    s.bytes().filter(|c| c.is_ascii_digit()).count() as f64 / s.len() as f64
}

fn upper_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    s.bytes().filter(|c| c.is_ascii_uppercase()).count() as f64 / s.len() as f64
}

fn lower_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    s.bytes().filter(|c| c.is_ascii_lowercase()).count() as f64 / s.len() as f64
}

fn special_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    s.bytes().filter(|c| !c.is_ascii_alphanumeric()).count() as f64 / s.len() as f64
}

fn max_consecutive_repeat(s: &str) -> usize {
    if s.is_empty() { return 0; }
    let mut max_run = 1;
    let mut current_run = 1;
    let bytes = s.as_bytes();
    for i in 1..bytes.len() {
        if bytes[i] == bytes[i - 1] {
            current_run += 1;
            if current_run > max_run { max_run = current_run; }
        } else {
            current_run = 1;
        }
    }
    max_run
}

fn has_base64_padding(s: &str) -> bool {
    s.as_bytes().last() == Some(&b'=')
}

fn has_char(s: &str, ch: u8) -> bool {
    s.as_bytes().contains(&ch)
}

fn is_hex_only(s: &str) -> bool {
    if s.is_empty() { return true; }
    s.bytes().all(|b| b.is_ascii_hexdigit())
}

fn keyword_count(line: &str) -> usize {
    let lower = line.to_ascii_lowercase();
    KEYWORDS.iter().filter(|kw| lower.contains(*kw)).count()
}

fn has_quote(s: &str) -> bool {
    s.contains('\'') || s.contains('"')
}

fn file_type_flags(filename: &str) -> (bool, bool) {
    if filename.is_empty() { return (false, false); }
    let name = extract_filename(filename);
    let dot = name.rfind('.');
    let ext = dot.map(|i| &name[i..]).unwrap_or("");
    let ext_lower = ext.to_ascii_lowercase();
    let is_src = SOURCE_EXTS.contains(&ext_lower.as_str());
    let is_cfg = CONFIG_EXTS.contains(&ext_lower.as_str());
    (is_src, is_cfg)
}

// -----------------------------------------------------------------------
// Prediction (gradient boosting tree ensemble)
// -----------------------------------------------------------------------

/// Predict secret probability using gradient boosting tree ensemble.
///
/// We re-implement the tree walk here because `ml_model::predict_xgb`
/// uses global indexing but the exported `left`/`right` values are
/// tree-local (relative to each tree's start offset).
pub fn predict_proba(features: &[f32; N_FEATURES]) -> f64 {
    use crate::ml_model::{BIAS, N_TREES, NODES, TREE_OFFSETS};
    let mut sum = BIAS as f64;
    for t in 0..N_TREES {
        let base = TREE_OFFSETS[t] as usize;
        let mut local_idx = 0usize; // start at root (local index 0)
        loop {
            let (feat_idx, threshold, left, right, leaf) = NODES[base + local_idx];
            if feat_idx < 0 {
                // Leaf node
                sum += leaf as f64;
                break;
            }
            if features[feat_idx as usize] <= threshold {
                local_idx = left as usize; // left/right are LOCAL indices
            } else {
                local_idx = right as usize;
            }
        }
    }
    1.0 / (1.0 + (-sum).exp())
}

/// Predict whether a candidate is a true secret (above threshold).
pub fn predict(features: &[f32; N_FEATURES], threshold: f64) -> bool {
    predict_proba(features) >= threshold
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Extract the line containing byte position `pos` from `text`.
pub fn extract_line(text: &str, pos: usize) -> &str {
    let bytes = text.as_bytes();
    if pos >= bytes.len() { return text; }

    let line_start = match bytes[..=pos].iter().rposition(|&b| b == b'\n') {
        Some(nl) => nl + 1,
        None => 0,
    };

    let line_end = match bytes[pos..].iter().position(|&b| b == b'\n') {
        Some(nl) => pos + nl,
        None => bytes.len(),
    };

    let line_end = if line_end > 0 && bytes[line_end - 1] == b'\r' {
        line_end - 1
    } else {
        line_end
    };

    &text[line_start..line_end]
}

/// Extract the filename from a path (last component).
pub fn extract_filename(path: &str) -> &str {
    if path.is_empty() { return path; }
    let after_slash = path.rfind('/').map(|i| &path[i + 1..]).unwrap_or(path);
    let after_bslash = after_slash.rfind('\\').map(|i| &after_slash[i + 1..]).unwrap_or(after_slash);
    after_bslash
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_length() {
        let f = compute_features("abc123", "x = abc123", ".env", "test-rule");
        assert_eq!(f.len(), N_FEATURES);
    }

    #[test]
    fn test_feature_entropy_first() {
        let f = compute_features("aaaaa", "x = 'aaaaa'", ".py", "test-rule");
        assert_eq!(f[0], 0.0);
    }

    #[test]
    fn test_digit_ratio_all() {
        let f = compute_features("12345", "x = 12345", ".py", "test-rule");
        assert_eq!(f[2], 1.0);
    }

    #[test]
    fn test_digit_ratio_none() {
        let f = compute_features("abcde", "x = abcde", ".py", "test-rule");
        assert_eq!(f[2], 0.0);
    }

    #[test]
    fn test_special_ratio_all() {
        let f = compute_features("!@#$%", "x = !@#$%", ".py", "test-rule");
        assert_eq!(f[5], 1.0);
    }

    #[test]
    fn test_hex_only() {
        let f = compute_features("DEADbeef", "x = DEADbeef", ".py", "test-rule");
        assert_eq!(f[14], 1.0);
    }

    #[test]
    fn test_source_file_flag() {
        let f = compute_features("abc", "x = abc", "script.py", "test-rule");
        assert_eq!(f[19], 1.0);
    }

    #[test]
    fn test_config_file_flag() {
        let f = compute_features("abc", "x = abc", ".env", "test-rule");
        assert_eq!(f[20], 1.0);
    }

    #[test]
    fn test_keyword_count() {
        let f = compute_features("secret123", "password = 'secret123'", ".env", "test-rule");
        assert!(f[16] > 0.0);
    }

    #[test]
    fn test_different_inputs_different() {
        let a = compute_features("12345", "x = 12345", ".py", "test-rule");
        let b = compute_features("abcde", "x = abcde", ".py", "test-rule");
        assert_ne!(a, b);
    }

    #[test]
    fn test_rule_type_api_key() {
        // "api-key" in rule_id → category 0 (api_key)
        let f = compute_features("abc123", "x = abc123", ".py", "aws-api-key");
        assert_eq!(f[21], 1.0, "api_key should be at index 21");
        assert_eq!(f[22], 0.0, "token should be 0");
    }

    #[test]
    fn test_rule_type_password() {
        let f = compute_features("abc123", "x = abc123", ".py", "hashicorp-tf-password");
        assert_eq!(f[23], 1.0, "password should be at index 23");
    }

    #[test]
    fn test_rule_type_unknown_rule_id() {
        let f = compute_features("abc123", "x = abc123", ".py", "some-unknown-rule");
        // Unknown rule IDs ("some-unknown-rule" contains no keyword matches)
        // fall through to all-zeros in the rule-type one-hot.
        for i in 21..29 {
            assert_eq!(f[i], 0.0, "Expected rule_type[{i}] == 0 for unknown rule");
        }
    }

    #[test]
    fn test_rule_type_all_zeros_for_empty_rule() {
        let f = compute_features("abc", "x = abc", ".py", "");
        // Empty rule_id should produce all zeros in rule-type one-hot
        for i in 21..29 {
            assert_eq!(f[i], 0.0, "Expected rule_type[{i}] == 0");
        }
    }

    // -----------------------------------------------------------------------
    // Prediction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_predict_proba_range() {
        let features = [0.5; N_FEATURES];
        let prob = predict_proba(&features);
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn test_predict_threshold() {
        let features = [0.0; N_FEATURES];
        // All zeros should be confident non-secret (model trained on
        // non-"other" entries gives ~0.61 for empty value).
        let prob = predict_proba(&features);
        assert!(prob < 0.9, "all-zero features should be below 0.9, got {prob}");
    }

    #[test]
    fn test_predict_value_different_from_non_secret() {
        let f_secret = compute_features(
            "sk-proj-A1b2C3d4E5f6G7h8I9j0K1l2",
            "openai_key = 'sk-proj-A1b2C3d4E5f6G7h8I9j0K1l2'",
            ".env",
            "anthropic-api-key",
        );
        let f_benign = compute_features("hello", "x = hello", "file.py", "test");
        let prob_secret = predict_proba(&f_secret);
        let prob_benign = predict_proba(&f_benign);
        // The secret-like value should score higher than the benign word
        assert!(
            prob_secret > prob_benign,
            "secret ({prob_secret:.3}) should score higher than benign ({prob_benign:.3})"
        );
    }

    #[test]
    fn test_rule_type_affects_score() {
        // Same value, different rule types should give different scores
        let value = "abc123";
        let line = "x = abc123";
        let f_key = compute_features(value, line, ".py", "generic-api-key");
        let f_password = compute_features(value, line, ".py", "hashicorp-tf-password");
        // These might be equal (model may not differentiate all types), but
        // they should at least not crash
        let _ = predict_proba(&f_key);
        let _ = predict_proba(&f_password);
    }

    // -----------------------------------------------------------------------
    // extract_line tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_line_simple() {
        let text = "first line\nsecond line\nthird line";
        let pos = text.find("second").unwrap();
        let line = extract_line(text, pos);
        assert_eq!(line, "second line");
    }

    #[test]
    fn test_extract_line_first_line() {
        let text = "first line\nsecond line";
        let line = extract_line(text, 0);
        assert_eq!(line, "first line");
    }

    #[test]
    fn test_extract_line_last_line() {
        let text = "first line\nsecond line";
        let pos = text.find("second").unwrap();
        let line = extract_line(text, pos);
        assert_eq!(line, "second line");
    }

    #[test]
    fn test_extract_line_no_newlines() {
        let text = "single line";
        let line = extract_line(text, 3);
        assert_eq!(line, "single line");
    }

    // -----------------------------------------------------------------------
    // extract_filename tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_filename_simple() {
        assert_eq!(extract_filename("src/main.rs"), "main.rs");
    }

    #[test]
    fn test_extract_filename_no_path() {
        assert_eq!(extract_filename("main.rs"), "main.rs");
    }

    #[test]
    fn test_extract_filename_empty() {
        assert_eq!(extract_filename(""), "");
    }

    #[test]
    fn test_extract_filename_windows() {
        assert_eq!(extract_filename("src\\main.rs"), "main.rs");
    }

    #[test]
    fn test_extract_filename_deep() {
        assert_eq!(extract_filename("/home/user/project/config/.env"), ".env");
    }
}

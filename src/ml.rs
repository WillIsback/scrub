/// Logistic regression post-filter for caviarder.
///
/// After regex matching, each candidate is scored by a tiny (~200 byte)
/// logistic regression model. If the score falls below a configurable
/// threshold, the alert is suppressed (false positive filter).
///
/// Features are computed identically to the Python training pipeline
/// in `ml/features.py`.

use crate::ml_model::{BIAS, COEFFS, N_FEATURES};

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

// -----------------------------------------------------------------------
// Feature computation
// -----------------------------------------------------------------------

/// Compute the 24-element feature vector for a potential credential.
pub fn compute_features(value: &str, line: &str, filename: &str) -> [f64; N_FEATURES] {
    let (is_src, is_cfg) = file_type_flags(filename);
    [
        crate::shannon_entropy(value),           // 0: entropy
        log_len(value),                          // 1: log_len
        digit_ratio(value),                      // 2: digit_ratio
        upper_ratio(value),                      // 3: upper_ratio
        lower_ratio(value),                      // 4: lower_ratio
        special_ratio(value),                    // 5: special_ratio
        max_consecutive_repeat(value) as f64,    // 6: max_consecutive_repeat
        has_base64_padding(value) as u8 as f64,  // 7: has_base64_padding
        has_char(value, b':') as u8 as f64,       // 8: has_colon
        has_char(value, b'/') as u8 as f64,       // 9: has_slash
        has_char(value, b'.') as u8 as f64,       // 10: has_dot
        has_char(value, b'-') as u8 as f64,       // 11: has_hyphen
        has_char(value, b'_') as u8 as f64,       // 12: has_underscore
        has_char(value, b'=') as u8 as f64,       // 13: has_equals
        is_hex_only(value) as u8 as f64,         // 14: is_hex_only
        log_len(line),                           // 15: line_log_len
        keyword_count(line) as f64,              // 16: keyword_count
        has_char(line, b'=') as u8 as f64,        // 17: is_assignment
        has_quote(line) as u8 as f64,            // 18: has_quotes
        is_src as u8 as f64,                     // 19: is_source_file
        is_cfg as u8 as f64,                     // 20: is_config_file
        0.0,                                     // 21 reserved
        0.0,                                     // 22 reserved
        0.0,                                     // 23 reserved
    ]
}

fn log_len(s: &str) -> f64 {
    if s.is_empty() { 0.0 } else { (s.len() as f64).ln() }
}

fn digit_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    let count = s.chars().filter(|c| c.is_ascii_digit()).count();
    count as f64 / s.len() as f64
}

fn upper_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    let count = s.chars().filter(|c| c.is_ascii_uppercase()).count();
    count as f64 / s.len() as f64
}

fn lower_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    let count = s.chars().filter(|c| c.is_ascii_lowercase()).count();
    count as f64 / s.len() as f64
}

fn special_ratio(s: &str) -> f64 {
    if s.is_empty() { return 0.0; }
    let count = s.chars().filter(|c| !c.is_ascii_alphanumeric()).count();
    count as f64 / s.len() as f64
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
    // ends with "=" or "=="
    let bytes = s.as_bytes();
    bytes.last() == Some(&b'=')
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
// Prediction
// -----------------------------------------------------------------------

/// Logistic regression sigmoid: σ(w·x + b)
pub fn predict_proba(features: &[f64; N_FEATURES]) -> f64 {
    let logit: f64 = COEFFS.iter()
        .zip(features.iter())
        .map(|(w, x)| w * x)
        .sum::<f64>() + BIAS;
    1.0 / (1.0 + (-logit).exp())
}

/// Predict whether a candidate is a true secret (above threshold).
pub fn predict(features: &[f64; N_FEATURES], threshold: f64) -> bool {
    predict_proba(features) >= threshold
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Extract the line containing byte position `pos` from `text`.
pub fn extract_line(text: &str, pos: usize) -> &str {
    let bytes = text.as_bytes();
    if pos >= bytes.len() { return text; }

    // Find start of line (previous newline or start of text)
    let line_start = match bytes[..=pos].iter().rposition(|&b| b == b'\n') {
        Some(nl) => nl + 1,
        None => 0,
    };

    // Find end of line (next newline or end of text)
    let line_end = match bytes[pos..].iter().position(|&b| b == b'\n') {
        Some(nl) => pos + nl,
        None => bytes.len(),
    };

    // Trim trailing \r if present
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
    // Handle both / and \ path separators
    let after_slash = path.rfind('/').map(|i| &path[i + 1..]).unwrap_or(path);
    let after_bslash = after_slash.rfind('\\').map(|i| &after_slash[i + 1..]).unwrap_or(after_slash);
    after_bslash
}

// -----------------------------------------------------------------------
// Validate against Python feature computation
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Feature extraction tests (must match Python implementation)
    // -----------------------------------------------------------------------

    #[test]
    fn test_entropy_uniform() {
        assert_eq!(crate::shannon_entropy("aaaaa"), 0.0);
    }

    #[test]
    fn test_entropy_empty() {
        assert_eq!(crate::shannon_entropy(""), 0.0);
    }

    #[test]
    fn test_features_length() {
        let f = compute_features("abc123", "x = abc123", ".env");
        assert_eq!(f.len(), N_FEATURES);
    }

    #[test]
    fn test_feature_entropy_first() {
        let f = compute_features("aaaaa", "x = 'aaaaa'", ".py");
        assert_eq!(f[0], 0.0);
    }

    #[test]
    fn test_digit_ratio_all() {
        let f = compute_features("12345", "x = 12345", ".py");
        assert_eq!(f[2], 1.0);
    }

    #[test]
    fn test_digit_ratio_none() {
        let f = compute_features("abcde", "x = abcde", ".py");
        assert_eq!(f[2], 0.0);
    }

    #[test]
    fn test_special_ratio_all() {
        let f = compute_features("!@#$%", "x = !@#$%", ".py");
        assert_eq!(f[5], 1.0);
    }

    #[test]
    fn test_hex_only() {
        let f = compute_features("DEADbeef", "x = DEADbeef", ".py");
        assert_eq!(f[14], 1.0);
    }

    #[test]
    fn test_source_file_flag() {
        let f = compute_features("abc", "x = abc", "script.py");
        assert_eq!(f[19], 1.0);
    }

    #[test]
    fn test_config_file_flag() {
        let f = compute_features("abc", "x = abc", ".env");
        assert_eq!(f[20], 1.0);
    }

    #[test]
    fn test_keyword_count() {
        let f = compute_features("secret123", "password = 'secret123'", ".env");
        assert!(f[16] > 0.0);
    }

    #[test]
    fn test_different_inputs_different() {
        let a = compute_features("12345", "x = 12345", ".py");
        let b = compute_features("abcde", "x = abcde", ".py");
        assert_ne!(a, b);
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
    fn test_predict_proba_zero_features() {
        let features = [0.0; N_FEATURES];
        let prob = predict_proba(&features);
        // With all-zero features, logit = BIAS = -13.25
        // sigmoid(-13.25) ≈ 0.000002
        assert!(prob < 0.001);
    }

    #[test]
    fn test_predict_threshold() {
        let features = [0.0; N_FEATURES];
        let result = predict(&features, 0.5);
        assert!(!result, "zero features should be below threshold");
    }

    #[test]
    fn test_predict_high_value() {
        // A high-entropy, mixed-char string should score above threshold
        let f = compute_features(
            "sk-proj-A1b2C3d4E5f6G7h8I9j0K1l2",
            "openai_key = 'sk-proj-A1b2C3d4E5f6G7h8I9j0K1l2'",
            ".env",
        );
        let prob = predict_proba(&f);
        assert!(prob > 0.5, "API key should score high, got {prob}");
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

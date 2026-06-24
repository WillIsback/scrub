pub mod rules;
pub mod ml;
pub mod ml_model;

use regex::Regex;

pub struct Rule {
    pub id: String,
    pub regex: Regex,
    pub entropy: Option<f64>,
}

pub struct Redactor {
    rules: Vec<Rule>,
    placeholder: String,
    filename: String,
    ml_threshold: f64,
}

pub struct Outcome {
    pub text: String,
    pub counts: Vec<(String, usize)>,
}

impl Outcome {
    pub fn total(&self) -> usize {
        self.counts.iter().map(|(_, c)| c).sum()
    }
}

impl Redactor {
    /// Create a new Redactor from a list of rules and a placeholder string.
    pub fn new(rules: Vec<Rule>, placeholder: impl Into<String>) -> Self {
        Redactor {
            rules,
            placeholder: placeholder.into(),
            filename: String::new(),
            ml_threshold: 0.0,
        }
    }

    /// Enable ML-based false positive filtering at the given threshold.
    /// Values below the threshold are suppressed (not redacted).
    pub fn with_ml_threshold(mut self, threshold: f64) -> Self {
        self.ml_threshold = threshold;
        self
    }

    /// Set the filename for ML feature computation (file extension matters).
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = filename.into();
        self
    }

    /// Apply every rule in order, returning the scrubbed text and per-rule counts.
    pub fn redact(&self, input: &str) -> Outcome {
        let mut text = input.to_string();
        let mut counts = Vec::new();

        for rule in &self.rules {
            let mut count = 0usize;

            let needs_filter = rule.entropy.is_some() || self.ml_threshold > 0.0;

            if needs_filter {
                // Slow path: collect matches, apply entropy/ML filters, then
                // replace in reverse order to preserve positions.
                struct MatchEntry {
                    start: usize,
                    end: usize,
                    capture_start: Option<usize>,
                    capture_end: Option<usize>,
                }
                let mut matches: Vec<MatchEntry> = Vec::new();

                for caps in rule.regex.captures_iter(&text) {
                    let full = caps.get(0).unwrap();
                    let (check_str, value_str, capture_start, capture_end) = match caps.get(1) {
                        Some(v) => (v.as_str(), v.as_str(), Some(v.start()), Some(v.end())),
                        None => (full.as_str(), full.as_str(), None, None),
                    };

                    // Entropy check
                    if let Some(min_entropy) = rule.entropy {
                        if shannon_entropy(check_str) < min_entropy {
                            continue;
                        }
                    }

                    // ML threshold check
                    if self.ml_threshold > 0.0 {
                        let line = crate::ml::extract_line(&text, full.start());
                        let features =
                            crate::ml::compute_features(value_str, line, &self.filename, &rule.id);
                        if !crate::ml::predict(&features, self.ml_threshold) {
                            continue;
                        }
                    }

                    matches.push(MatchEntry {
                        start: full.start(),
                        end: full.end(),
                        capture_start,
                        capture_end,
                    });
                }

                // Replace in reverse order to preserve positions
                for entry in matches.into_iter().rev() {
                    let (rs, re) = match (entry.capture_start, entry.capture_end) {
                        (Some(s), Some(e)) => (s, e),
                        _ => (entry.start, entry.end),
                    };
                    // Preserve context around the captured value
                    let before = &text[entry.start..rs].to_string();
                    let after = &text[re..entry.end].to_string();
                    let replacement = if before.is_empty() && after.is_empty() {
                        self.placeholder.clone()
                    } else {
                        format!("{}{}{}", before, self.placeholder, after)
                    };
                    text.replace_range(entry.start..entry.end, &replacement);
                    count += 1;
                }
            } else {
                // Fast path: no entropy or ML filter — use highly optimized
                // regex::Regex::replace_all in a single pass.
                let replaced = rule.regex.replace_all(&text, |caps: &regex::Captures| {
                    count += 1;
                    if caps.len() >= 2 {
                        let full_match = caps.get(0).unwrap();
                        let value = caps.get(1).unwrap();
                        let before =
                            &full_match.as_str()[..value.start() - full_match.start()];
                        let after =
                            &full_match.as_str()[value.end() - full_match.start()..];
                        format!("{}{}{}", before, self.placeholder, after)
                    } else {
                        self.placeholder.clone()
                    }
                });
                text = replaced.into_owned();
            }

            if count > 0 {
                counts.push((rule.id.clone(), count));
            }
        }

        Outcome { text, counts }
    }
}

pub fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut freq = [0usize; 256];
    for &b in s.as_bytes() {
        freq[b as usize] += 1;
    }
    let len = s.len() as f64;
    let mut entropy = 0.0_f64;
    for &count in freq.iter() {
        if count == 0 {
            continue;
        }
        let p = count as f64 / len;
        entropy -= p * p.log2();
    }
    entropy
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    #[test]
    fn entropy_empty() {
        assert_eq!(shannon_entropy(""), 0.0);
    }

    #[test]
    fn entropy_same_char() {
        let e = shannon_entropy("AAAA");
        assert!(e < 0.1);
    }

    #[test]
    fn entropy_high() {
        let e = shannon_entropy("sk-proj-abc123ABC/+def456DEF=");
        assert!(e > 4.0);
    }

    #[test]
    fn redact_replaces_matched_text() {
        let rule = Rule {
            id: "test-key".into(),
            regex: Regex::new(r"sk-[A-Za-z0-9]{20,}").unwrap(),
            entropy: None,
        };
        let redactor = Redactor::new(vec![rule], "[CAVIARDER]");
        let outcome = redactor.redact("my key is sk-abc123DEF456ghi789jkl012");
        assert_eq!(outcome.text, "my key is [CAVIARDER]");
        assert_eq!(outcome.total(), 1);
    }

    #[test]
    fn redact_multiple_matches_same_rule() {
        let rule = Rule {
            id: "test-key".into(),
            regex: Regex::new(r"AKIA[A-Z0-9]{16}").unwrap(),
            entropy: None,
        };
        let redactor = Redactor::new(vec![rule], "[CAVIARDER]");
        let outcome = redactor.redact("keys: AKIAIOSFODNN7EXAMPLE and AKIAZZZZZZZZZZZZZZZZ");
        assert_eq!(outcome.text, "keys: [CAVIARDER] and [CAVIARDER]");
        assert_eq!(outcome.total(), 2);
    }

    #[test]
    fn redact_multiple_rules_applied_in_order() {
        let rule1 = Rule {
            id: "aws".into(),
            regex: Regex::new(r"AKIA[A-Z0-9]{16}").unwrap(),
            entropy: None,
        };
        let rule2 = Rule {
            id: "generic".into(),
            regex: Regex::new(r"(?i)password=\S+").unwrap(),
            entropy: None,
        };
        let redactor = Redactor::new(vec![rule1, rule2], "[CAVIARDER]");
        let outcome = redactor.redact("aws=AKIAIOSFODNN7EXAMPLE password=hunter2");
        assert_eq!(outcome.text, "aws=[CAVIARDER] [CAVIARDER]");
        assert_eq!(outcome.total(), 2);
    }

    #[test]
    fn redact_entropy_filter_low_entropy() {
        let rule = Rule {
            id: "high-entropy-only".into(),
            regex: Regex::new(r"\b[A-Za-z0-9/+=-]{10,}\b").unwrap(),
            entropy: Some(4.0),
        };
        let redactor = Redactor::new(vec![rule], "[CAVIARDER]");
        let outcome = redactor.redact("low entropy AAAAAAAA high entropy sk-proj-abc123DEF456");
        // "AAAAAAAA" is 8 identical chars: entropy 0.0 < 4.0 → not redacted
        // "sk-proj-abc123DEF456" has mixed chars: entropy >= 4.0 → redacted
        assert_eq!(
            outcome.text,
            "low entropy AAAAAAAA high entropy [CAVIARDER]"
        );
        assert_eq!(outcome.total(), 1);
    }

    #[test]
    fn redact_empty_input() {
        let rule = Rule {
            id: "test".into(),
            regex: Regex::new(r"[A-Z]+").unwrap(),
            entropy: None,
        };
        let redactor = Redactor::new(vec![rule], "[CAVIARDER]");
        let outcome = redactor.redact("");
        assert_eq!(outcome.text, "");
        assert_eq!(outcome.total(), 0);
    }

    #[test]
    fn redact_no_match() {
        let rule = Rule {
            id: "test".into(),
            regex: Regex::new(r"SECRET_\d+").unwrap(),
            entropy: None,
        };
        let redactor = Redactor::new(vec![rule], "[CAVIARDER]");
        let outcome = redactor.redact("just regular text here");
        assert_eq!(outcome.text, "just regular text here");
        assert_eq!(outcome.total(), 0);
    }

    // -----------------------------------------------------------------------
    // ML threshold filter tests
    // -----------------------------------------------------------------------

    #[test]
    fn redact_ml_threshold_disabled_by_default() {
        let rule = Rule {
            id: "test".into(),
            regex: Regex::new(r"\b[A-Za-z0-9]{10,}\b").unwrap(),
            entropy: None,
        };
        let redactor = Redactor::new(vec![rule], "[CAVIARDER]")
            .with_filename("file.env");
        let outcome = redactor.redact("x = abc123def456");
        // Default threshold 0.0 = disabled, so this should be redacted
        assert_eq!(outcome.text, "x = [CAVIARDER]");
        assert_eq!(outcome.total(), 1);
    }

    #[test]
    fn redact_ml_threshold_low_threshold_keeps_low_entropy() {
        // A threshold below the model's score for "hello" (~0.72) keeps the match.
        let rule = Rule {
            id: "test".into(),
            regex: Regex::new(r"\b[A-Za-z]{5,}\b").unwrap(),
            entropy: None,
        };
        let redactor = Redactor::new(vec![rule], "[CAVIARDER]")
            .with_ml_threshold(0.5)  // Below model's ~0.72 — keeps match
            .with_filename("file.py");
        let outcome = redactor.redact("x = hello");
        assert_eq!(outcome.text, "x = [CAVIARDER]");
        assert_eq!(outcome.total(), 1);
    }

    #[test]
    fn redact_ml_threshold_with_entropy() {
        // Both entropy filter and ML filter active
        let rule = Rule {
            id: "test".into(),
            regex: Regex::new(r"\b[A-Za-z0-9/+=-]{10,}\b").unwrap(),
            entropy: Some(3.0),  // Only matches with entropy >= 3.0
        };
        let redactor = Redactor::new(vec![rule], "[CAVIARDER]")
            .with_ml_threshold(0.5)
            .with_filename("file.env");
        // "AAAAABBBBB" has low entropy (~1.0) → blocked by entropy filter first
        // "sk-proj-A1b2C3d4E5f6" has high entropy → but ML score might be low
        let outcome = redactor.redact("x = AAAAABBBBB y = sk-proj-A1b2C3d4E5f6");
        // "AAAAABBBBB" has entropy ≈ 1.0 < 3.0 → dropped by entropy
        // "sk-proj-A1b2C3d4E5f6" has entropy > 3.0 → passed entropy, now ML decides.
        // With the XGBoost model, this should score ≥ 0.5 or not — we just check
        // that no more than 1 match passes both filters.
        assert!(outcome.total() <= 1);
    }
}

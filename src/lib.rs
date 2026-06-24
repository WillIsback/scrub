pub mod rules;

use regex::Regex;

pub struct Rule {
    pub id: String,
    pub regex: Regex,
    pub entropy: Option<f64>,
}

pub struct Redactor {
    rules: Vec<Rule>,
    placeholder: String,
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
        }
    }

    /// Apply every rule in order, returning the scrubbed text and per-rule counts.
    pub fn redact(&self, input: &str) -> Outcome {
        let mut text = input.to_string();
        let mut counts = Vec::new();

        for rule in &self.rules {
            let mut count = 0usize;

            if let Some(min_entropy) = rule.entropy {
                // Entropy threshold: only redact matches that meet the threshold.
                // If the regex has a capture group, only group 1 is checked for
                // entropy and redacted; the rest of the match is preserved.
                struct MatchEntry {
                    start: usize,
                    end: usize,
                    capture_start: Option<usize>,
                    capture_end: Option<usize>,
                }
                let mut matches: Vec<MatchEntry> = Vec::new();
                for caps in rule.regex.captures_iter(&text) {
                    let full = caps.get(0).unwrap();
                    let (check_str, capture_start, capture_end) = if caps.len() >= 2 {
                        let v = caps.get(1).unwrap();
                        (v.as_str(), Some(v.start()), Some(v.end()))
                    } else {
                        (full.as_str(), None, None)
                    };
                    if shannon_entropy(check_str) >= min_entropy {
                        matches.push(MatchEntry {
                            start: full.start(),
                            end: full.end(),
                            capture_start,
                            capture_end,
                        });
                    }
                }
                for entry in matches.into_iter().rev() {
                    let (rs, re) = match (entry.capture_start, entry.capture_end) {
                        (Some(s), Some(e)) => (s, e),
                        _ => (entry.start, entry.end),
                    };
                    text.replace_range(rs..re, &self.placeholder);
                    count += 1;
                }
            } else {
                // No entropy threshold: global replacement.
                // If the regex has a capture group, only group 1 is redacted
                // and the rest of the match is preserved as context.
                let replaced = rule.regex.replace_all(&text, |caps: &regex::Captures| {
                    count += 1;
                    if caps.len() >= 2 {
                        // Preserve context around the captured value
                        let full_match = caps.get(0).unwrap();
                        let value = caps.get(1).unwrap();
                        let before = &full_match.as_str()[..value.start() - full_match.start()];
                        let after = &full_match.as_str()[value.end() - full_match.start()..];
                        format!("{}{}{}", before, self.placeholder, after)
                    } else {
                        self.placeholder.as_str().to_owned()
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
}

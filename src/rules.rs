use crate::Rule;
use anyhow::Result;
use regex::RegexBuilder;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct GitleaksConfig {
    pub rules: Vec<GitleaksRule>,
}

#[derive(Deserialize)]
pub struct GitleaksRule {
    pub id: String,
    pub regex: Option<String>,
    pub entropy: Option<f64>,
}

/// Load rules from the embedded gitleaks.toml (compile-time).
pub fn load_embedded() -> Result<Vec<Rule>> {
    let toml_str = include_str!("../config/gitleaks.toml");
    load_from_str(toml_str)
}

/// Load rules from the embedded custom.toml (compile-time).
pub fn load_embedded_custom() -> Result<Vec<Rule>> {
    let toml_str = include_str!("../config/custom.toml");
    load_from_str(toml_str)
}

/// Load rules from a TOML string.
pub fn load_from_str(toml_str: &str) -> Result<Vec<Rule>> {
    let config: GitleaksConfig = toml::from_str(toml_str)?;
    let mut rules = Vec::new();
    for gr in config.rules {
        let regex_str = match gr.regex {
            Some(r) => r,
            None => {
                eprintln!("scrub: warning: skipping rule '{}' (no regex field)", gr.id);
                continue;
            }
        };
        match RegexBuilder::new(&regex_str)
            .size_limit(50 * (1 << 20)) // 50 MB limit for large gitleaks regexes
            .build()
        {
            Ok(regex) => {
                rules.push(Rule {
                    id: gr.id,
                    regex,
                    entropy: gr.entropy,
                });
            }
            Err(e) => {
                eprintln!("scrub: warning: skipping rule '{}': {}", gr.id, e);
            }
        }
    }
    Ok(rules)
}

/// Load rules from a TOML file at the given path.
pub fn load_from_path(path: &str) -> Result<Vec<Rule>> {
    let toml_str = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("failed to read rules file {}: {}", path, e))?;
    load_from_str(&toml_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_from_str_parses_valid_toml() {
        let toml = r#"
[[rules]]
id = "test-rule"
regex = '''AKIA[A-Z0-9]{16}'''
entropy = 3.0

[[rules]]
id = "simple-rule"
regex = '''sk-[A-Za-z0-9]+'''
"#;
        let rules = load_from_str(toml).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].id, "test-rule");
        assert_eq!(rules[0].entropy, Some(3.0));
        assert_eq!(rules[1].id, "simple-rule");
        assert_eq!(rules[1].entropy, None);
    }

    #[test]
    fn load_from_str_skips_bad_regex() {
        let toml = r#"
[[rules]]
id = "good-rule"
regex = '''[A-Z]+'''

[[rules]]
id = "bad-rule"
regex = '''[invalid'''
"#;
        let rules = load_from_str(toml).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "good-rule");
    }

    #[test]
    fn load_from_str_skips_rule_without_regex() {
        let toml = r#"
[[rules]]
id = "no-regex-rule"

[[rules]]
id = "good-rule"
regex = '''[A-Z]+'''
"#;
        let rules = load_from_str(toml).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "good-rule");
    }
}

use caviarder::{Redactor, Rule};
use criterion::{criterion_group, criterion_main, Criterion};
use regex::Regex;
use std::time::Duration;

/// Build a redactor with a single rule matching the given pattern,
/// and benchmark it against a matching input string.
fn bench_single_rule(c: &mut Criterion, rule_id: &str, pattern: &str, input: &str) {
    let rule = Rule {
        id: rule_id.into(),
        regex: Regex::new(pattern).unwrap(),
        entropy: None,
    };
    let redactor = Redactor::new(vec![rule], "[CAVIARDER]");

    let mut group = c.benchmark_group(format!("rule/{}", rule_id));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);
    group.bench_function("single_match", |b| {
        b.iter(|| redactor.redact(std::hint::black_box(input)))
    });
    group.finish();
}

fn bench_rules(c: &mut Criterion) {
    // AWS Access Key
    bench_single_rule(
        c,
        "aws-key",
        r"AKIA[A-Z0-9]{16}",
        "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
    );

    // Generic API Key (key=... pattern)
    bench_single_rule(
        c,
        "generic-api-key",
        r#"(?i)[\w.-]{0,50}?(?:key|secret|token)[\s'"]{0,3}(?:=|:)[\x60'"\s=]{0,5}([\w.=-]{10,150})"#,
        "api_key = sk-proj-abc123DEF456ghi789jkl012mnopqrXYZ",
    );

    // GitHub Token
    bench_single_rule(
        c,
        "github-token",
        r"\b((?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{36,255})\b",
        "token = ghp_abcdef1234567890abcdef1234567890abcdef12",
    );

    // Password field
    bench_single_rule(
        c,
        "password",
        r#"(?i)password[\s'"]{0,3}(?:=|:)[\x60'"\s]{0,5}([^\s'"]{8,})"#,
        "password = MyS3cureP@ssw0rd!",
    );

    // Slack Token (benchmark uses pattern, input is NOT a real token)
    bench_single_rule(
        c,
        "slack-token",
        r"(xox[bp])-[0-9]{10,13}-[a-zA-Z0-9\-]{20,}",
        "slack_token = xoxb-SLACK-BENCH-TOKEN-NOT-REAL-00000",
    );

    // JWT / eyJ...
    bench_single_rule(
        c,
        "jwt",
        r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        "token = eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.doeR3Jkf4kHwFJdb9o3Ml6A6zVn5W8xYQ2SsKJmNpQo",
    );

    // RSA Private Key block (multi-line)
    bench_single_rule(
        c,
        "private-key",
        r"-----BEGIN\s?(RSA|EC|DSA|OPENSSH|PGP)?\s?PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...",
    );
}

criterion_group!(benches, bench_rules);
criterion_main!(benches);

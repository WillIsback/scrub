use caviarder::{Redactor, Rule};
use criterion::{criterion_group, criterion_main, Criterion};
use regex::Regex;
use std::time::Duration;

/// Generate a text buffer of `target_size` bytes containing a mix of clean lines
/// and embedded secrets at the given `density` (0.0 – 1.0).
fn generate_input(target_size: usize, density: f64) -> String {
    let clean_line = "    host = localhost\n    port = 8080\n    debug = false\n";
    let secret_line = "    password = sk-proj-ABC123def456GHI789jkl012MNO345pqr678\n";
    let mut buf = String::with_capacity(target_size);
    while buf.len() < target_size {
        if buf.len() as f64 % 1024.0 / 1024.0 < density {
            buf.push_str(secret_line);
        } else {
            buf.push_str(clean_line);
        }
    }
    buf
}

fn build_redactor() -> Redactor {
    let rule = Rule {
        id: "bench-key".into(),
        regex: Regex::new(r"sk-proj-[A-Za-z0-9]{20,}").unwrap(),
        entropy: None,
    };
    Redactor::new(vec![rule], "[CAVIARDER]")
}

fn bench_throughput(c: &mut Criterion) {
    let redactor = build_redactor();
    let input = generate_input(10 * 1024 * 1024, 0.05); // 10 MB, 5% secrets

    let mut group = c.benchmark_group("throughput");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);
    group.throughput(criterion::Throughput::Bytes(input.len() as u64));
    group.bench_function("10_mb_5pct", |b| {
        b.iter(|| redactor.redact(std::hint::black_box(&input)))
    });
    group.finish();
}

criterion_group!(benches, bench_throughput);
criterion_main!(benches);

use std::process::Command;

fn cav_binary() -> &'static str {
    if cfg!(debug_assertions) {
        "target/debug/cav"
    } else {
        "target/release/cav"
    }
}

#[test]
fn test_stdin_no_secrets() {
    let output = Command::new(cav_binary())
        .arg("--check")
        .arg("--no-default")
        .arg("--rules")
        .arg("config/gitleaks.toml")
        .arg("tests/fixtures/edge.txt")
        .output()
        .expect("failed to execute cav");
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn test_file_redact_basic() {
    let output = Command::new(cav_binary())
        .arg("--no-default")
        .arg("--rules")
        .arg("config/gitleaks.toml")
        .arg("tests/fixtures/basic.txt")
        .output()
        .expect("failed to execute cav");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    // The fixture contains several known secret patterns.
    // At minimum, verify the output is shorter than the input (secrets were replaced).
    let input = std::fs::read_to_string("tests/fixtures/basic.txt").unwrap();
    assert!(stdout.len() < input.len(), "output should be shorter than input");
    // Verify the AWS key was redacted
    assert!(!stdout.contains("AKIAIOSFODNN7EXAMPLE"), "AKIA key should be redacted");
}

#[test]
fn test_stdin_pipe() {
    let mut child = Command::new(cav_binary())
        .arg("--no-default")
        .arg("--rules")
        .arg("config/gitleaks.toml")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("failed to spawn cav");
    use std::io::Write;
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"key AKIAIOSFODNN7EXAMPLE")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("[CAVIARDER]"));
    assert!(!stdout.contains("AKIAIOSFODNN7EXAMPLE"));
}

#[test]
fn test_check_mode_finds_secrets() {
    let output = Command::new(cav_binary())
        .arg("--check")
        .arg("--no-default")
        .arg("--rules")
        .arg("config/gitleaks.toml")
        .arg("tests/fixtures/basic.txt")
        .output()
        .expect("failed to execute cav");
    // --check with secrets should exit 1
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("found"), "should report findings on stderr");
}

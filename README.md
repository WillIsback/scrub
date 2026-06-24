# caviarder

[![Crates.io](https://img.shields.io/crates/v/caviarder)](https://crates.io/crates/caviarder)

**Redact secrets from text using gitleaks detection rules.**

`caviarder` (French for "to redact") is a fast Rust CLI that reads text from a file or stdin and replaces
detected secrets (API keys, tokens, passwords) with a placeholder. It uses
[gitleaks](https://github.com/gitleaks/gitleaks)' community-maintained rule
set — 220+ patterns — embedded at compile time.

## Why

Sharing config files, logs, or debug output often leaks secrets without you
noticing. `gitleaks` catches them during CI but isn't designed for ad-hoc
scrubbing. `caviarder` fills that gap:

- **Pipe anything**: `cat file | cav` — works like `cat` but safe.
- **Check before commit**: `cav --check file` — exit 1 if secrets found.
- **Zero config**: 220+ rules baked in. No setup, no server, no dependencies.

## Installation

### From source

```bash
git clone https://github.com/WillIsback/caviarder.git
cd scrub
cargo build --release
cp target/release/cav ~/.local/bin/
```

Requires Rust 1.70+.

### From crates.io

```bash
cargo install caviarder
```

## Usage

### Redact a file

```bash
cav config.yml
# → prints redacted version to stdout
```

### Pipe from any command

```bash
cat config.yml | cav
echo "$API_KEY" | cav
env | cav
```

### Save redacted output

```bash
cav config.yml -o safe.yml
# or
cat config.yml | cav > safe.yml
```

### Check for secrets (exit 1 if found)

```bash
cav --check deploy.sh
echo $?   # 1 if secrets were found
```

### Scan with detailed stats

```bash
cav --stats entrypoint.sh
# → stderr: per-rule redaction counts
```

### List all detection rules

```bash
cav --list-rules
```

### Add custom rules

Create a TOML file with your own patterns:

```toml
[[rules]]
id = "my-app-token"
regex = '''myapp-[A-Za-z0-9]{32}'''
entropy = 3.0
```

Then use it alongside the built-in rules:

```bash
cav --rules my-rules.toml config.yml
```

## How it works

| Layer | File | Role |
|-------|------|------|
| Rules | `config/gitleaks.toml` | 220+ regex patterns from gitleaks (embedded at compile time) |
| Engine | `lib.rs` | Applies rules in order, optional Shannon entropy filtering, capture-group preservation |
| CLI | `main.rs` | clap argument parsing, stdin/file I/O, exit codes |

The pipeline is: **text → regex scan → entropy check → replacement**.

If a rule has an entropy threshold, only matches whose Shannon entropy meets
the threshold are redacted — reducing false positives on short or repetitive
strings.

If a rule defines a capture group `( )`, only the captured portion is redacted
and the rest of the match is preserved as context:

```
--api-key "sk-abc123DEF456ghi789jkl012mnopqrXYZ"
→ --api-key "[REDACTED]"
```

## How it differs from gitleaks

| | `gitleaks` | `caviarder` |
|---|-----------|------------|
| Purpose | CI/CD scanning | Ad-hoc redaction |
| Output | JSON/CSV report | Redacted text on stdout |
| Installation | Go binary + config | Single Rust binary, no deps |
| Rule updates | Manual config pull | Recompile to refresh |

## License

MIT

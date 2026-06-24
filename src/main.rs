use std::io::Read;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use caviarder::rules;
use caviarder::{Redactor, Rule};

/// Redact secrets and PII from text using gitleaks detection rules.
#[derive(Parser, Debug)]
#[command(
    name = "cav",
    version,
    about,
    after_help = "EXAMPLES:\n  \
        cat config.yml | cav               Pipe from cat into cav\n  \
        cav config.yml                     Redact a file (same as cat | cav)\n  \
        echo \"key=abc123\" | cav            Redact piped text\n  \
        cav --check config.yml             Check for secrets (exit 1 if found)\n  \
        cav config.yml -o clean.yml        Save redacted output to a file\n  \
        cav --stats config.yml             Show per-rule redaction counts\n  \
        cav --list-rules                   List all loaded detection rules\n  \
        cav --rules my-rules.toml file     Use additional custom rules"
)]
struct Cli {
    /// Input file (default: stdin)
    input: Option<PathBuf>,

    /// Write redacted text here (default: stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Replacement string for redacted values (default: "[CAVIARDER]")
    #[arg(short, long, default_value = "[CAVIARDER]")]
    placeholder: String,

    /// Scan only: exit 1 if secrets found, no output written
    #[arg(short, long)]
    check: bool,

    /// Path to custom gitleaks.toml (default: embedded rules)
    #[arg(short, long)]
    rules: Option<PathBuf>,

    /// Print per-rule redaction counts to stderr
    #[arg(short, long)]
    stats: bool,

    /// List all compiled rule names and exit
    #[arg(long)]
    list_rules: bool,

    /// Don't load embedded default rules (use with --rules)
    #[arg(long)]
    no_default: bool,

    /// ML false-positive filter threshold (0.0 = disabled).
    /// Higher values suppress more matches. Try 0.5 for balanced filtering.
    #[arg(long, default_value = "0.0")]
    ml_threshold: f64,

    /// Filename for ML feature computation (file extension matters).
    /// Only used with --ml-threshold.
    #[arg(long, default_value = "")]
    filename: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // --list-rules: just print rule names and exit
    if cli.list_rules {
        let rules = load_rules(&cli)?;
        for rule in &rules {
            println!("{}", rule.id);
        }
        return Ok(());
    }

    // --check and --output are mutually exclusive
    if cli.check && cli.output.is_some() {
        anyhow::bail!("--check and --output cannot be used together");
    }

    // Read input
    let input = match &cli.input {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?,
        None => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .context("failed to read stdin")?;
            buf
        }
    };

    // Build redactor
    let rules = load_rules(&cli)?;
    if rules.is_empty() {
        anyhow::bail!("no rules loaded (use --rules or check config/gitleaks.toml)");
    }
    let mut redactor = Redactor::new(rules, &cli.placeholder);
    if cli.ml_threshold > 0.0 {
        redactor = redactor.with_ml_threshold(cli.ml_threshold);
        if !cli.filename.is_empty() {
            redactor = redactor.with_filename(&cli.filename);
        }
    }
    let outcome = redactor.redact(&input);

    // --check mode
    if cli.check {
        for (rule_id, count) in &outcome.counts {
            eprintln!("{}: {}", rule_id, count);
        }
        let total = outcome.total();
        if total > 0 {
            eprintln!("cav: found {} potential secret(s)", total);
            std::process::exit(1);
        }
        eprintln!("cav: no secrets found");
        return Ok(());
    }

    // Write output
    match &cli.output {
        Some(path) => std::fs::write(path, &outcome.text)
            .with_context(|| format!("failed to write {}", path.display()))?,
        None => {
            std::io::stdout()
                .write_all(outcome.text.as_bytes())
                .context("failed to write stdout")?;
        }
    }

    // --stats
    if cli.stats {
        for (rule_id, count) in &outcome.counts {
            eprintln!("{}: {}", rule_id, count);
        }
        eprintln!("cav: {} redaction(s)", outcome.total());
    }

    Ok(())
}

/// Load rules based on CLI flags. Merges embedded and custom rules unless --no-default.
fn load_rules(cli: &Cli) -> Result<Vec<Rule>> {
    let mut rules = Vec::new();

    if !cli.no_default {
        let embedded = rules::load_embedded().context("failed to load embedded rules")?;
        rules.extend(embedded);
        let custom =
            rules::load_embedded_custom().context("failed to load embedded custom rules")?;
        rules.extend(custom);
    }

    if let Some(rules_path) = &cli.rules {
        let custom = rules::load_from_path(&rules_path.to_string_lossy())?;
        rules.extend(custom);
    }

    if rules.is_empty() && cli.no_default && cli.rules.is_none() {
        anyhow::bail!("--no-default requires --rules to specify a rules file");
    }

    Ok(rules)
}

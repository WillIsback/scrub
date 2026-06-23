use std::io::Read;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use scrub::rules;
use scrub::{Redactor, Rule};

/// Redact secrets and PII from text using gitleaks detection rules.
#[derive(Parser, Debug)]
#[command(name = "scrub", version, about)]
struct Cli {
    /// Input file (default: stdin)
    input: Option<PathBuf>,

    /// Write redacted text here (default: stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Replacement string for redacted values
    #[arg(short, long, default_value = "[REDACTED]")]
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
    let redactor = Redactor::new(rules, &cli.placeholder);
    let outcome = redactor.redact(&input);

    // --check mode
    if cli.check {
        for (rule_id, count) in &outcome.counts {
            eprintln!("{}: {}", rule_id, count);
        }
        let total = outcome.total();
        if total > 0 {
            eprintln!("scrub: found {} potential secret(s)", total);
            std::process::exit(1);
        }
        eprintln!("scrub: no secrets found");
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
        eprintln!("scrub: {} redaction(s)", outcome.total());
    }

    Ok(())
}

/// Load rules based on CLI flags. Merges embedded and custom rules unless --no-default.
fn load_rules(cli: &Cli) -> Result<Vec<Rule>> {
    let mut rules = Vec::new();

    if !cli.no_default {
        let embedded = rules::load_embedded().context("failed to load embedded rules")?;
        rules.extend(embedded);
        let custom = rules::load_embedded_custom().context("failed to load embedded custom rules")?;
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

use std::env;
use std::fs;
use std::path::PathBuf;

use serde_json::json;
use theorem_bayes::SignedPriorPack;

const PRIOR_PACK_PATH_ENV: &str = "THEOREM_PRIOR_PACK_PATH";
const PRIOR_PACK_SECRET_ENV: &str = "THEOREM_PRIOR_PACK_VERIFICATION_SECRET";
const PRIOR_PACK_TASK_FAMILY_ENV: &str = "THEOREM_PRIOR_PACK_TASK_FAMILY";

#[derive(Debug)]
struct Args {
    command: String,
    path: PathBuf,
    secret: String,
    task_family: String,
    read_day_index: i64,
    half_life_days: f64,
    prior_strength: f64,
}

fn main() {
    if let Err(error) = run(env::args().skip(1)) {
        eprintln!("theorem-prior-pack: {error}");
        std::process::exit(1);
    }
}

fn run<I>(args: I) -> Result<(), String>
where
    I: IntoIterator<Item = String>,
{
    let raw_args = args.into_iter().collect::<Vec<_>>();
    if raw_args
        .iter()
        .any(|arg| matches!(arg.as_str(), "help" | "-h" | "--help"))
    {
        print_usage();
        return Ok(());
    }
    let args = parse_args(raw_args)?;
    let pack: SignedPriorPack = serde_json::from_slice(
        &fs::read(&args.path).map_err(|error| format!("read {}: {error}", args.path.display()))?,
    )
    .map_err(|error| format!("parse {}: {error}", args.path.display()))?;
    pack.verify(args.secret.as_bytes())
        .map_err(|error| format!("verify {}: {error}", args.path.display()))?;
    let suggestions = pack.affordance_suggestions(
        &args.task_family,
        args.read_day_index,
        args.half_life_days,
        args.prior_strength,
    );
    let output = json!({
        "status": "verified",
        "path": args.path,
        "pack_id": pack.pack_id(),
        "graph_version": pack.payload.graph_version,
        "tenant_count": pack.payload.tenant_count,
        "source_bin_count": pack.payload.source_bin_count,
        "rollback_parent_pack_id": pack.rollback_parent_pack_id(),
        "task_family": args.task_family,
        "suggestions": suggestions,
    });

    match args.command.as_str() {
        "status" | "suggest" => {
            println!(
                "{}",
                serde_json::to_string_pretty(&output)
                    .map_err(|error| format!("serialize status: {error}"))?
            );
            Ok(())
        }
        other => Err(format!("unknown command {other}")),
    }
}

fn parse_args<I>(args: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut iter = args.into_iter();
    let mut command = "status".to_string();
    let mut path = env::var(PRIOR_PACK_PATH_ENV).ok().map(PathBuf::from);
    let mut secret = env::var(PRIOR_PACK_SECRET_ENV).ok();
    let mut task_family = env::var(PRIOR_PACK_TASK_FAMILY_ENV).unwrap_or_else(|_| "coding".into());
    let mut read_day_index = 0_i64;
    let mut half_life_days = 7.0_f64;
    let mut prior_strength = 1.0_f64;

    if let Some(first) = iter.next() {
        if first.starts_with('-') {
            parse_flag(
                first,
                &mut iter,
                &mut path,
                &mut secret,
                &mut task_family,
                &mut read_day_index,
                &mut half_life_days,
                &mut prior_strength,
            )?;
        } else {
            command = first;
        }
    }

    while let Some(flag) = iter.next() {
        parse_flag(
            flag,
            &mut iter,
            &mut path,
            &mut secret,
            &mut task_family,
            &mut read_day_index,
            &mut half_life_days,
            &mut prior_strength,
        )?;
    }

    Ok(Args {
        command,
        path: path.ok_or_else(|| {
            format!("missing --path or {PRIOR_PACK_PATH_ENV} for signed prior pack")
        })?,
        secret: secret.ok_or_else(|| {
            format!("missing --secret or {PRIOR_PACK_SECRET_ENV} for prior pack verification")
        })?,
        task_family,
        read_day_index,
        half_life_days,
        prior_strength,
    })
}

fn parse_flag<I>(
    flag: String,
    iter: &mut I,
    path: &mut Option<PathBuf>,
    secret: &mut Option<String>,
    task_family: &mut String,
    read_day_index: &mut i64,
    half_life_days: &mut f64,
    prior_strength: &mut f64,
) -> Result<(), String>
where
    I: Iterator<Item = String>,
{
    match flag.as_str() {
        "--path" => *path = Some(PathBuf::from(next_value(iter, "--path")?)),
        "--secret" => *secret = Some(next_value(iter, "--secret")?),
        "--task-family" => *task_family = next_value(iter, "--task-family")?,
        "--read-day" => {
            *read_day_index = parse_value(next_value(iter, "--read-day")?, "--read-day")?
        }
        "--half-life-days" => {
            *half_life_days =
                parse_value(next_value(iter, "--half-life-days")?, "--half-life-days")?
        }
        "--prior-strength" => {
            *prior_strength =
                parse_value(next_value(iter, "--prior-strength")?, "--prior-strength")?
        }
        "help" | "-h" | "--help" => {}
        other => return Err(format!("unknown flag {other}")),
    }
    Ok(())
}

fn next_value<I>(iter: &mut I, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    iter.next()
        .ok_or_else(|| format!("missing value for {flag}"))
}

fn parse_value<T>(raw: String, flag: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    raw.parse()
        .map_err(|_| format!("invalid value for {flag}: {raw}"))
}

fn print_usage() {
    println!(
        "usage: theorem-prior-pack <status|suggest> --path <signed-pack.json> --secret <verification-secret> [--task-family coding] [--read-day N] [--half-life-days N] [--prior-strength N]"
    );
}

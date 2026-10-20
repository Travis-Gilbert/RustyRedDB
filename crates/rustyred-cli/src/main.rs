mod output;

use std::collections::BTreeMap;
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use rustyred::Database;
use serde_json::Value;

use output::fail;

fn main() -> ExitCode {
    let mut args = env::args().skip(1).collect::<Vec<_>>();
    if args.len() < 2 {
        eprintln!("usage: rustyred <create|query|inspect|verify> PATH");
        return ExitCode::from(2);
    }
    let command = args.remove(0);
    let path = PathBuf::from(args.remove(0));
    match command.as_str() {
        "create" => create(&path),
        "query" => query(&path, &args),
        "inspect" => inspect(&path, &args),
        "verify" => verify(&path, &args),
        _ => {
            eprintln!("unknown command");
            ExitCode::from(2)
        }
    }
}

fn create(path: &std::path::Path) -> ExitCode {
    match Database::create(path) {
        Ok(db) => {
            let inspection = match db.inspect() {
                Ok(value) => value,
                Err(err) => return fail(&err, 1),
            };
            println!(
                "{}",
                serde_json::json!({
                    "ok": true,
                    "path": inspection.path,
                    "database_id": inspection.database_id,
                })
            );
            ExitCode::SUCCESS
        }
        Err(err) => fail(&err, 3),
    }
}

fn query(path: &std::path::Path, args: &[String]) -> ExitCode {
    let Some(cypher) = flag(args, "--cypher") else {
        eprintln!("query requires --cypher");
        return ExitCode::from(2);
    };
    let params = match flag(args, "--params-json") {
        None => BTreeMap::new(),
        Some(text) => match serde_json::from_str::<Value>(text) {
            Ok(Value::Object(map)) => map.into_iter().collect(),
            Ok(_) => {
                eprintln!("--params-json must be a JSON object");
                return ExitCode::from(2);
            }
            Err(err) => {
                eprintln!("--params-json: {err}");
                return ExitCode::from(2);
            }
        },
    };
    let db = match Database::open_read_only(path) {
        Ok(db) => db,
        Err(err) => return fail(&err, 3),
    };
    match db.query(cypher, &params) {
        Ok(result) => {
            println!(
                "{}",
                serde_json::json!({
                    "ok": true,
                    "columns": result.columns,
                    "rows": result.rows,
                    "truncated": result.truncated,
                })
            );
            ExitCode::SUCCESS
        }
        Err(err) => fail(&err, 4),
    }
}

fn inspect(path: &std::path::Path, args: &[String]) -> ExitCode {
    if !args.is_empty() && !args.iter().any(|arg| arg == "--json") {
        eprintln!("inspect accepts only --json");
        return ExitCode::from(2);
    }
    let db = match Database::open_read_only(path) {
        Ok(db) => db,
        Err(err) => return fail(&err, 3),
    };
    match db.inspect() {
        Ok(inspection) => {
            println!(
                "{}",
                serde_json::json!({
                    "ok": true,
                    "path": inspection.path,
                    "database_id": inspection.database_id,
                    "writable": inspection.writable,
                    "version": inspection.version,
                    "nodes": inspection.nodes,
                    "edges": inspection.edges,
                })
            );
            ExitCode::SUCCESS
        }
        Err(err) => fail(&err, 1),
    }
}

fn verify(path: &std::path::Path, args: &[String]) -> ExitCode {
    if !args.is_empty() && !args.iter().any(|arg| arg == "--json") {
        eprintln!("verify accepts only --json");
        return ExitCode::from(2);
    }
    let db = match Database::open_read_only(path) {
        Ok(db) => db,
        Err(err) => return fail(&err, 3),
    };
    match db.verify() {
        Ok(report) => {
            println!("{}", serde_json::json!({"ok": report.ok, "report": report}));
            if report.ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(5)
            }
        }
        Err(err) => fail(&err, 1),
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(String::as_str)
}

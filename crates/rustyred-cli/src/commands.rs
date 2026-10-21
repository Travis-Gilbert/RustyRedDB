use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Subcommand;
use rustyred::Database;
use serde_json::Value;

use crate::output::fail;

#[derive(Subcommand)]
pub enum Command {
    /// Create an empty database.
    Create {
        #[arg(value_name = "PATH")]
        path: PathBuf,
    },
    /// Run a read-only Cypher query.
    Query {
        #[arg(value_name = "PATH")]
        path: PathBuf,
        #[arg(long, value_name = "QUERY", value_parser = clap::builder::NonEmptyStringValueParser::new())]
        cypher: String,
        /// Query parameters as a JSON object.
        #[arg(long, value_name = "JSON", default_value = "{}", value_parser = parse_params)]
        params_json: BTreeMap<String, Value>,
    },
    /// Inspect database metadata without writing to the database.
    Inspect {
        #[arg(value_name = "PATH")]
        path: PathBuf,
        /// Emit JSON (the default output format).
        #[arg(long)]
        json: bool,
    },
    /// Verify database integrity without writing to the database.
    Verify {
        #[arg(value_name = "PATH")]
        path: PathBuf,
        /// Emit JSON (the default output format).
        #[arg(long)]
        json: bool,
    },
}

impl Command {
    pub fn run(self) -> ExitCode {
        match self {
            Self::Create { path } => create(&path),
            Self::Query {
                path,
                cypher,
                params_json,
            } => query(&path, &cypher, &params_json),
            Self::Inspect { path, json: _ } => inspect(&path),
            Self::Verify { path, json: _ } => verify(&path),
        }
    }
}

fn parse_params(text: &str) -> Result<BTreeMap<String, Value>, String> {
    match serde_json::from_str(text).map_err(|err| err.to_string())? {
        Value::Object(map) => Ok(map.into_iter().collect()),
        _ => Err("must be a JSON object".into()),
    }
}

fn create(path: &Path) -> ExitCode {
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

fn query(path: &Path, cypher: &str, params: &BTreeMap<String, Value>) -> ExitCode {
    let db = match Database::open_read_only(path) {
        Ok(db) => db,
        Err(err) => return fail(&err, 3),
    };
    match db.query(cypher, params) {
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

fn inspect(path: &Path) -> ExitCode {
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

fn verify(path: &Path) -> ExitCode {
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

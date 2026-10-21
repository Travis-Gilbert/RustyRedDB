use std::process::ExitCode;

use rustyred::{Error, ErrorKind};

pub fn fail(err: &Error, status: u8) -> ExitCode {
    eprintln!("{err}");
    println!(
        "{}",
        serde_json::to_string(&err.body()).expect("error body")
    );
    if err.kind == ErrorKind::CommitOutcomeUnknown {
        eprintln!("commit outcome is unknown; close and reopen the database, then reconcile before retrying");
        ExitCode::from(6)
    } else {
        ExitCode::from(status)
    }
}

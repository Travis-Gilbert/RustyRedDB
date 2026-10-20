use std::process::ExitCode;

use rustyred::Error;

pub fn fail(err: &Error, status: u8) -> ExitCode {
    eprintln!("{err}");
    println!(
        "{}",
        serde_json::to_string(&err.body()).expect("error body")
    );
    ExitCode::from(status)
}

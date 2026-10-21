mod commands;
mod output;

use std::process::ExitCode;

use clap::Parser;

#[derive(Parser)]
#[command(name = "rustyred", version, about, disable_help_subcommand = true)]
struct Cli {
    #[command(subcommand)]
    command: commands::Command,
}

fn main() -> ExitCode {
    Cli::parse().command.run()
}

//! Explicit local read-only operator. Output must be a new file.
use rustyred_thg_core::semantic_inventory::{inventory_semantic_file, InventoryFileKind};
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 4 {
        return Err(
            "usage: semantic_inventory <bare-journal|rr> <source-file> <new-output.json>".into(),
        );
    }
    let kind = match args[1].to_str() {
        Some("bare-journal") => InventoryFileKind::BareJournal,
        Some("rr") => InventoryFileKind::RustyRed,
        _ => return Err("file kind must be bare-journal or rr".into()),
    };
    let source = Path::new(&args[2]);
    let output = Path::new(&args[3]);
    // Complete the scan before creating anything. Never open source as writer.
    let inventory = inventory_semantic_file(source, kind)?;
    let fingerprint = inventory.fingerprint()?;
    let bytes = serde_json::to_vec_pretty(&serde_json::json!({
        "schema":"theorem.semantic.physical-inventory-envelope/1",
        "fingerprint":fingerprint,"inventory":inventory
    }))?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(output)?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    println!("inventory written; fingerprint={fingerprint}");
    Ok(())
}

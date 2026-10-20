use rustyred::{Database, ErrorKind, OpenOptions};

#[test]
fn a_corrupt_header_is_a_bounded_open_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.rr");
    let db = Database::create_with(&path, &OpenOptions::new().device_name("writer")).unwrap();
    drop(db);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[..8].copy_from_slice(b"NOTREDDB");
    std::fs::write(&path, &bytes).unwrap();
    let err = match Database::open(&path) {
        Ok(_) => panic!("expected a corrupt header to fail"),
        Err(error) => error,
    };
    assert_eq!(err.kind, ErrorKind::Storage);
    assert!(!err.code.is_empty());
    assert!(!err.message.is_empty());
}

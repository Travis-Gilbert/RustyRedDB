use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_GRAPHBLAS");
    println!("cargo:rerun-if-env-changed=RUSTYRED_GRAPHBLAS_PREFIX");
    if env::var_os("CARGO_FEATURE_GRAPHBLAS").is_none() {
        return;
    }

    let prefix = env::var_os("RUSTYRED_GRAPHBLAS_PREFIX")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env::var_os("HOME").expect(
                "HOME is required when graphblas is enabled without RUSTYRED_GRAPHBLAS_PREFIX",
            ))
            .join(".cache/rustyred-thg-graphblas/install")
        });
    let lib_dir = prefix.join("lib");
    assert!(
        lib_dir.join(dylib_name("graphblas")).exists(),
        "GraphBLAS native library is missing from {}",
        lib_dir.display()
    );

    #[cfg(target_os = "macos")]
    println!(
        "cargo:rustc-link-arg-tests=-Wl,-rpath,{}",
        lib_dir.display()
    );
    #[cfg(target_os = "linux")]
    println!(
        "cargo:rustc-link-arg-tests=-Wl,-rpath,{}",
        lib_dir.display()
    );
}

#[cfg(target_os = "macos")]
fn dylib_name(name: &str) -> String {
    format!("lib{name}.dylib")
}

#[cfg(target_os = "linux")]
fn dylib_name(name: &str) -> String {
    format!("lib{name}.so")
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn dylib_name(name: &str) -> String {
    format!("{name}.unsupported")
}

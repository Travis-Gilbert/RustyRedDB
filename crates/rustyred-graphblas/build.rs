use std::env;
use std::path::PathBuf;
use std::process::Command;

// Wires SuiteSparse:GraphBLAS + LAGraph into the crate: locates the install
// prefix, emits link directives + an rpath, and generates FFI bindings over
// the public C API with bindgen.
//
// Prefix resolution:
//   1. RUSTYRED_GRAPHBLAS_PREFIX (a prebuilt install: lib/ + include/suitesparse/)
//   2. (next D1 step) a vendored build driven from this build script, mirroring
//      the FalkorDB vendoring pattern named in the handoff.
fn main() {
    let prefix = resolve_prefix();
    let lib_dir = prefix.join("lib");
    assert!(
        lib_dir.exists(),
        "GraphBLAS prefix has no lib/ dir: {}",
        lib_dir.display()
    );

    // Link the shared libraries; embed an rpath so test/bench binaries locate
    // them at runtime without DYLD_* environment variables.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=graphblas");
    println!("cargo:rustc-link-lib=dylib=lagraph");
    // LAGraphX (experimental) provides k-truss; the rest of the algorithms are
    // in stable LAGraph.
    println!("cargo:rustc-link-lib=dylib=lagraphx");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTYRED_GRAPHBLAS_PREFIX");

    // SuiteSparse installs headers under include/suitesparse; keep include/ too
    // for non-SuiteSparse layouts.
    let inc_suitesparse = prefix.join("include").join("suitesparse");
    let inc_root = prefix.join("include");

    let mut builder = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", inc_suitesparse.display()))
        .clang_arg(format!("-I{}", inc_root.display()))
        .layout_tests(false)
        // C header comments (e.g. LAGraph Matrix-Market docs with `%%` lines and
        // fenced blocks) would otherwise become Rust doc-comments and fail as
        // doctests. We don't surface the FFI as public docs, so drop them.
        .generate_comments(false)
        // Public API surface only (no GB_* internals).
        .allowlist_function("(GrB|GxB|LAGr|LAGraph)_.*")
        .allowlist_type("(GrB|GxB|LAGr|LAGraph)_.*")
        .allowlist_var("(GrB|GxB|LAGr|LAGraph|LG)_.*");

    // macOS: point libclang at the active SDK so <stddef.h>/<stdint.h> resolve.
    if let Some(sdk) = macos_sdk_path() {
        builder = builder.clang_arg(format!("-isysroot{sdk}"));
    }

    let bindings = builder
        .generate()
        .expect("failed to generate GraphBLAS/LAGraph bindings");
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}

fn resolve_prefix() -> PathBuf {
    // 1. Prebuilt install (dev fast path / CI cache).
    if let Ok(p) = env::var("RUSTYRED_GRAPHBLAS_PREFIX") {
        let pb = PathBuf::from(&p);
        assert!(
            pb.join("lib").exists(),
            "RUSTYRED_GRAPHBLAS_PREFIX={p} but {p}/lib is missing; build GraphBLAS + LAGraph first"
        );
        return pb;
    }

    // 2. Vendored build (FalkorDB pattern): graphblas.sh builds + installs into a
    //    stable, space-free per-user cache, skipped if already present. We must
    //    NOT build under OUT_DIR: the workspace path may contain spaces, which
    //    breaks LAGraph's (unquoted) include flags. $HOME/.cache is space-free.
    let home = env::var("HOME").expect("HOME not set and RUSTYRED_GRAPHBLAS_PREFIX unset");
    let prefix = PathBuf::from(home).join(".cache/rustyred-thg-graphblas/install");
    let built = prefix.join("lib/libgraphblas.dylib").exists();
    if !built {
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("graphblas.sh");
        println!(
            "cargo:warning=rustyred-thg-graphblas: building vendored GraphBLAS + LAGraph into {} (set RUSTYRED_GRAPHBLAS_PREFIX to skip)",
            prefix.display()
        );
        let status = Command::new("bash")
            .arg(&script)
            .arg(&prefix)
            .status()
            .expect("failed to spawn graphblas.sh");
        assert!(status.success(), "graphblas.sh failed: {status}");
    }
    prefix
}

fn macos_sdk_path() -> Option<String> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = Command::new("xcrun").arg("--show-sdk-path").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (!s.is_empty()).then_some(s)
}

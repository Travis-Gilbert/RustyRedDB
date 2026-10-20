# turbovec 1.0.0 — Theorem vendor patch (32-bit / wasm32)

This directory is upstream `turbovec` 1.0.0, vendored so that 32-bit targets
compile and run. It is wired in through `[patch.crates-io]` in
`rustyredcore_THG/Cargo.toml`, so both `rustyred-thg-core` and `rustyred-web`
resolve to it instead of the registry copy.

- **Upstream source**: crates.io `turbovec` 1.0.0
  (`registry+https://github.com/rust-lang/crates.io-index`), checksum
  `1131bbaacc5d3c093f09c2e3b470f03bbd218b1a0843140ae6a71a8c692b9ece`
- **Upstream repository**: <https://github.com/RyanCodrai/turbovec>
- **Why**: `SPEC-THEOREMWEB-PWA-HTML-CANVAS-1.1` §33 requires the RustyRed
  projection to run in the browser, and §36 makes that store RustyRed's own —
  the vector capability cannot be excluded or replaced. Upstream refuses to
  compile below 64-bit pointers, and `wasm32` has 32-bit pointers.

## The delta from upstream: exactly two source edits

Everything else in this tree — `src/`, `tests/`, `examples/`, `README.md`,
`LICENSE`, `Cargo.toml.orig` — is upstream, byte for byte.

### 1. `src/lib.rs` — the 64-bit refusal is removed

Upstream carries:

```rust
#[cfg(not(target_pointer_width = "64"))]
compile_error!("turbovec requires a 64-bit target (target_pointer_width = \"64\")");
```

It is replaced by a comment recording the audit below. No behaviour changes on
any target.

### 2. `src/io.rs` — a positional-read arm for neither-Unix-nor-Windows targets

Upstream declares `read_exact_at` only under `#[cfg(unix)]` and
`#[cfg(windows)]`, so on `wasm32` the function did not exist and every caller
in `io.rs`, `io_v7.rs` and `convert.rs` failed with *cannot find function
`read_exact_at`* — the six errors in the Phase G blocker measurement. This
vendor adds:

```rust
#[cfg(not(any(unix, windows)))]
pub(crate) fn read_exact_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = f;
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(buf)
}
```

`&File` implements both `Read` and `Seek`, so this is the same seek-then-read
the Windows arm performs with `seek_read`, against the same `UnexpectedEof`
failure mode.

## Why removing the refusal is sound: the checked-arithmetic audit

Upstream's comment asks for "a dedicated checked-arithmetic pass first". This
is that pass. The overflow class it names — a `usize` size computation
overflowing at 32 bits and indexing out of bounds — was traced to every place
an untrusted count can reach a size computation:

| Site | Guard |
|---|---|
| `convert::read_legacy`, the u64 row count | `usize::try_from(u64)`, erroring `n_vectors does not fit this platform's usize` — it cannot truncate |
| `convert::read_legacy`, before any size | place bound `n_vectors.saturating_mul(5) > bytes.len()` rejects a header describing more rows than the bytes could hold |
| `TurboQuantIndex::from_parts` | `checked_mul` chain returning `FromPartsError::PackedCodesSizeOverflow` |
| `io_v7` header walk | `checked_mul` / `checked_add` throughout |
| every other parsed count | a `u32` field, and `u32 as usize` is lossless at 32 bits |
| `pack::blocked_geometry` and friends | only ever re-derive a length that already exists as a populated buffer (the validated `packed_codes`), so the product cannot exceed the `usize` already holding it; `MAX_DIM` = 16384 bounds the `dim` term on every target |

The one `u64 as usize` truncation in the whole crate is inside
`#[cfg(test)] fn reseal_for_test`, and it is followed by `checked_mul`. The
SIMD kernels are `cfg`-gated per architecture, so a 32-bit build takes the
scalar paths by construction rather than through a runtime check.

## Verification

Command that exercises the real target:

```sh
cargo +nightly check --lib --target wasm32-unknown-unknown
```

What was measured, and the results:

- **`wasm32-unknown-unknown` and `wasm32-wasip1` both compile.** The only
  other change needed in the graph was enabling the `getrandom` 0.2 `js`
  backend (declared target-specifically in `rustyred-thg-core`).
- **`to_bytes` output is byte-identical across pointer widths.** An index
  built from a fixed-seed corpus at `dim=64, n=300, bit_width=4` hashes to
  `fnv1a 948b920290979fac` (126227 bytes) on 64-bit native and on 32-bit
  `wasm32-wasip1`; the `TurboQuantIndex` case at `dim=128, n=200,
  bit_width=3` hashes to `f38d4c3d24f2e510` (177187 bytes) on both. Top-5
  nearest ids are identical.
- **The crate's own suite runs on 32-bit with `overflow-checks` on** (the dev
  profile), which turns any `usize` wraparound into a panic. The lib suite
  gives 83 pass / 20 fail, and every failure is environmental rather than
  arithmetic — the three named classes are `std::thread::spawn` (WASI without
  threads), `std::panic::catch_unwind` (wasm is `panic = abort`, so the
  crate's own panic-safety tests cannot run), and `std::env::temp_dir()`
  (unsupported by WASI std). Integration tests that do not touch the
  filesystem pass: `batch_matches_single`, `codebook_determinism`,
  `query_scale_invariance`, `swap_remove`.
- `rayon` is **not** a blocker: `rayon::current_num_threads()` reports 1 on
  wasm, `par_iter`/`par_chunks_mut` run correctly, and the pool never spawns a
  thread. Several `encode`/`pack`/`search` paths use rayon, so this mattered.

`tests/wasm32_parity.rs` in this tree is the repo-authored oracle for the
claims above; it pins the digests, so it fails on any target whose encoding
diverges from the 64-bit reference. It runs natively as part of
`cargo test -p turbovec`, and on 32-bit via a `wasm32-wasip1` cargo runner.

## Exit path

The two edits are upstream-shaped and small: a target gate removed, and one
missing `cfg` arm added. If upstream lands a 32-bit path, delete this
directory, drop the `[patch.crates-io]` entry, and re-pin the registry
version. The parity oracle is worth keeping either way.

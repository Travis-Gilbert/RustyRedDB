//! 32-bit parity oracle for the Theorem vendor patch.
//!
//! `THEOREM-PATCH.md` lifts upstream's refusal to compile below 64-bit
//! pointers. The claim that licenses that is not "it compiles" but "it
//! produces the same index": pointer width changes `usize` arithmetic, and a
//! 32-bit `usize` is the only thing that could silently change which bytes the
//! encoder addresses.
//!
//! So this test pins the exact bytes. The digests below were measured
//! identical on 64-bit native (`aarch64-apple-darwin`) and on 32-bit
//! `wasm32-wasip1`, overflow checks on in both. Running this file on any
//! target therefore asserts that target's encoding against the 64-bit
//! reference: a divergence is a failure, not a footnote.
//!
//! The inputs are fixed-seed and the sizes are deliberately small so the whole
//! file runs in seconds under a debug wasm interpreter.

fn lcg_fill(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let unit = ((state >> 40) as f32) / ((1u64 << 24) as f32);
        out.push(unit - 0.5);
    }
    out
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// `IdMapIndex` is the type `rustyred-thg-core`'s graph vector index wraps, so
/// its serialized form is the one that reaches RustyRed's store.
#[test]
fn id_map_encoding_is_identical_to_the_64_bit_reference() {
    use turbovec::IdMapIndex;

    let (dim, n, bits) = (64usize, 300usize, 4usize);
    let vectors = lcg_fill(n * dim, 0x5eed_1234);
    let ids: Vec<u64> = (0..n as u64).map(|i| i * 7 + 3).collect();

    let mut index = IdMapIndex::new(dim, bits).expect("construct");
    index.add_with_ids(&vectors, &ids).expect("add");

    let image = index.to_bytes();
    assert_eq!(
        image.len(),
        126_227,
        "serialized length moved; the pinned digest below belongs to 126227 bytes"
    );
    assert_eq!(
        fnv1a(&image),
        0x948b_9202_9097_9fac,
        "IdMapIndex encoding differs from the 64-bit reference"
    );

    // A byte-identical image is not by itself a working index; the ranking has
    // to survive the round trip too.
    let query = &vectors[..dim];
    let (_, got) = index.search(query, 5);
    assert_eq!(got, vec![3, 171, 2033, 1095, 262], "top-5 ids moved");

    let reloaded = IdMapIndex::from_bytes(&image).expect("from_bytes");
    assert_eq!(reloaded.len(), n);
    let (_, got_again) = reloaded.search(query, 5);
    assert_eq!(got_again, got, "round trip changed the ranking");
}

/// The bare `TurboQuantIndex` form, at a different `dim`/`bit_width` so the
/// block geometry and the byte-group count both differ from the case above.
#[test]
fn turbo_quant_encoding_is_identical_to_the_64_bit_reference() {
    use turbovec::TurboQuantIndex;

    let (dim, n, bits) = (128usize, 200usize, 3usize);
    let vectors = lcg_fill(n * dim, 0xabcd_9876);

    let mut index = TurboQuantIndex::new(dim, bits).expect("construct");
    index.add(&vectors);

    let image = index.to_bytes();
    assert_eq!(
        image.len(),
        177_187,
        "serialized length moved; the pinned digest below belongs to 177187 bytes"
    );
    assert_eq!(
        fnv1a(&image),
        0xf38d_4c3d_24f2_e510,
        "TurboQuantIndex encoding differs from the 64-bit reference"
    );

    let reloaded = TurboQuantIndex::from_bytes(&image).expect("from_bytes");
    assert_eq!(reloaded.len(), n);
}

# Relational operators bench

Invocation: `cargo run --manifest-path rustyredcore_THG/crates/rustyred-thg-core/bench/relational/Cargo.toml --release -- --write-out`

Seeds: 0,1,2,3,4. Stddev footnotes are sample standard deviation.

| workload | mean ns | stddev ns | mean intermediates | stddev intermediates | note |
|---|---:|---:|---:|---:|---|
| triangle.generic | 7233 | 5387 | 37.00 | 6.32 | cyclic RO2; generic intermediates stay near output, binary exceeds |
| triangle.binary | 15692 | 12648 | 74.00 | 12.65 | cyclic RO2; generic intermediates stay near output, binary exceeds |
| brief.left | 1942 | 832 | 16.00 | 0.00 | RO1 brief-assembly; miss pads instead of dropping the run |
| coverage.anti | 5750 | 7040 | 1.00 | 0.00 | RO5 emptiness gate; count is the uncovered-section witness |
| p8.four_way | 8167 | 2847 | 252.00 | 0.00 | P8 four-way equi over receipt-shaped keys |
| steering.cold_start | 0 | 0 | 0.00 | 0.00 | candidates=join_colt_binary,join_mask reason=cold_start_native_floor |

Cyclic separation: generic mean intermediates 37.00 vs binary 74.00 (binary exceeds generic).
Shape key: `join|rels:3|kinds:|keys:`

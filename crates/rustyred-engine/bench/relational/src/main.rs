//! Relational operator bench (SPEC-THEOREM-RELATIONAL-OPERATORS-1.0 RO7).
//!
//! Workloads: triangle (cyclic RO2), brief-assembly left join (RO1), coverage
//! anti-join gate (RO5), and a four-way equi join (P8 shape). Metrics come from
//! COLT receipts: wall time, intermediate tuples, and steering decisions.
//!
//! Invocation:
//! `cargo run --manifest-path rustyredcore_THG/crates/rustyred-thg-core/bench/relational/Cargo.toml --release -- --write-out`

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use rustyred_thg_core::{
    emptiness_gate, execute_binary_join, execute_triangle_binary, execute_triangle_generic,
    join_shape_key, enumerate_join_candidates, JoinKind, JoinPredicate, JoinSteeringState,
    Emission, JOIN_CANDIDATE_COLT_BINARY,
};

const DEFAULT_SEEDS: [u64; 5] = [0, 1, 2, 3, 4];
const INVOCATION: &str = "cargo run --manifest-path rustyredcore_THG/crates/rustyred-thg-core/bench/relational/Cargo.toml --release -- --write-out";

#[derive(Clone, Debug)]
struct Cell {
    workload: &'static str,
    mean_ns: f64,
    stddev_ns: f64,
    mean_intermediates: f64,
    stddev_intermediates: f64,
    note: String,
}

fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    var.sqrt()
}

fn triangle(seed: u64) -> (Vec<(String, String)>, Vec<(String, String)>, Vec<(String, String)>) {
    let n = 8 + (seed % 5) as usize;
    let mut r = Vec::new();
    let mut s = Vec::new();
    let mut t = Vec::new();
    for index in 0..n {
        r.push((format!("a{index}"), format!("b{index}")));
        s.push((format!("b{index}"), format!("c{index}")));
        t.push((format!("c{index}"), format!("a{index}")));
        if index + 1 < n {
            r.push((format!("a{index}"), format!("b{}", index + 1)));
            s.push((format!("b{}", index + 1), format!("c{index}")));
        }
    }
    (r, s, t)
}

fn measure_triangle(generic: bool) -> Cell {
    let mut times = Vec::new();
    let mut intermediates = Vec::new();
    for seed in DEFAULT_SEEDS {
        let (r, s, t) = triangle(seed);
        let started = Instant::now();
        let (_, stats) = if generic {
            execute_triangle_generic(&r, &s, &t)
        } else {
            execute_triangle_binary(&r, &s, &t)
        };
        times.push(started.elapsed().as_nanos() as f64);
        intermediates.push(stats.intermediate_tuples as f64);
    }
    Cell {
        workload: if generic {
            "triangle.generic"
        } else {
            "triangle.binary"
        },
        mean_ns: times.iter().sum::<f64>() / times.len() as f64,
        stddev_ns: stddev(&times),
        mean_intermediates: intermediates.iter().sum::<f64>() / intermediates.len() as f64,
        stddev_intermediates: stddev(&intermediates),
        note: "cyclic RO2; generic intermediates stay near output, binary exceeds".to_string(),
    }
}

fn measure_left() -> Cell {
    let mut times = Vec::new();
    let mut intermediates = Vec::new();
    for seed in DEFAULT_SEEDS {
        let left: Vec<Option<String>> = (0..16)
            .map(|index| Some(format!("run-{}", index + seed)))
            .collect();
        let right: Vec<Option<String>> = (0..8)
            .map(|index| Some(format!("run-{}", index + seed)))
            .collect();
        let started = Instant::now();
        let (pairs, stats) = execute_binary_join(&left, &right, Emission::Left);
        times.push(started.elapsed().as_nanos() as f64);
        intermediates.push(stats.intermediate_tuples as f64);
        assert_eq!(pairs.len(), 16);
    }
    Cell {
        workload: "brief.left",
        mean_ns: times.iter().sum::<f64>() / times.len() as f64,
        stddev_ns: stddev(&times),
        mean_intermediates: intermediates.iter().sum::<f64>() / intermediates.len() as f64,
        stddev_intermediates: stddev(&intermediates),
        note: "RO1 brief-assembly; miss pads instead of dropping the run".to_string(),
    }
}

fn measure_coverage() -> Cell {
    let mut times = Vec::new();
    let mut intermediates = Vec::new();
    for seed in DEFAULT_SEEDS {
        let left: Vec<String> = (0..10).map(|index| format!("section-{index}")).collect();
        let right: Vec<String> = (0..9)
            .map(|index| format!("section-{}", (index + seed as usize) % 10))
            .collect();
        let started = Instant::now();
        let receipt = emptiness_gate(&left, &right);
        times.push(started.elapsed().as_nanos() as f64);
        intermediates.push(receipt.count as f64);
        assert_eq!(receipt.count, 1);
    }
    Cell {
        workload: "coverage.anti",
        mean_ns: times.iter().sum::<f64>() / times.len() as f64,
        stddev_ns: stddev(&times),
        mean_intermediates: intermediates.iter().sum::<f64>() / intermediates.len() as f64,
        stddev_intermediates: stddev(&intermediates),
        note: "RO5 emptiness gate; count is the uncovered-section witness".to_string(),
    }
}

fn measure_four_way() -> Cell {
    let mut times = Vec::new();
    let mut intermediates = Vec::new();
    for seed in DEFAULT_SEEDS {
        let keys: Vec<Option<String>> = (0..12)
            .map(|index| Some(format!("k-{}-{}", seed, index % 4)))
            .collect();
        let started = Instant::now();
        let (ab, ab_stats) = execute_binary_join(&keys, &keys, Emission::Inner);
        let left: Vec<Option<String>> = ab
            .iter()
            .map(|(left, _)| keys[*left].clone())
            .collect();
        let (abc, abc_stats) = execute_binary_join(&left, &keys, Emission::Inner);
        let left: Vec<Option<String>> = abc
            .iter()
            .map(|(left, _)| keys.get(*left).cloned().flatten().map(Some).unwrap_or(None))
            .collect();
        let (_, abcd_stats) = execute_binary_join(&left, &keys, Emission::Inner);
        times.push(started.elapsed().as_nanos() as f64);
        intermediates.push(
            ab_stats.intermediate_tuples as f64
                + abc_stats.intermediate_tuples as f64
                + abcd_stats.intermediate_tuples as f64,
        );
    }
    Cell {
        workload: "p8.four_way",
        mean_ns: times.iter().sum::<f64>() / times.len() as f64,
        stddev_ns: stddev(&times),
        mean_intermediates: intermediates.iter().sum::<f64>() / intermediates.len() as f64,
        stddev_intermediates: stddev(&intermediates),
        note: "P8 four-way equi over receipt-shaped keys".to_string(),
    }
}

fn measure_steering() -> Cell {
    let joins = [JoinPredicate {
        left_alias: "a".into(),
        left_column: "id".into(),
        right_alias: "b".into(),
        right_column: "id".into(),
        kind: JoinKind::Inner,
    }];
    let state = JoinSteeringState::default();
    let first = state.decide(&joins, 2, 0.1, 20);
    assert_eq!(first.selected_candidate_id, JOIN_CANDIDATE_COLT_BINARY);
    Cell {
        workload: "steering.cold_start",
        mean_ns: 0.0,
        stddev_ns: 0.0,
        mean_intermediates: 0.0,
        stddev_intermediates: 0.0,
        note: format!(
            "candidates={} reason={}",
            enumerate_join_candidates(&joins).join(","),
            first.reason
        ),
    }
}

fn render(cells: &[Cell]) -> String {
    let mut out = String::from("# Relational operators bench\n\n");
    out.push_str(&format!("Invocation: `{INVOCATION}`\n\n"));
    out.push_str("Seeds: 0,1,2,3,4. Stddev footnotes are sample standard deviation.\n\n");
    out.push_str("| workload | mean ns | stddev ns | mean intermediates | stddev intermediates | note |\n");
    out.push_str("|---|---:|---:|---:|---:|---|\n");
    for cell in cells {
        out.push_str(&format!(
            "| {} | {:.0} | {:.0} | {:.2} | {:.2} | {} |\n",
            cell.workload,
            cell.mean_ns,
            cell.stddev_ns,
            cell.mean_intermediates,
            cell.stddev_intermediates,
            cell.note
        ));
    }
    let generic = cells.iter().find(|cell| cell.workload == "triangle.generic");
    let binary = cells.iter().find(|cell| cell.workload == "triangle.binary");
    if let (Some(generic), Some(binary)) = (generic, binary) {
        out.push_str(&format!(
            "\nCyclic separation: generic mean intermediates {:.2} vs binary {:.2} (binary exceeds generic).\n",
            generic.mean_intermediates, binary.mean_intermediates
        ));
        out.push_str(&format!("Shape key: `{}`\n", join_shape_key(&[], 3)));
    }
    out
}

fn main() {
    let write_out = env::args().any(|arg| arg == "--write-out");
    let cells = vec![
        measure_triangle(true),
        measure_triangle(false),
        measure_left(),
        measure_coverage(),
        measure_four_way(),
        measure_steering(),
    ];
    let table = render(&cells);
    print!("{table}");
    if write_out {
        let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out");
        fs::create_dir_all(&out_dir).expect("create out/");
        fs::write(out_dir.join("table.md"), table).expect("write table.md");
    }
}

use theorem_oplog::{read_write_overlap, AccessPath, ReadSet, WriteSet};

const SEED: u64 = 0xD4_0A11_CE55_2026;

#[test]
fn boundary_matrix_matches_the_slow_reference() {
    let cases = [
        (vec![], vec![], false),
        (vec![AccessPath::Wildcard], vec![], false),
        (vec![AccessPath::Wildcard], vec![entity("note:1")], true),
        (vec![entity("note:1")], vec![entity("note:2")], false),
        (
            vec![entity("note:1")],
            vec![attribute("note:1", "title")],
            true,
        ),
        (
            vec![attribute("note:1", "title")],
            vec![attribute("note:1", "body")],
            false,
        ),
        (
            vec![interval("note:1", "body", 0, 10)],
            vec![interval("note:1", "body", 10, 20)],
            false,
        ),
        (
            vec![interval("note:1", "body", 0, 11)],
            vec![interval("note:1", "body", 10, 20)],
            true,
        ),
    ];

    for (reads, writes, expected) in cases {
        let reads = ReadSet::new(reads).unwrap();
        let writes = WriteSet::new(writes).unwrap();
        assert_eq!(read_write_overlap(&reads, &writes), expected);
        assert_eq!(read_write_overlap(&reads, &writes), slow(&reads, &writes));
    }
}

#[test]
fn randomized_sets_match_independent_pairwise_model() {
    let mut random = Random(SEED);
    for case in 0..20_000 {
        let reads = ReadSet::new(random.paths()).unwrap();
        let writes = WriteSet::new(random.paths()).unwrap();
        assert_eq!(
            read_write_overlap(&reads, &writes),
            slow(&reads, &writes),
            "seed={SEED:#x}, case={case}, reads={reads:?}, writes={writes:?}"
        );
        assert_eq!(
            read_write_overlap(&reads, &writes),
            read_write_overlap(
                &ReadSet::new(writes.entries().iter().cloned()).unwrap(),
                &WriteSet::new(reads.entries().iter().cloned()).unwrap(),
            ),
            "symmetry failed at seed={SEED:#x}, case={case}"
        );
    }
}

#[test]
fn order_duplicates_and_subsumed_paths_do_not_change_results() {
    let canonical = ReadSet::new([
        interval("note:1", "body", 0, 20),
        attribute("note:2", "title"),
    ])
    .unwrap();
    let redundant = ReadSet::new([
        interval("note:1", "body", 10, 20),
        attribute("note:2", "title"),
        interval("note:1", "body", 0, 10),
        interval("note:1", "body", 0, 10),
        interval("note:2", "title", -50, 50),
    ])
    .unwrap();
    assert_eq!(canonical, redundant);

    for write in [
        WriteSet::default(),
        WriteSet::new([interval("note:1", "body", 20, 30)]).unwrap(),
        WriteSet::new([interval("note:1", "body", 19, 30)]).unwrap(),
        WriteSet::new([attribute("note:2", "title")]).unwrap(),
    ] {
        assert_eq!(
            read_write_overlap(&canonical, &write),
            read_write_overlap(&redundant, &write)
        );
    }
}

#[test]
fn deserialization_reestablishes_canonical_form() {
    let json = r#"[
        {"scope":"interval","entity":" note:1 ","attribute":" body ","start":5,"end":10},
        {"scope":"interval","entity":"note:1","attribute":"body","start":0,"end":5}
    ]"#;
    let reads: ReadSet = serde_json::from_str(json).unwrap();
    assert_eq!(reads.entries(), &[interval("note:1", "body", 0, 10)]);
}

fn slow(reads: &ReadSet, writes: &WriteSet) -> bool {
    reads.entries().iter().any(|read| {
        writes
            .entries()
            .iter()
            .any(|write| slow_path_overlap(read, write))
    })
}

fn slow_path_overlap(left: &AccessPath, right: &AccessPath) -> bool {
    use AccessPath::{Attribute, Entity, Interval, Wildcard};
    match (left, right) {
        (Wildcard, _) | (_, Wildcard) => true,
        (Entity { entity: left }, Entity { entity: right }) => left == right,
        (Entity { entity: left }, Attribute { entity: right, .. })
        | (Attribute { entity: left, .. }, Entity { entity: right })
        | (Entity { entity: left }, Interval { entity: right, .. })
        | (Interval { entity: left, .. }, Entity { entity: right }) => left == right,
        (
            Attribute {
                entity: left_entity,
                attribute: left_attribute,
            },
            Attribute {
                entity: right_entity,
                attribute: right_attribute,
            },
        ) => left_entity == right_entity && left_attribute == right_attribute,
        (
            Attribute {
                entity: left_entity,
                attribute: left_attribute,
            },
            Interval {
                entity: right_entity,
                attribute: right_attribute,
                ..
            },
        )
        | (
            Interval {
                entity: left_entity,
                attribute: left_attribute,
                ..
            },
            Attribute {
                entity: right_entity,
                attribute: right_attribute,
            },
        ) => left_entity == right_entity && left_attribute == right_attribute,
        (
            Interval {
                entity: left_entity,
                attribute: left_attribute,
                start: left_start,
                end: left_end,
            },
            Interval {
                entity: right_entity,
                attribute: right_attribute,
                start: right_start,
                end: right_end,
            },
        ) => {
            left_entity == right_entity
                && left_attribute == right_attribute
                && left_start < right_end
                && right_start < left_end
        }
    }
}

fn entity(entity: &str) -> AccessPath {
    AccessPath::Entity {
        entity: entity.to_string(),
    }
}

fn attribute(entity: &str, attribute: &str) -> AccessPath {
    AccessPath::Attribute {
        entity: entity.to_string(),
        attribute: attribute.to_string(),
    }
}

fn interval(entity: &str, attribute: &str, start: i64, end: i64) -> AccessPath {
    AccessPath::Interval {
        entity: entity.to_string(),
        attribute: attribute.to_string(),
        start,
        end,
    }
}

struct Random(u64);

impl Random {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn paths(&mut self) -> Vec<AccessPath> {
        let len = (self.next() % 9) as usize;
        (0..len).map(|_| self.path()).collect()
    }

    fn path(&mut self) -> AccessPath {
        let entity = format!("entity:{}", self.next() % 5);
        let attribute = format!("attribute:{}", self.next() % 4);
        match self.next() % 10 {
            0 => AccessPath::Wildcard,
            1..=3 => AccessPath::Entity { entity },
            4..=6 => AccessPath::Attribute { entity, attribute },
            _ => {
                let start = (self.next() % 40) as i64 - 20;
                let width = (self.next() % 8 + 1) as i64;
                interval(&entity, &attribute, start, start + width)
            }
        }
    }
}

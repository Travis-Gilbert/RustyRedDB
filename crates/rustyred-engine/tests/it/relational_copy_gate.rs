use std::collections::{BTreeSet, VecDeque};

const RELATIONAL_SOURCE: &str = include_str!("../../src/relational.rs");

fn declared_type_names(source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for line in source.lines() {
        let trimmed = line.trim_start();
        for keyword in ["struct ", "enum "] {
            let Some(index) = trimmed.find(keyword) else {
                continue;
            };
            let prefix = &trimmed[..index];
            if !prefix.is_empty() && !prefix.starts_with("pub") {
                continue;
            }
            let name = trimmed[index + keyword.len()..]
                .chars()
                .take_while(|character| character.is_ascii_alphanumeric() || *character == '_')
                .collect::<String>();
            if !name.is_empty() {
                names.insert(name);
            }
        }
    }
    names
}

fn declaration_span(source: &str, name: &str) -> Option<(usize, usize)> {
    let declaration_start = [format!("struct {name}"), format!("enum {name}")]
        .into_iter()
        .filter_map(|needle| source.find(&needle))
        .min()?;
    let opening_offset = source[declaration_start..].find('{')?;
    let opening = declaration_start + opening_offset;
    let mut depth = 0usize;
    for (offset, character) in source[opening..].char_indices() {
        match character {
            '{' => depth += 1,
            '}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some((opening + 1, opening + offset));
                }
            }
            _ => {}
        }
    }
    None
}

fn contains_identifier(source: &str, identifier: &str) -> bool {
    source.match_indices(identifier).any(|(index, _)| {
        let before = source[..index].chars().next_back();
        let after = source[index + identifier.len()..].chars().next();
        !before.is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
            && !after.is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
    })
}

fn row_authority_field(body: &str) -> Option<String> {
    body.lines().find_map(|line| {
        let (field, _) = line.split_once(':')?;
        let field = field
            .split_whitespace()
            .next_back()?
            .trim_matches(|character: char| !character.is_ascii_alphanumeric() && character != '_');
        let folded = field.to_ascii_lowercase();
        (folded == "rows"
            || folded.starts_with("rows_")
            || folded.ends_with("_rows")
            || folded.contains("row_map")
            || folded.contains("row_cache"))
        .then(|| field.to_string())
    })
}

fn copy_gate_violations(source: &str) -> Vec<String> {
    let local_types = declared_type_names(source);
    let mut queue = VecDeque::from(["RelationalLens".to_string()]);
    let mut visited = BTreeSet::new();
    let mut violations = Vec::new();

    while let Some(name) = queue.pop_front() {
        if !visited.insert(name.clone()) {
            continue;
        }
        let Some((body_start, body_end)) = declaration_span(source, &name) else {
            violations.push(format!("reachable declaration is missing: {name}"));
            continue;
        };
        let body = &source[body_start..body_end];

        for forbidden in ["ScalarValue", "Value", "ResolvedRow"] {
            if contains_identifier(body, forbidden) {
                violations.push(format!(
                    "{name} stores forbidden relational value type {forbidden}"
                ));
            }
        }
        if let Some(field) = row_authority_field(body) {
            violations.push(format!("{name} stores row authority in field {field}"));
        }

        for candidate in &local_types {
            if !visited.contains(candidate) && contains_identifier(body, candidate) {
                queue.push_back(candidate.clone());
            }
        }
    }

    violations
}

fn inject_lens_field(source: &str, field: &str) -> String {
    let (_, body_end) = declaration_span(source, "RelationalLens")
        .expect("RelationalLens must have a braced declaration");
    let mut mutated = source.to_string();
    mutated.insert_str(body_end, &format!("    {field}\n"));
    mutated
}

#[test]
fn lens_reachable_declarations_store_no_rows_or_values() {
    let violations = copy_gate_violations(RELATIONAL_SOURCE);
    assert!(
        violations.is_empty(),
        "relational copy gate violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn gate_rejects_deliberately_reintroduced_row_and_property_copies() {
    let row_copy = inject_lens_field(
        RELATIONAL_SOURCE,
        "rows: BTreeMap<RowId, BTreeMap<ColumnId, ScalarValue>>>,",
    );
    let row_violations = copy_gate_violations(&row_copy);
    assert!(row_violations
        .iter()
        .any(|violation| violation.contains("ScalarValue")));
    assert!(row_violations
        .iter()
        .any(|violation| violation.contains("field rows")));

    let property_copy = inject_lens_field(
        RELATIONAL_SOURCE,
        "properties_by_row: BTreeMap<RowId, Value>,",
    );
    let property_violations = copy_gate_violations(&property_copy);
    assert!(property_violations
        .iter()
        .any(|violation| violation.contains("Value")));
}

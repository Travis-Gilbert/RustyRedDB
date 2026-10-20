use datafrog::{PrefixFilter, Relation};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};

// ponytail: avoids intermediate Vec allocation from split_whitespace().collect::<Vec<_>>().join()
fn collapse_ws(s: &str, sep: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for w in s.split_whitespace() {
        if !out.is_empty() {
            out.push_str(sep);
        }
        out.push_str(w);
    }
    out
}

pub const DATALOG_RULE_IDS: [&str; 15] = [
    "unsupported_claim",
    "dependent_claim",
    "source_reused_support",
    "likely_duplicate_entity",
    "evidence_path_too_long",
    "claim_has_no_independent_support",
    "object_in_unresolved_tension_neighborhood",
    "code_symbol_touched_by_failing_postmortem_pattern",
    "context_atom_tainted_by_generated_artifact",
    "private_source_reaches_export_candidate",
    "demolition_window",
    "conflict_set",
    "vacancy_duration",
    "ownership_chain",
    "same_entity",
];

pub fn canonical_json(value: &Value) -> Result<String, String> {
    serde_json::to_string(&canonicalize_json(value))
        .map_err(|err| format!("could not serialize JSON: {err}"))
}

fn canonicalize_json(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries = map.iter().collect::<Vec<_>>();
            entries.sort_unstable_by(|(left, _), (right, _)| left.cmp(right));
            let mut canonical = Map::new();
            for (key, child) in entries {
                canonical.insert(key.clone(), canonicalize_json(child));
            }
            Value::Object(canonical)
        }
        Value::Array(values) => Value::Array(values.iter().map(canonicalize_json).collect()),
        _ => value.clone(),
    }
}

pub fn stable_hash_value(value: &Value) -> Result<String, String> {
    Ok(sha256_hex(&canonical_json(value)?))
}

pub fn derive_datalog_receipt(payload: &Value) -> Result<Value, String> {
    let (raw_facts, requested_rule_ids) = datalog_payload_parts(payload)?;
    let facts = normalize_facts(raw_facts)?;
    let index = facts_by_relation(&facts);
    let selected_rule_ids = selected_rule_ids(requested_rule_ids);
    let mut warnings: Vec<String> = Vec::new();
    let mut derived: Vec<Value> = Vec::new();

    for rule_id in &selected_rule_ids {
        match rule_id.as_str() {
            "unsupported_claim" => derived.extend(unsupported_claim(&index)?),
            "dependent_claim" => derived.extend(dependent_claim(&index)?),
            "source_reused_support" => derived.extend(source_reused_support(&index)?),
            "likely_duplicate_entity" => derived.extend(likely_duplicate_entity(&index)?),
            "evidence_path_too_long" => derived.extend(evidence_path_too_long(&index)?),
            "claim_has_no_independent_support" => {
                derived.extend(claim_has_no_independent_support(&index)?)
            }
            "object_in_unresolved_tension_neighborhood" => {
                derived.extend(object_in_unresolved_tension_neighborhood(&index)?)
            }
            "code_symbol_touched_by_failing_postmortem_pattern" => {
                derived.extend(code_symbol_touched_by_failing_postmortem_pattern(&index)?)
            }
            "context_atom_tainted_by_generated_artifact" => {
                derived.extend(context_atom_tainted_by_generated_artifact(&index)?)
            }
            "private_source_reaches_export_candidate" => {
                derived.extend(private_source_reaches_export_candidate(&index)?)
            }
            "demolition_window" => derived.extend(demolition_window(&index)?),
            "conflict_set" => derived.extend(conflict_set(&index)?),
            "vacancy_duration" => derived.extend(vacancy_duration(&index, 5)?),
            "ownership_chain" => derived.extend(ownership_chain(&index)?),
            "same_entity" => derived.extend(same_entity(&index)?),
            unknown => warnings.push(format!("Unknown Datalog rule skipped: {unknown}")),
        }
    }

    let mut deduped: BTreeMap<String, Value> = BTreeMap::new();
    for fact in derived {
        let fact_id = object_field(&fact, "fact_id").to_string();
        deduped.insert(fact_id, fact);
    }
    let mut derived_facts: Vec<Value> = deduped.into_values().collect();
    derived_facts.sort_by(|left, right| {
        (
            object_field(left, "rule_id"),
            object_field(left, "subject_id"),
            object_field(left, "fact_id"),
        )
            .cmp(&(
                object_field(right, "rule_id"),
                object_field(right, "subject_id"),
                object_field(right, "fact_id"),
            ))
    });

    Ok(json!({
        "engine": "python-reference-datalog",
        "fact_pack_hash": fact_pack_hash_for_facts(&facts)?,
        "rule_ids": selected_rule_ids,
        "derived_count": derived_facts.len(),
        "derived_facts": derived_facts,
        "warnings": warnings,
        "writeback_policy": "read-only",
    }))
}

fn unsupported_claim(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let (support_evidence, dependencies) = claim_support_indexes(index);
    let claims = relation(index, "claim");
    let candidates = Relation::from_iter(claims.iter().enumerate().filter_map(|(row, claim)| {
        let status = attr_string(claim, "status").to_lowercase();
        (!matches!(status.as_str(), "archived" | "refuted" | "superseded"))
            .then(|| (object_field(claim, "entity_id").to_owned(), row))
    }));
    let supported =
        Relation::from_iter(support_evidence.keys().chain(dependencies.keys()).cloned());
    let unsupported = static_antijoin(&candidates, &supported, |_, row| *row);
    let mut out = Vec::new();
    for row in unsupported.elements {
        let claim = &claims[row];
        let claim_id = object_field(claim, "entity_id");
        out.push(derived_fact(
                "unsupported_claim",
                "unsupported_claim",
                claim_id,
                "This claim has no supporting EvidenceLink or ClaimDependency in the current fact pack.",
                vec![object_field(claim, "fact_id").to_string()],
                json!({"status": attr_value_or(claim, "status", json!(""))}),
                1.0,
                "read-only",
            )?);
    }
    Ok(out)
}

fn dependent_claim(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for dep in select(index, "claim_dependency", |row| {
        !attr_string(row, "claim_id").is_empty()
    }) {
        let claim_id = attr_string(dep, "claim_id");
        out.push(derived_fact(
            "dependent_claim",
            "dependent_claim",
            &claim_id,
            "This claim depends on another graph object for its justification.",
            vec![object_field(dep, "fact_id").to_string()],
            json!({
                "depends_on_object_id": attr_value_or(dep, "depends_on_object_id", json!("")),
                "justification_type": attr_value_or(dep, "justification_type", json!("")),
                "strength": attr_value_or(dep, "strength", json!(0.0)),
            }),
            1.0,
            "read-only",
        )?);
    }
    Ok(out)
}

fn source_reused_support(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let mut input = Vec::new();
    for dep in relation(index, "claim_dependency") {
        let source = attr_string(dep, "depends_on_object_id");
        if !source.is_empty() {
            input.push((format!("object:{source}"), dep.clone()));
        }
    }
    for link in relation(index, "evidence_link") {
        let source = attr_string(link, "artifact_id");
        if !source.is_empty() {
            input.push((format!("artifact:{source}"), link.clone()));
        }
    }
    let claims = Relation::from_iter(input.iter().filter_map(|(source, row)| {
        let claim = attr_string(row, "claim_id");
        (!claim.is_empty()).then(|| (source.clone(), claim))
    }));
    let pairs = Relation::from_join(&claims, &claims, |source, left, right| {
        (source.clone(), left.clone(), right.clone())
    });
    let reused = Relation::from_iter(
        pairs
            .elements
            .into_iter()
            .filter(|(_, left, right)| left != right)
            .map(|(source, _, _)| (source, ())),
    );
    let by_source = joined_groups(&input, &reused);
    let mut out = Vec::new();
    for (source_id, facts) in by_source {
        let claim_ids: BTreeSet<String> = facts
            .iter()
            .map(|fact| attr_string(fact, "claim_id"))
            .filter(|claim_id| !claim_id.is_empty())
            .collect();
        out.push(derived_fact(
            "source_reused_support",
            "source_reused_support",
            &source_id,
            "The same source is reused as support for multiple claims.",
            facts
                .iter()
                .map(|fact| object_field(fact, "fact_id").to_string())
                .collect(),
            json!({"claim_ids": claim_ids.into_iter().collect::<Vec<_>>()}),
            0.8,
            "read-only",
        )?);
    }
    Ok(out)
}

fn likely_duplicate_entity(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let input: Vec<_> = relation(index, "object")
        .iter()
        .filter_map(|row| {
            let title = normalize_title(&attr_string(row, "title"));
            (!title.is_empty()).then(|| (title, row.clone()))
        })
        .collect();
    let titles = Relation::from_iter(
        input
            .iter()
            .enumerate()
            .map(|(row, (title, _))| (title.clone(), row)),
    );
    let pairs = Relation::from_join(&titles, &titles, |title, left, right| {
        (title.clone(), *left, *right)
    });
    let duplicate = Relation::from_iter(
        pairs
            .elements
            .into_iter()
            .filter(|(_, left, right)| left < right)
            .map(|(title, _, _)| (title, ())),
    );
    let by_title = joined_groups(&input, &duplicate);
    let mut out = Vec::new();
    for (title, objects) in by_title {
        out.push(derived_fact(
            "likely_duplicate_entity",
            "likely_duplicate_entity",
            object_field(&objects[0], "entity_id"),
            "Multiple graph objects share the same normalized title.",
            objects
                .iter()
                .map(|object| object_field(object, "fact_id").to_string())
                .collect(),
            json!({
                "normalized_title": title,
                "duplicate_object_ids": objects[1..]
                    .iter()
                    .map(|object| object_field(object, "entity_id").to_string())
                    .collect::<Vec<_>>(),
            }),
            0.7,
            "proposal-only",
        )?);
    }
    Ok(out)
}

fn evidence_path_too_long(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    let max_length = 3_i64;
    for path in select(index, "evidence_path", |row| {
        integer_attr(row, "path_length", 0) > max_length
    }) {
        let path_length = integer_attr(path, "path_length", 0);
        out.push(derived_fact(
            "evidence_path_too_long",
            "evidence_path_too_long",
            object_field(path, "entity_id"),
            "The evidence path exceeds the configured symbolic derivation depth.",
            vec![object_field(path, "fact_id").to_string()],
            json!({"path_length": path_length, "max_length": max_length}),
            0.9,
            "read-only",
        )?);
    }
    Ok(out)
}

fn claim_has_no_independent_support(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let (support_evidence, dependencies) = claim_support_indexes(index);
    let mut sources = Vec::new();
    for (claim, rows) in &dependencies {
        for row in rows {
            let source = attr_string(row, "depends_on_object_id");
            if !source.is_empty() {
                sources.push((claim.clone(), format!("object:{source}")));
            }
        }
    }
    for (claim, rows) in &support_evidence {
        for row in rows {
            let source = attr_string(row, "artifact_id");
            if !source.is_empty() {
                sources.push((claim.clone(), format!("artifact:{source}")));
            }
        }
    }
    let sources = Relation::from_vec(sources);
    let pairs = Relation::from_join(&sources, &sources, |claim, left, right| {
        (claim.clone(), left.clone(), right.clone())
    });
    let independent = Relation::from_iter(
        pairs
            .elements
            .into_iter()
            .filter(|(_, left, right)| left != right)
            .map(|(claim, _, _)| claim),
    );
    let claims = relation(index, "claim");
    let candidates = Relation::from_iter(
        claims
            .iter()
            .enumerate()
            .map(|(row, claim)| (object_field(claim, "entity_id").to_owned(), row)),
    );
    let selected = static_antijoin(&candidates, &independent, |_, row| *row);
    let mut out = Vec::new();
    for row in selected.elements {
        let claim = &claims[row];
        let claim_id = object_field(claim, "entity_id");
        let mut source_refs: BTreeSet<String> = BTreeSet::new();
        for dep in dependencies.get(claim_id).into_iter().flatten() {
            let source_id = attr_string(dep, "depends_on_object_id");
            if !source_id.is_empty() {
                source_refs.insert(format!("object:{source_id}"));
            }
        }
        for link in support_evidence.get(claim_id).into_iter().flatten() {
            let artifact_id = attr_string(link, "artifact_id");
            if !artifact_id.is_empty() {
                source_refs.insert(format!("artifact:{artifact_id}"));
            }
        }
        let mut dependency_ids = vec![object_field(claim, "fact_id").to_string()];
        dependency_ids.extend(
            dependencies
                .get(claim_id)
                .into_iter()
                .flatten()
                .map(|fact| object_field(fact, "fact_id").to_string()),
        );
        dependency_ids.extend(
            support_evidence
                .get(claim_id)
                .into_iter()
                .flatten()
                .map(|fact| object_field(fact, "fact_id").to_string()),
        );
        let support_sources: Vec<String> = source_refs.into_iter().collect();
        out.push(derived_fact(
            "claim_has_no_independent_support",
            "claim_has_no_independent_support",
            claim_id,
            "This claim does not have two independent support sources in the current fact pack.",
            dependency_ids,
            json!({
                "support_source_count": support_sources.len(),
                "support_sources": support_sources,
            }),
            0.85,
            "read-only",
        )?);
    }
    Ok(out)
}

fn object_in_unresolved_tension_neighborhood(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for edge in select(index, "edge", |row| {
        attr_string(row, "edge_type").eq_ignore_ascii_case("contradicts")
            || attr_string(row, "acceptance_status").eq_ignore_ascii_case("contested")
    }) {
        let edge_type = attr_string(edge, "edge_type").to_lowercase();
        let status = attr_string(edge, "acceptance_status").to_lowercase();
        for object_id in [
            attr_string(edge, "from_object_id"),
            attr_string(edge, "to_object_id"),
        ] {
            if object_id.is_empty() {
                continue;
            }
            out.push(derived_fact(
                "object_in_unresolved_tension_neighborhood",
                "object_in_unresolved_tension_neighborhood",
                &object_id,
                "This object is adjacent to a contradicting or contested edge.",
                vec![object_field(edge, "fact_id").to_string()],
                json!({
                    "edge_id": object_field(edge, "entity_id"),
                    "edge_type": edge_type,
                    "acceptance_status": status,
                }),
                0.8,
                "read-only",
            )?);
        }
    }
    Ok(out)
}

fn code_symbol_touched_by_failing_postmortem_pattern(
    index: &RelationIndex,
) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for atom in select(index, "context_atom", |row| {
        let metadata = attr_object(row, "metadata");
        attr_string(row, "kind") == "code_symbol"
            && [
                "failing_postmortem_pattern",
                "postmortem_failure",
                "failed_tests",
            ]
            .iter()
            .any(|key| metadata.get(*key).is_some_and(truthy))
    }) {
        out.push(derived_fact(
            "code_symbol_touched_by_failing_postmortem_pattern",
            "code_symbol_touched_by_failing_postmortem_pattern",
            object_field(atom, "entity_id"),
            "This code symbol is linked to a failing postmortem or failed-test pattern.",
            vec![object_field(atom, "fact_id").to_string()],
            json!({"title": attr_value_or(atom, "title", json!(""))}),
            0.85,
            "read-only",
        )?);
    }
    Ok(out)
}

fn context_atom_tainted_by_generated_artifact(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let mut out = Vec::new();
    for atom in select(index, "context_atom", |row| {
        let metadata = attr_object(row, "metadata");
        metadata.get("generated").is_some_and(truthy)
            || object_value_lower(metadata.get("source_kind")) == "generated_artifact"
            || object_value_lower(metadata.get("provenance")) == "generated"
    }) {
        out.push(derived_fact(
            "context_atom_tainted_by_generated_artifact",
            "context_atom_tainted_by_generated_artifact",
            object_field(atom, "entity_id"),
            "This context atom was derived from generated material and should not be treated as independent evidence.",
            vec![object_field(atom, "fact_id").to_string()],
            json!({"artifact_id": attr_value_or(atom, "artifact_id", json!(""))}),
            0.9,
            "read-only",
        )?);
    }
    Ok(out)
}

fn private_source_reaches_export_candidate(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let objects_by_id = facts_by_id(relation(index, "object"));
    let objects: Vec<_> = objects_by_id.into_iter().collect();
    let atoms = relation(index, "context_atom");
    let object_keys = Relation::from_iter(
        objects
            .iter()
            .enumerate()
            .map(|(row, (id, _))| (id.clone(), row)),
    );
    let atom_keys = Relation::from_iter(
        atoms
            .iter()
            .enumerate()
            .map(|(row, atom)| (attr_string_or_empty_if_falsey(atom, "object_pk"), row)),
    );
    let linked = Relation::from_join(&atom_keys, &object_keys, |_, atom, object| {
        (*atom, Some(*object))
    });
    let existing_ids = Relation::from_iter(objects.iter().map(|(id, _)| id.clone()));
    let unlinked = static_antijoin(&atom_keys, &existing_ids, |_, atom| (*atom, None));
    let candidates = linked.merge(unlinked);
    let selected = Relation::from_leapjoin(
        &candidates,
        PrefixFilter::from(|(atom, object): &(usize, Option<usize>)| {
            let metadata = attr_object(&atoms[*atom], "metadata");
            let properties = object
                .map(|row| attr_object(&objects[row].1, "properties"))
                .unwrap_or_default();
            let private = metadata.get("private").is_some_and(truthy)
                || object_value_lower(metadata.get("source_visibility")) == "private"
                || properties.get("private").is_some_and(truthy)
                || object_value_lower(properties.get("visibility")) == "private";
            let export = metadata.get("export_candidate").is_some_and(truthy)
                || metadata.get("public_export").is_some_and(truthy)
                || object_value_lower(metadata.get("export_visibility")) == "public";
            private && export
        }),
        |row, &()| *row,
    );
    let mut out = Vec::new();
    for (atom_row, object_row) in selected.elements {
        let atom = &atoms[atom_row];
        let object_id = attr_string_or_empty_if_falsey(atom, "object_pk");
        let mut dependency_ids = vec![object_field(atom, "fact_id").to_string()];
        if let Some(object_row) = object_row {
            let object = &objects[object_row].1;
            dependency_ids.push(object_field(object, "fact_id").to_string());
        }
        out.push(derived_fact(
            "private_source_reaches_export_candidate",
            "private_source_reaches_export_candidate",
            object_field(atom, "entity_id"),
            "A private source is marked as a public export candidate.",
            dependency_ids,
            json!({
                "object_pk": object_id,
                "artifact_id": attr_value_or(atom, "artifact_id", json!("")),
            }),
            0.95,
            "proposal-only",
        )?);
    }
    Ok(out)
}

fn demolition_window(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let present = structure_observations_by_parcel(index, "structure_present");
    let absent = structure_observations_by_parcel(index, "structure_absent");
    let mut out = Vec::new();

    for (parcel, present_obs) in present {
        let Some(absent_obs) = absent.get(&parcel) else {
            continue;
        };
        let present_years: BTreeSet<i64> = present_obs.iter().map(|(year, _)| *year).collect();
        let absent_years: BTreeSet<i64> = absent_obs.iter().map(|(year, _)| *year).collect();
        let earliest_absent = absent_years.into_iter().find(|absent_year| {
            present_years
                .iter()
                .any(|present_year| present_year < absent_year)
        });
        let Some(earliest_absent) = earliest_absent else {
            continue;
        };
        let Some(latest_present) = present_years
            .iter()
            .filter(|present_year| **present_year < earliest_absent)
            .max()
            .copied()
        else {
            continue;
        };

        let present_facts: Vec<Value> = present_obs
            .iter()
            .filter(|(year, _)| *year == latest_present)
            .map(|(_, fact)| fact.clone())
            .collect();
        let absent_facts: Vec<Value> = absent_obs
            .iter()
            .filter(|(year, _)| *year == earliest_absent)
            .map(|(_, fact)| fact.clone())
            .collect();
        let present_sources = source_ids(&present_facts);
        let absent_sources = source_ids(&absent_facts);
        let present_label = if present_sources.is_empty() {
            "an earlier survey".to_string()
        } else {
            present_sources.join(", ")
        };
        let absent_label = if absent_sources.is_empty() {
            "a later survey".to_string()
        } else {
            absent_sources.join(", ")
        };

        let mut dependency_ids: Vec<String> = present_facts
            .iter()
            .map(|fact| object_field(fact, "fact_id").to_string())
            .collect();
        dependency_ids.extend(
            absent_facts
                .iter()
                .map(|fact| object_field(fact, "fact_id").to_string()),
        );
        out.push(derived_fact(
            "demolition_window",
            "demolition_between",
            &parcel,
            &format!(
                "A structure stood at parcel {parcel} in {latest_present} \
                 (per {present_label}) and was absent by {earliest_absent} \
                 (per {absent_label}); demolition occurred between \
                 {latest_present} and {earliest_absent}."
            ),
            dependency_ids,
            json!({
                "parcel_id": parcel,
                "earliest_year": latest_present,
                "latest_year": earliest_absent,
                "window_years": earliest_absent - latest_present,
                "present_source_ids": present_sources,
                "absent_source_ids": absent_sources,
            }),
            1.0,
            "read-only",
        )?);
    }

    Ok(out)
}

fn conflict_set(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let mut grouped: BTreeMap<(String, String), Vec<Value>> = BTreeMap::new();
    for fact in relation(index, "source_assertion") {
        let parcel = attr_string_or_entity_if_falsey(fact, "parcel_id")
            .trim()
            .to_string();
        let field = attr_string(fact, "field").trim().to_string();
        if parcel.is_empty() || field.is_empty() {
            continue;
        }
        grouped
            .entry((parcel, field))
            .or_default()
            .push(fact.clone());
    }

    let mut out = Vec::new();
    for ((parcel, field), assertions) in grouped {
        let mut values_to_sources: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for fact in &assertions {
            let value = attr_string(fact, "value").trim().to_string();
            if value.is_empty() {
                continue;
            }
            let source = attr_string(fact, "source_id");
            let source = if source.is_empty() {
                object_field(fact, "entity_id").to_string()
            } else {
                source
            };
            values_to_sources.entry(value).or_default().insert(source);
        }
        if values_to_sources.len() < 2 {
            continue;
        }

        let rendered = values_to_sources
            .iter()
            .map(|(value, sources)| {
                format!(
                    "{} (per {})",
                    value,
                    sources.iter().cloned().collect::<Vec<_>>().join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        let dependency_ids = sorted_fact_ids(&assertions);
        let source_ids = assertions
            .iter()
            .map(|fact| {
                let source = attr_string(fact, "source_id");
                if source.is_empty() {
                    object_field(fact, "entity_id").to_string()
                } else {
                    source
                }
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let values = values_to_sources
            .iter()
            .map(|(value, sources)| {
                (
                    value.clone(),
                    Value::Array(sources.iter().cloned().map(Value::String).collect()),
                )
            })
            .collect::<Map<_, _>>();

        out.push(derived_fact(
            "conflict_set",
            "source_conflict",
            &format!("{parcel}:{field}"),
            &format!("Sources disagree on {field} for parcel {parcel}: {rendered}."),
            dependency_ids,
            json!({
                "parcel_id": parcel,
                "field": field,
                "values": values,
                "source_ids": source_ids,
                "distinct_value_count": values_to_sources.len(),
            }),
            1.0,
            "read-only",
        )?);
    }
    Ok(out)
}

fn vacancy_duration(index: &RelationIndex, min_years: i64) -> Result<Vec<Value>, String> {
    let mut by_parcel: BTreeMap<String, Vec<(i64, String, Value)>> = BTreeMap::new();
    for fact in relation(index, "assessor_status") {
        let parcel = attr_string_or_entity_if_falsey(fact, "parcel_id")
            .trim()
            .to_string();
        let Some(year) = attr(fact, "year").and_then(coerce_year) else {
            continue;
        };
        let status = attr_string(fact, "status").trim().to_lowercase();
        if parcel.is_empty() || status.is_empty() {
            continue;
        }
        by_parcel
            .entry(parcel)
            .or_default()
            .push((year, status, fact.clone()));
    }

    let mut out = Vec::new();
    for (parcel, mut observations) in by_parcel {
        observations.sort_by(|left, right| {
            (left.0, object_field(&left.2, "fact_id"))
                .cmp(&(right.0, object_field(&right.2, "fact_id")))
        });

        let mut best_run: Vec<(i64, String, Value)> = Vec::new();
        let mut current: Vec<(i64, String, Value)> = Vec::new();
        for observation in observations {
            if observation.1 == "vacant" {
                current.push(observation);
            } else if observation.1 == "occupied" {
                if current.len() > best_run.len() {
                    best_run = current;
                }
                current = Vec::new();
            }
        }
        if current.len() > best_run.len() {
            best_run = current;
        }
        if best_run.is_empty() {
            continue;
        }

        let earliest = best_run.first().map(|item| item.0).unwrap_or_default();
        let latest = best_run.last().map(|item| item.0).unwrap_or_default();
        let span = latest - earliest;
        if span < min_years {
            continue;
        }
        let run_facts = best_run
            .iter()
            .map(|(_, _, fact)| fact.clone())
            .collect::<Vec<_>>();
        let source_ids = source_ids(&run_facts);

        out.push(derived_fact(
            "vacancy_duration",
            "prolonged_vacancy",
            &parcel,
            &format!(
                "Parcel {parcel} was assessed vacant from {earliest} to {latest} \
                 ({span} years) with no intervening occupied record."
            ),
            sorted_fact_ids(&run_facts),
            json!({
                "parcel_id": parcel,
                "earliest_year": earliest,
                "latest_year": latest,
                "vacant_years": span,
                "min_years": min_years,
                "observation_count": run_facts.len(),
                "source_ids": source_ids,
            }),
            1.0,
            "read-only",
        )?);
    }
    Ok(out)
}

fn ownership_chain(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let mut by_parcel: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for fact in relation(index, "ownership") {
        let parcel = attr_string_or_entity_if_falsey(fact, "parcel_id")
            .trim()
            .to_string();
        if parcel.is_empty() {
            continue;
        }
        by_parcel.entry(parcel).or_default().push(fact.clone());
    }

    let mut out = Vec::new();
    for (parcel, records) in by_parcel {
        if records.len() < 2 {
            continue;
        }

        let mut ordered = records.clone();
        ordered.sort_by(|left, right| {
            let left_year = attr(left, "from_year").and_then(coerce_year);
            let right_year = attr(right, "from_year").and_then(coerce_year);
            (
                left_year.is_none(),
                left_year.unwrap_or_default(),
                object_field(left, "fact_id"),
            )
                .cmp(&(
                    right_year.is_none(),
                    right_year.unwrap_or_default(),
                    object_field(right, "fact_id"),
                ))
        });

        let mut chain: Vec<Value> = Vec::new();
        for record in &ordered {
            let owner_type = attr_string(record, "owner_type").trim().to_lowercase();
            chain.push(json!({
                "owner": attr_string(record, "owner"),
                "owner_type": owner_type,
                "from_year": optional_year_value(record, "from_year"),
                "to_year": optional_year_value(record, "to_year"),
                "source_id": attr_string(record, "source_id"),
            }));
        }

        let mut foreclosure_year: Option<i64> = None;
        let mut land_bank_year: Option<i64> = None;
        for link in &chain {
            let owner_type = object_field(link, "owner_type");
            if owner_type == "tax_foreclosure" && foreclosure_year.is_none() {
                foreclosure_year = link.get("from_year").and_then(coerce_year);
            } else if owner_type == "land_bank"
                && foreclosure_year.is_some()
                && land_bank_year.is_none()
            {
                land_bank_year = link.get("from_year").and_then(coerce_year);
            }
        }
        let distressed = foreclosure_year.is_some() && land_bank_year.is_some();
        let owners_label = chain
            .iter()
            .map(|link| {
                let label = [
                    object_field(link, "owner"),
                    object_field(link, "owner_type"),
                    "unknown",
                ]
                .into_iter()
                .find(|value| !value.is_empty())
                .unwrap_or("unknown");
                match link.get("from_year").and_then(coerce_year) {
                    Some(year) => format!("{label} ({year})"),
                    None => label.to_string(),
                }
            })
            .collect::<Vec<_>>()
            .join(", ");
        let mut reason = format!(
            "Parcel {parcel} passed through {} owners: {owners_label}.",
            chain.len()
        );
        if distressed {
            reason.push_str(&format!(
                " Acquired by the land bank via tax foreclosure ({} to {}).",
                foreclosure_year.unwrap_or_default(),
                land_bank_year.unwrap_or_default()
            ));
        }

        out.push(derived_fact(
            "ownership_chain",
            "ownership_chain",
            &parcel,
            &reason,
            sorted_fact_ids(&records),
            json!({
                "parcel_id": parcel,
                "owner_count": chain.len(),
                "chain": chain,
                "tax_foreclosure_to_land_bank": distressed,
                "foreclosure_year": foreclosure_year,
                "land_bank_year": land_bank_year,
            }),
            1.0,
            "read-only",
        )?);
    }
    Ok(out)
}

type RelationIndex = BTreeMap<String, Vec<Value>>;

fn sha256_hex(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn datalog_payload_parts(
    payload: &Value,
) -> Result<(&Vec<Value>, Option<Vec<String>>), String> {
    if let Some(facts) = payload.as_array() {
        return Ok((facts, None));
    }
    let facts = payload
        .get("facts")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            "datalog payload expected a JSON array or an object with facts array".to_string()
        })?;
    let rule_ids = payload
        .get("rule_ids")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect::<Vec<_>>()
        });
    Ok((facts, rule_ids))
}

fn selected_rule_ids(requested_rule_ids: Option<Vec<String>>) -> Vec<String> {
    match requested_rule_ids {
        Some(ids) if !ids.is_empty() => ids,
        _ => DATALOG_RULE_IDS
            .iter()
            .map(|rule_id| (*rule_id).to_string())
            .collect(),
    }
}

pub(crate) fn normalize_facts(raw_facts: &[Value]) -> Result<Vec<Value>, String> {
    raw_facts.iter().map(normalize_fact).collect()
}

fn normalize_fact(raw: &Value) -> Result<Value, String> {
    let relation = field_string(raw, "relation");
    let entity_id = field_string(raw, "entity_id");
    if relation.trim().is_empty() {
        return Err("DatalogFact requires relation".to_string());
    }
    if entity_id.trim().is_empty() {
        return Err("DatalogFact requires entity_id".to_string());
    }
    let attributes = raw
        .get("attributes")
        .filter(|value| value.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let source_ref = field_string(raw, "source_ref");
    let fallback_id = stable_hash_value(&json!({
        "relation": relation,
        "entity_id": entity_id,
        "attributes": attributes,
        "source_ref": source_ref,
    }))?;
    let fact_id = field_string(raw, "fact_id");
    Ok(json!({
        "fact_id": if fact_id.is_empty() { fallback_id } else { fact_id },
        "relation": field_string(raw, "relation"),
        "entity_id": field_string(raw, "entity_id"),
        "attributes": raw
            .get("attributes")
            .filter(|value| value.is_object())
            .cloned()
            .unwrap_or_else(|| json!({})),
        "source_ref": field_string(raw, "source_ref"),
    }))
}

pub(crate) fn fact_pack_hash_for_facts(facts: &[Value]) -> Result<String, String> {
    let mut sorted = facts.to_vec();
    sorted.sort_by(|left, right| object_field(left, "fact_id").cmp(object_field(right, "fact_id")));
    stable_hash_value(&Value::Array(sorted))
}

fn facts_by_relation(facts: &[Value]) -> RelationIndex {
    let mut index: RelationIndex = BTreeMap::new();
    for fact in facts {
        index
            .entry(object_field(fact, "relation").to_string())
            .or_default()
            .push(fact.clone());
    }
    for facts in index.values_mut() {
        facts.sort_by(|left, right| {
            object_field(left, "fact_id").cmp(object_field(right, "fact_id"))
        });
    }
    index
}

fn claim_support_indexes(
    index: &RelationIndex,
) -> (BTreeMap<String, Vec<Value>>, BTreeMap<String, Vec<Value>>) {
    let mut support_evidence: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut dependencies: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for link in relation(index, "evidence_link") {
        let relation_type = attr_string(link, "relation_type").to_lowercase();
        if matches!(
            relation_type.as_str(),
            "supports" | "derived_from" | "cites" | "references"
        ) {
            support_evidence
                .entry(attr_string(link, "claim_id"))
                .or_default()
                .push(link.clone());
        }
    }
    for dep in relation(index, "claim_dependency") {
        dependencies
            .entry(attr_string(dep, "claim_id"))
            .or_default()
            .push(dep.clone());
    }
    (support_evidence, dependencies)
}

fn facts_by_id(facts: &[Value]) -> BTreeMap<String, Value> {
    facts
        .iter()
        .map(|fact| (object_field(fact, "entity_id").to_string(), fact.clone()))
        .collect()
}

#[derive(Clone)]
struct EntityPath {
    target: String,
    confidence: f64,
    dependency_fact_ids: Vec<String>,
    path_length: usize,
}

fn same_entity(index: &RelationIndex) -> Result<Vec<Value>, String> {
    let mut unique = BTreeMap::<(String, String), (f64, String)>::new();
    for fact in relation(index, "match_edge") {
        let left = attr_string(fact, "left_record_id");
        let right = attr_string(fact, "right_record_id");
        if left.is_empty() || right.is_empty() || left == right {
            continue;
        }
        let key = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        let candidate = (
            log2_weight_probability(attr(fact, "weight").and_then(value_to_f64).unwrap_or(0.0)),
            object_field(fact, "fact_id").to_string(),
        );
        unique
            .entry(key)
            .and_modify(|existing| {
                if candidate.0 > existing.0
                    || (candidate.0 == existing.0 && candidate.1 < existing.1)
                {
                    *existing = candidate.clone();
                }
            })
            .or_insert(candidate);
    }

    let mut adjacent = BTreeMap::<String, Vec<(String, f64, String)>>::new();
    for ((left, right), (weight, fact_id)) in unique {
        adjacent
            .entry(left.clone())
            .or_default()
            .push((right.clone(), weight, fact_id.clone()));
        adjacent
            .entry(right)
            .or_default()
            .push((left, weight, fact_id));
    }
    for neighbors in adjacent.values_mut() {
        neighbors.sort_by(|left, right| (&left.0, &left.2).cmp(&(&right.0, &right.2)));
    }

    let mut derived = Vec::new();
    for source in adjacent.keys() {
        let mut best = BTreeMap::<String, EntityPath>::new();
        let mut queue = VecDeque::new();
        for (target, confidence, dependency) in &adjacent[source] {
            queue.push_back(EntityPath {
                target: target.clone(),
                confidence: *confidence,
                dependency_fact_ids: vec![dependency.clone()],
                path_length: 1,
            });
        }
        while let Some(path) = queue.pop_front() {
            if path.target == *source || path.path_length >= adjacent.len() {
                continue;
            }
            let improves = best.get(&path.target).is_none_or(|prior| {
                path.confidence > prior.confidence + f64::EPSILON
                    || ((path.confidence - prior.confidence).abs() <= f64::EPSILON
                        && path.dependency_fact_ids < prior.dependency_fact_ids)
            });
            if !improves {
                continue;
            }
            best.insert(path.target.clone(), path.clone());
            if path.path_length + 1 >= adjacent.len() {
                continue;
            }
            for (target, confidence, dependency) in &adjacent[&path.target] {
                if target == source || path.dependency_fact_ids.contains(dependency) {
                    continue;
                }
                let mut dependencies = path.dependency_fact_ids.clone();
                dependencies.push(dependency.clone());
                dependencies.sort();
                dependencies.dedup();
                queue.push_back(EntityPath {
                    target: target.clone(),
                    confidence: path.confidence * confidence * 0.9,
                    dependency_fact_ids: dependencies,
                    path_length: path.path_length + 1,
                });
            }
        }
        for path in best.into_values().filter(|path| source < &path.target) {
            derived.push(derived_fact(
                "same_entity",
                "same_entity",
                source,
                "Threshold-crossing match edges connect these records.",
                path.dependency_fact_ids,
                json!({"target_id": path.target, "path_length": path.path_length}),
                path.confidence,
                "read-only",
            )?);
        }
    }
    Ok(derived)
}

fn log2_weight_probability(weight: f64) -> f64 {
    if weight >= 0.0 {
        1.0 / (1.0 + (-weight).exp2())
    } else {
        let odds = weight.exp2();
        odds / (1.0 + odds)
    }
}

fn structure_observations_by_parcel(
    index: &RelationIndex,
    relation_name: &str,
) -> BTreeMap<String, Vec<(i64, Value)>> {
    let mut grouped: BTreeMap<String, Vec<(i64, Value)>> = BTreeMap::new();
    for fact in relation(index, relation_name) {
        let parcel = attr_string_or_entity_if_falsey(fact, "parcel_id")
            .trim()
            .to_string();
        let Some(year) = attr(fact, "year").and_then(coerce_year) else {
            continue;
        };
        if !parcel.is_empty() {
            grouped
                .entry(parcel)
                .or_default()
                .push((year, fact.clone()));
        }
    }
    grouped
}

fn source_ids(facts: &[Value]) -> Vec<String> {
    facts
        .iter()
        .filter_map(|fact| attr(fact, "source_id"))
        .filter(|value| python_truthy(value))
        .map(value_to_string)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn sorted_fact_ids(facts: &[Value]) -> Vec<String> {
    facts
        .iter()
        .map(|fact| object_field(fact, "fact_id").to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn optional_year_value(fact: &Value, key: &str) -> Value {
    attr(fact, key)
        .and_then(coerce_year)
        .map(Value::from)
        .unwrap_or(Value::Null)
}

fn relation<'a>(index: &'a RelationIndex, relation: &str) -> &'a [Value] {
    index.get(relation).map(Vec::as_slice).unwrap_or(&[])
}

fn derived_fact(
    rule_id: &str,
    relation: &str,
    subject_id: &str,
    reason: &str,
    dependency_fact_ids: Vec<String>,
    attributes: Value,
    confidence: f64,
    writeback_policy: &str,
) -> Result<Value, String> {
    let hash_payload = json!({
        "rule_id": rule_id,
        "relation": relation,
        "subject_id": subject_id,
        "attributes": attributes,
        "dependency_fact_ids": dependency_fact_ids,
    });
    let fact_id = stable_hash_value(&hash_payload)?;
    Ok(json!({
        "fact_id": fact_id,
        "rule_id": rule_id,
        "relation": relation,
        "subject_id": subject_id,
        "reason": reason,
        "dependency_fact_ids": hash_payload["dependency_fact_ids"].clone(),
        "attributes": hash_payload["attributes"].clone(),
        "confidence": confidence,
        "writeback_policy": writeback_policy,
    }))
}

fn field_string(value: &Value, key: &str) -> String {
    value.get(key).map(value_to_string).unwrap_or_default()
}

fn object_field<'a>(value: &'a Value, key: &str) -> &'a str {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn attr<'a>(fact: &'a Value, key: &str) -> Option<&'a Value> {
    fact.get("attributes")
        .and_then(Value::as_object)
        .and_then(|attributes| attributes.get(key))
}

fn attr_value_or(fact: &Value, key: &str, default: Value) -> Value {
    attr(fact, key).cloned().unwrap_or(default)
}

fn attr_string(fact: &Value, key: &str) -> String {
    attr(fact, key).map(value_to_string).unwrap_or_default()
}

fn attr_string_or_empty_if_falsey(fact: &Value, key: &str) -> String {
    attr(fact, key)
        .filter(|value| python_truthy(value))
        .map(value_to_string)
        .unwrap_or_default()
}

fn attr_string_or_entity_if_falsey(fact: &Value, key: &str) -> String {
    attr(fact, key)
        .filter(|value| python_truthy(value))
        .map(value_to_string)
        .unwrap_or_else(|| object_field(fact, "entity_id").to_string())
}

fn attr_object(fact: &Value, key: &str) -> Map<String, Value> {
    attr(fact, key)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default()
}

fn integer_attr(fact: &Value, key: &str, default: i64) -> i64 {
    attr(fact, key).map(value_to_i64).unwrap_or(default)
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(value) => value.to_string(),
        Value::Null => "None".to_string(),
        other => other.to_string(),
    }
}

fn value_to_f64(value: &Value) -> Option<f64> {
    if let Some(value) = value.as_f64() {
        return Some(value);
    }
    value.as_str().and_then(|value| value.parse::<f64>().ok())
}

fn value_to_i64(value: &Value) -> i64 {
    if let Some(value) = value.as_i64() {
        return value;
    }
    if let Some(value) = value.as_u64() {
        return i64::try_from(value).unwrap_or(i64::MAX);
    }
    if let Some(value) = value.as_f64() {
        return value as i64;
    }
    value
        .as_str()
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(0)
}

fn coerce_year(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(_) | Value::Null => None,
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                return Some(value);
            }
            if let Some(value) = number.as_u64() {
                return i64::try_from(value).ok();
            }
            number.as_f64().map(|value| value as i64)
        }
        Value::String(value) => first_four_digit_year(value),
        other => first_four_digit_year(&value_to_string(other)),
    }
}

fn first_four_digit_year(value: &str) -> Option<i64> {
    let chars: Vec<char> = value.chars().collect();
    for window in chars.windows(4) {
        if window.iter().all(|ch| ch.is_ascii_digit()) {
            let year: String = window.iter().collect();
            return year.parse::<i64>().ok();
        }
    }
    None
}

fn object_value_lower(value: Option<&Value>) -> String {
    value
        .map(value_to_string)
        .unwrap_or_default()
        .to_lowercase()
}

fn truthy(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::String(value) => matches!(
            value.to_lowercase().as_str(),
            "1" | "true" | "yes" | "private" | "public"
        ),
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Null => false,
    }
}

fn python_truthy(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::String(value) => !value.is_empty(),
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        Value::Null => false,
    }
}

fn normalize_title(value: &str) -> String {
    let mut cleaned = String::with_capacity(value.len());
    for ch in value.to_lowercase().chars() {
        if ch.is_ascii_lowercase() || ch.is_ascii_digit() {
            cleaned.push(ch);
        } else {
            cleaned.push(' ');
        }
    }
    collapse_ws(&cleaned, " ")
}

/// Datafrog 2.0.1's Relation::from_antijoin delegates to JoinInput::recent,
/// which is empty for a static Relation. Its static leapjoin correctly visits
/// the relation; the negative predicate uses the sorted relation's key index.
fn static_antijoin<K: Ord, V: Ord, O: Ord>(
    input: &Relation<(K, V)>,
    excluded: &Relation<K>,
    mut project: impl FnMut(&K, &V) -> O,
) -> Relation<O> {
    Relation::from_leapjoin(
        input,
        PrefixFilter::from(|row: &(K, V)| excluded.elements.binary_search(&row.0).is_err()),
        |row, &()| project(&row.0, &row.1),
    )
}

/// Rows remain distinct by their input position, as in the Python fact pack.
/// Datafrog evaluates unary predicates; it does not deduplicate evidence IDs.
fn select<'a>(
    index: &'a RelationIndex,
    name: &str,
    predicate: impl Fn(&Value) -> bool,
) -> Vec<&'a Value> {
    let rows = relation(index, name);
    let input = Relation::from_iter(0..rows.len());
    Relation::from_leapjoin(
        &input,
        PrefixFilter::from(|row: &usize| predicate(&rows[*row])),
        |row, &()| *row,
    )
    .elements
    .into_iter()
    .map(|row| &rows[row])
    .collect()
}

/// A keyed join selects complete, stable evidence groups after the rule has
/// proved its relational condition. Integer positions preserve Python order.
fn joined_groups(
    input: &[(String, Value)],
    selected: &Relation<(String, ())>,
) -> BTreeMap<String, Vec<Value>> {
    let rows = Relation::from_iter(
        input
            .iter()
            .enumerate()
            .map(|(row, (key, _))| (key.clone(), row)),
    );
    let joined = Relation::from_join(&rows, selected, |key, row, &()| (key.clone(), *row));
    let mut groups = BTreeMap::<String, Vec<Value>>::new();
    for (key, row) in joined.elements {
        groups.entry(key).or_default().push(input[row].1.clone());
    }
    groups
}

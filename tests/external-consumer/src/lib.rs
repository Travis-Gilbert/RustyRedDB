//! This file is copied unchanged into both Git-source and packaged consumers.
#[cfg(test)]
mod tests {
    use rustyred::engine::{
        edge_time_interval, FullTextDesignation, FullTextIndex, SpatialDesignation, SpatialIndex,
    };
    use rustyred::{Database, EdgeRecord, GraphMutation, GraphMutationBatch, NodeRecord};
    use serde_json::json;
    use std::collections::BTreeMap;

    #[test]
    fn an_outside_crate_preserves_atomic_graph_vectors_payloads_and_history() {
        let dir = std::env::temp_dir().join(format!("rustyred-external-{}", std::process::id()));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).unwrap();
        }
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("outside.rr");
        let mut db = Database::create(&path).unwrap();
        assert_eq!(db.stats().unwrap().nodes_total, 0);
        db.designate_vector("Part", "embedding", 3).unwrap();
        db.mutate(GraphMutationBatch::new([
            GraphMutation::NodeUpsert(NodeRecord::new(
                "part:1",
                ["Part"],
                json!({"sku": "X", "embedding": [1.0, 0.0, 0.0]}),
            )),
            GraphMutation::NodeUpsert(NodeRecord::new(
                "part:2",
                ["Part"],
                json!({"sku": "Y", "embedding": [0.0, 1.0, 0.0]}),
            )),
            GraphMutation::EdgeUpsert(EdgeRecord::new(
                "edge:1",
                "part:1",
                "RELATED",
                "part:2",
                json!({"t_start_ms": 100, "t_end_ms": 200}),
            )),
        ]))
        .unwrap();
        let original = db
            .engine()
            .pin_graph_history()
            .unwrap()
            .head_roots()
            .logical_commit;
        let binary = b"exact-file-contents\x00\xff";
        db.put_file_node("file:1", "parts.bin", binary).unwrap();
        db.upsert_node(NodeRecord::new(
            "part:1",
            ["Part"],
            json!({"sku": "X-updated", "embedding": [1.0, 0.0, 0.0]}),
        ))
        .unwrap();
        drop(db);
        let before = std::fs::read(&path).unwrap();
        let db = Database::open_read_only(&path).unwrap();
        let rows = db
            .query("MATCH (p:Part) RETURN p", &BTreeMap::new())
            .unwrap();
        assert_eq!(rows.rows.len(), 2);
        assert_eq!(db.read_file_node("file:1").unwrap(), binary);
        let hits = db
            .engine()
            .vector_search(Some("Part"), "embedding", &[1.0, 0.0, 0.0], 2)
            .unwrap();
        assert_eq!(
            hits.iter().map(|hit| hit.0.as_str()).collect::<Vec<_>>(),
            ["part:1", "part:2"]
        );
        assert!(hits[0].1 < hits[1].1);
        let edge = db.engine().get_edge("edge:1").unwrap().unwrap();
        assert_eq!(edge.from_id, "part:1");
        assert_eq!(edge.to_id, "part:2");
        let interval = edge_time_interval(&edge).unwrap();
        assert!(interval.contains_ms(150));
        assert!(!interval.contains_ms(201));
        let old = db
            .engine()
            .pin_graph_history()
            .unwrap()
            .revision(original)
            .unwrap();
        assert_eq!(
            old.get_node("part:1").unwrap().unwrap().properties["sku"],
            "X"
        );
        assert_eq!(
            db.engine().get_node("part:1").unwrap().unwrap().properties["sku"],
            "X-updated"
        );
        assert!(db.verify().unwrap().ok);
        drop(old);
        drop(db);
        assert_eq!(std::fs::read(&path).unwrap(), before);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn explicit_fulltext_and_spatial_primitives_are_usable_without_product_runtime() {
        let mut text = FullTextIndex::for_designation(FullTextDesignation {
            label: "Part".into(),
            property: "description".into(),
        });
        text.upsert("bolt", "steel bolt fastener");
        text.upsert("washer", "copper washer ring");
        assert_eq!(text.search("bolt", 1)[0].0, "bolt");
        let mut spatial = SpatialIndex::for_designation(SpatialDesignation {
            label: "Place".into(),
            lat_property: "lat".into(),
            lon_property: "lon".into(),
            resolution: 8,
        });
        spatial.upsert("flint", 43.0125, -83.6875).unwrap();
        spatial.upsert("detroit", 42.3314, -83.0458).unwrap();
        assert_eq!(
            spatial.radius_search(43.0125, -83.6875, 1.0).unwrap(),
            ["flint"]
        );
        assert_eq!(spatial.bbox_search(43.0, -83.7, 43.1, -83.6), ["flint"]);
    }
}

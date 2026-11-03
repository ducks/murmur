use murmur_research_runner::{
    reader_transfer::{self, graphs, predict, pretrain},
    temporal_stream::StreamConfig,
    trained_strength::mean_abs,
};
use std::collections::HashSet;
fn config() -> StreamConfig {
    StreamConfig {
        seeds: vec![23],
        delays: vec![4, 8, 12],
        train_tokens: 150,
        test_tokens: 100,
        units: 16,
        degree: 3,
        learning_rate: 0.05,
    }
}
#[test]
fn graphs_are_reproducible_and_mean_matched() {
    let c = config();
    let a = graphs(&c, 23, 4);
    let b = graphs(&c, 23, 4);
    for (x, y) in a.graphs.iter().zip(&b.graphs) {
        assert_eq!(
            x.edges.iter().map(|e| e.weight).collect::<Vec<_>>(),
            y.edges.iter().map(|e| e.weight).collect::<Vec<_>>()
        );
    }
    assert!((mean_abs(&a.graphs[1]) - mean_abs(&a.graphs[3])).abs() < 1e-12);
    assert!(a.graphs[1]
        .edges
        .iter()
        .zip(&a.graphs[2].edges)
        .any(|(x, y)| x.weight != y.weight));
}
#[test]
fn outputs_are_frozen_reproducible_and_delayed() {
    let c = config();
    let g = pretrain(&c, 23, 4, false);
    let before: Vec<_> = g.edges.iter().map(|e| e.weight).collect();
    let a = predict(&c, 23, 12, g.clone());
    let b = predict(&c, 23, 12, g.clone());
    assert_eq!(a, b);
    assert_eq!(a.len(), 88);
    assert_eq!(before, g.edges.iter().map(|e| e.weight).collect::<Vec<_>>());
}
#[test]
fn seed_roles_do_not_overlap() {
    let a = StreamConfig::parse(include_str!("../configs/reader-transfer/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/reader-transfer/test.cfg")).unwrap();
    let offsets = [0u64, 31000, 32000, 33000, 34000, 37000];
    let roles = |c: &StreamConfig| {
        c.seeds
            .iter()
            .flat_map(|s| offsets.map(|o| s.wrapping_add(o)))
            .collect::<HashSet<_>>()
    };
    assert_eq!(roles(&a).len(), a.seeds.len() * offsets.len());
    assert_eq!(roles(&b).len(), b.seeds.len() * offsets.len());
    assert!(roles(&a).is_disjoint(&roles(&b)));
}
#[test]
fn invalid_seed_role_rejected() {
    let mut c = config();
    c.seeds = vec![3391001, 3394001];
    let p = std::env::temp_dir().join(format!(
        "murmur-reader-transfer-invalid-{}",
        std::process::id()
    ));
    assert!(!p.exists());
    assert!(reader_transfer::execute(&c, &p).is_err());
    assert!(!p.exists());
}

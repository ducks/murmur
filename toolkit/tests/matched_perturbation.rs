use murmur::graph::Graph;
use murmur_research_runner::{
    matched_perturbation::{self, matched_random},
    temporal_stream::StreamConfig,
};
use std::collections::HashSet;
fn config() -> StreamConfig {
    StreamConfig {
        seeds: vec![19],
        delays: vec![4],
        train_tokens: 120,
        test_tokens: 100,
        units: 16,
        degree: 3,
        learning_rate: 0.05,
    }
}
#[test]
fn per_edge_perturbation_is_bounded_and_exactly_matched() {
    let original = Graph::new(16, 3, 19);
    let mut learned = original.clone();
    for (i, e) in learned.edges.iter_mut().enumerate() {
        e.weight = if i % 2 == 0 { 0.5 } else { -0.5 };
    }
    let (random, forced) = matched_random(&original, &learned, 777);
    assert!(forced > 0);
    for ((a, b), p) in original.edges.iter().zip(&learned.edges).zip(&random.edges) {
        assert_eq!((a.from, a.to), (p.from, p.to));
        assert!(((a.weight - b.weight).abs() - (a.weight - p.weight).abs()).abs() < 1e-12);
        assert!(p.weight.abs() <= 0.5 + 1e-12);
    }
}
#[test]
fn comparison_is_paired_and_deterministic() {
    let c = config();
    let a = matched_perturbation::run(&c, 19, 4);
    let b = matched_perturbation::run(&c, 19, 4);
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.learned_frozen, b.learned_frozen);
    assert_eq!(a.random_frozen, b.random_frozen);
    assert_eq!(a.rows.len(), 96);
    assert_eq!(a.random_frozen.len(), a.rows.len());
    assert_eq!(
        a.random.edges.iter().map(|e| e.weight).collect::<Vec<_>>(),
        b.random.edges.iter().map(|e| e.weight).collect::<Vec<_>>()
    );
}
#[test]
fn all_development_and_test_seed_roles_are_disjoint() {
    let dev = StreamConfig::parse(include_str!("../configs/matched-perturbation/dev.cfg")).unwrap();
    let test =
        StreamConfig::parse(include_str!("../configs/matched-perturbation/test.cfg")).unwrap();
    let roles = |c: &StreamConfig| {
        c.seeds
            .iter()
            .flat_map(|s| [0u64, 31000, 32000, 33000, 34000, 35000].map(|o| s.wrapping_add(o)))
            .collect::<HashSet<_>>()
    };
    assert!(roles(&dev).is_disjoint(&roles(&test)));
}
#[test]
fn rejects_overlapping_seed_roles() {
    let mut c = config();
    c.seeds = vec![591001, 593001];
    let path = std::env::temp_dir().join(format!("murmur-matched-invalid-{}", std::process::id()));
    assert!(!path.exists());
    assert!(matched_perturbation::execute(&c, &path).is_err());
    assert!(!path.exists());
}

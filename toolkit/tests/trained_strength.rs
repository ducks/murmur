use murmur::graph::Graph;
use murmur_research_runner::{
    temporal_stream::StreamConfig,
    trained_strength::{self, match_strength, mean_abs},
};
use std::collections::HashSet;
fn config() -> StreamConfig {
    StreamConfig {
        seeds: vec![23],
        delays: vec![4],
        train_tokens: 200,
        test_tokens: 120,
        units: 16,
        degree: 3,
        learning_rate: 0.05,
    }
}
#[test]
fn matching_preserves_sign_wiring_and_hits_target_with_clipping() {
    let g = Graph::new(16, 3, 23);
    for target in [0.0, 0.12, 0.35, 0.49] {
        let (s, _) = match_strength(&g, target).unwrap();
        assert!((mean_abs(&s) - target).abs() < 1e-12);
        for (a, b) in g.edges.iter().zip(&s.edges) {
            assert_eq!((a.from, a.to), (b.from, b.to));
            assert!(a.weight * b.weight >= 0.0);
            assert!(b.weight.abs() <= 0.5);
        }
    }
    assert!(match_strength(&g, 0.51).is_err());
    let mut z = g;
    for e in &mut z.edges {
        e.weight = 0.0;
    }
    assert!(match_strength(&z, 0.01).is_err());
}
#[test]
fn comparisons_are_matched_paired_and_reproducible() {
    let c = config();
    let a = trained_strength::run(&c, 23, 4);
    let b = trained_strength::run(&c, 23, 4);
    assert_eq!(a.predictions, b.predictions);
    assert_eq!(a.baseline.rows, b.baseline.rows);
    assert_eq!(a.predictions[0].len(), 116);
    assert!((mean_abs(&a.graphs[0]) - mean_abs(&a.graphs[3])).abs() < 1e-12);
    assert!((mean_abs(&a.graphs[1]) - mean_abs(&a.graphs[2])).abs() < 1e-12);
}
#[test]
fn seed_roles_do_not_overlap_across_groups() {
    let a = StreamConfig::parse(include_str!("../configs/trained-strength/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/trained-strength/test.cfg")).unwrap();
    let roles = |c: &StreamConfig| {
        c.seeds
            .iter()
            .flat_map(|s| [0u64, 31000, 32000, 33000, 34000, 35000].map(|o| s.wrapping_add(o)))
            .collect::<HashSet<_>>()
    };
    assert!(roles(&a).is_disjoint(&roles(&b)));
}
#[test]
fn rejects_colliding_seeds_before_output() {
    let mut c = config();
    c.seeds = vec![1791001, 1793001];
    let p = std::env::temp_dir().join(format!(
        "murmur-trained-strength-invalid-{}",
        std::process::id()
    ));
    assert!(!p.exists());
    assert!(trained_strength::execute(&c, &p).is_err());
    assert!(!p.exists());
}

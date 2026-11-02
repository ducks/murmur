use murmur_research_runner::{
    temporal_plasticity::{self, PlasticModel, WEIGHT_LIMIT},
    temporal_stream::{self, StreamConfig},
};
use std::collections::HashSet;

fn config() -> StreamConfig {
    StreamConfig {
        seeds: vec![17],
        delays: vec![4],
        train_tokens: 120,
        test_tokens: 100,
        units: 16,
        degree: 3,
        learning_rate: 0.05,
    }
}

#[test]
fn learning_changes_bounded_edges_and_evaluation_freezes_them() {
    let c = config();
    let mut m = PlasticModel::new(&c, 17);
    let initial = m.graph.edges.clone();
    for t in 0..120 {
        m.advance((t / 3) % 2, true);
        if t >= 4 {
            m.learn((t / 7) % 2, (t / 7) % 2);
        }
    }
    assert!(m
        .graph
        .edges
        .iter()
        .zip(&initial)
        .any(|(a, b)| (a.weight - b.weight).abs() > 1e-10));
    assert!(m
        .graph
        .edges
        .iter()
        .all(|e| e.weight.is_finite() && e.weight.abs() <= WEIGHT_LIMIT));
    assert!(m
        .graph
        .edges
        .iter()
        .zip(&initial)
        .all(|(a, b)| a.from == b.from && a.to == b.to));
    let trained: Vec<_> = m.graph.edges.iter().map(|e| e.weight).collect();
    m.reset_state();
    for t in 0..100 {
        m.advance(t % 2, false);
    }
    assert_eq!(
        trained,
        m.graph.edges.iter().map(|e| e.weight).collect::<Vec<_>>()
    );
}

#[test]
fn paired_runs_match_fixed_baseline_and_are_reproducible() {
    let c = config();
    let a = temporal_plasticity::run_delay(&c, 17, 4);
    let b = temporal_plasticity::run_delay(&c, 17, 4);
    let fixed = temporal_stream::run_delay(&c, 17, 4);
    assert_eq!(a.rows, b.rows);
    assert_eq!(
        a.rows.iter().map(|r| &r.baseline).collect::<Vec<_>>(),
        fixed.rows.iter().collect::<Vec<_>>()
    );
    assert_eq!(
        a.adaptive
            .edges
            .iter()
            .map(|e| e.weight)
            .collect::<Vec<_>>(),
        b.adaptive
            .edges
            .iter()
            .map(|e| e.weight)
            .collect::<Vec<_>>()
    );
}

#[test]
fn development_and_test_seed_roles_are_disjoint() {
    let dev = StreamConfig::parse(include_str!("../configs/temporal-plasticity/dev.cfg")).unwrap();
    let test =
        StreamConfig::parse(include_str!("../configs/temporal-plasticity/test.cfg")).unwrap();
    let roles = |c: &StreamConfig| {
        c.seeds
            .iter()
            .flat_map(|s| [0u64, 31000, 32000, 33000, 34000].map(|o| s.wrapping_add(o)))
            .collect::<HashSet<_>>()
    };
    assert!(roles(&dev).is_disjoint(&roles(&test)));
}

#[test]
fn overlapping_seed_roles_are_rejected_before_creating_output() {
    let mut c = config();
    c.seeds = vec![291001, 293001];
    let path =
        std::env::temp_dir().join(format!("murmur-plasticity-invalid-{}", std::process::id()));
    assert!(!path.exists());
    assert!(temporal_plasticity::execute(&c, &path).is_err());
    assert!(!path.exists());
}

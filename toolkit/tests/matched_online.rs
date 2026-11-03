use murmur_research_runner::{
    matched_online, temporal_plasticity::PlasticModel, temporal_stream::StreamConfig,
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
fn norm_matching_and_zero_trace_are_safe() {
    let c = config();
    let mut m = PlasticModel::new(&c, 23);
    assert_eq!(m.learn_with_update_norm(1, 0, 0.25), 0.0);
    for t in 0..10 {
        m.advance(t % 2, true);
    }
    let before: Vec<_> = m.graph.edges.iter().map(|e| e.weight).collect();
    let actual = m.learn_with_update_norm(1, 0, 0.01);
    let norm = m
        .graph
        .edges
        .iter()
        .zip(before)
        .map(|(e, w)| (e.weight - w).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 0.01).abs() < 1e-12);
    assert!((norm - actual).abs() < 1e-12);
}
#[test]
fn matched_run_is_paired_and_deterministic() {
    let c = config();
    let a = matched_online::run(&c, 23, 4);
    let b = matched_online::run(&c, 23, 4);
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.matched, b.matched);
    assert_eq!(a.matched.len(), 116);
    let x = a.checkpoints.last().unwrap();
    assert!(x.max_norm_difference < 1e-10);
    assert!((x.reference_norm_sum - x.control_norm_sum).abs() < 1e-9);
}
#[test]
fn role_seeds_across_configs_are_disjoint() {
    let a = StreamConfig::parse(include_str!("../configs/matched-online/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/matched-online/test.cfg")).unwrap();
    let roles = |c: &StreamConfig| {
        c.seeds
            .iter()
            .flat_map(|s| [0u64, 31000, 32000, 33000, 34000].map(|o| s.wrapping_add(o)))
            .collect::<HashSet<_>>()
    };
    assert!(roles(&a).is_disjoint(&roles(&b)));
}
#[test]
fn rejects_seed_collisions() {
    let mut c = config();
    c.seeds = vec![991001, 993001];
    let p = std::env::temp_dir().join(format!(
        "murmur-matched-online-invalid-{}",
        std::process::id()
    ));
    assert!(!p.exists());
    assert!(matched_online::execute(&c, &p).is_err());
    assert!(!p.exists());
}

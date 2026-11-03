use murmur_research_runner::{
    connection_strength::{self, impulse, scaled_graph},
    temporal_stream::StreamConfig,
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
fn scaling_preserves_wiring_and_respects_limits() {
    let c = config();
    let a = scaled_graph(&c, 23, 1.0);
    let b = scaled_graph(&c, 23, 3.0);
    for (x, y) in a.edges.iter().zip(&b.edges) {
        assert_eq!((x.from, x.to), (y.from, y.to));
        assert!((y.weight - (x.weight * 3.0).clamp(-0.5, 0.5)).abs() < 1e-14);
    }
    assert_eq!(a.input, b.input);
}
#[test]
fn impulse_is_deterministic_and_start_is_input_difference() {
    let g = scaled_graph(&config(), 23, 1.0);
    let a = impulse(&g, 100, 4, 8);
    assert_eq!(a, impulse(&g, 100, 4, 8));
    assert_eq!(a.len(), 9);
    assert!(a[0] > 0.0);
    assert!(a.iter().all(|x| x.is_finite() && *x >= 0.0));
}
#[test]
fn runs_are_paired_and_deterministic() {
    let c = config();
    let a = connection_strength::run(&c, 23, 4);
    let b = connection_strength::run(&c, 23, 4);
    assert_eq!(a.scaled, b.scaled);
    assert_eq!(a.baseline.rows, b.baseline.rows);
    assert_eq!(a.scaled[1].len(), 116);
}
#[test]
fn seeds_do_not_overlap() {
    let a = StreamConfig::parse(include_str!("../configs/connection-strength/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/connection-strength/test.cfg")).unwrap();
    let roles = |c: &StreamConfig| {
        c.seeds
            .iter()
            .flat_map(|s| [0u64, 31000, 32000, 33000, 34000, 35000].map(|o| s.wrapping_add(o)))
            .collect::<HashSet<_>>()
    };
    assert!(roles(&a).is_disjoint(&roles(&b)));
}

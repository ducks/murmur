use murmur_research_runner::{
    reader_signal, temporal_plasticity::PlasticModel, temporal_stream::StreamConfig,
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
fn reader_signal_is_algebraic_component() {
    let c = config();
    let mut m = PlasticModel::new(&c, 23);
    for t in 0..20 {
        m.advance(t % 2, true);
        let p = m.class_one_probability();
        for bit in [0.0, 1.0] {
            assert!(((bit - p) - ((bit - 0.5) + (0.5 - p))).abs() < 1e-15);
        }
        m.learn((t / 3) % 2, (t / 5) % 2);
    }
}
#[test]
fn paired_run_is_reproducible() {
    let c = config();
    let a = reader_signal::run(&c, 23, 4);
    let b = reader_signal::run(&c, 23, 4);
    assert_eq!(a.reader_frozen, b.reader_frozen);
    assert_eq!(a.reader_online, b.reader_online);
    assert_eq!(a.comparison.baseline.rows, b.comparison.baseline.rows);
    assert_eq!(a.reader_frozen.len(), 116);
    assert!(a
        .reader_graph
        .edges
        .iter()
        .zip(&a.comparison.graphs[0].edges)
        .any(|(x, y)| x.weight != y.weight));
}
#[test]
fn distinct_seed_roles() {
    let a = StreamConfig::parse(include_str!("../configs/reader-signal/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/reader-signal/test.cfg")).unwrap();
    let offsets = [0u64, 31000, 32000, 33000, 34000, 35000];
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

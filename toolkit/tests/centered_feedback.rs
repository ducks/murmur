use murmur_research_runner::{
    centered_feedback, temporal_plasticity::PlasticModel, temporal_stream::StreamConfig,
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
fn explicit_error_reproduces_original_learning() {
    let c = config();
    let mut a = PlasticModel::new(&c, 23);
    let mut b = PlasticModel::new(&c, 23);
    for t in 0..80 {
        let x = t % 2;
        a.advance(x, true);
        b.advance(x, true);
        let p = b.class_one_probability();
        a.learn((t / 3) % 2, (t / 5) % 2);
        b.learn_with_error((t / 3) % 2, (t / 5) as f64 % 2.0 - p);
    }
    assert_eq!(
        a.graph.edges.iter().map(|e| e.weight).collect::<Vec<_>>(),
        b.graph.edges.iter().map(|e| e.weight).collect::<Vec<_>>()
    );
}
#[test]
fn runs_are_deterministic_and_paired() {
    let c = config();
    let a = centered_feedback::run(&c, 23, 4);
    let b = centered_feedback::run(&c, 23, 4);
    assert_eq!(a.frozen, b.frozen);
    assert_eq!(a.centered_online, b.centered_online);
    assert_eq!(a.baseline.rows, b.baseline.rows);
    assert_eq!(a.centered_online.len(), 116);
    assert!(a.graphs[1]
        .edges
        .iter()
        .zip(&a.graphs[2].edges)
        .any(|(x, y)| x.weight != y.weight));
}
#[test]
fn disjoint_seed_roles() {
    let a = StreamConfig::parse(include_str!("../configs/centered-feedback/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/centered-feedback/test.cfg")).unwrap();
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

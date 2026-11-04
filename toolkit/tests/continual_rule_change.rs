use murmur_research_runner::{
    continual_rule_change::{self, Markov, Models},
    temporal_stream::StreamConfig,
};
use std::collections::HashSet;
fn config() -> StreamConfig {
    StreamConfig {
        seeds: vec![23],
        delays: vec![1],
        train_tokens: 120,
        test_tokens: 80,
        units: 16,
        degree: 3,
        learning_rate: 0.05,
    }
}
#[test]
fn seeded_rules_have_opposite_transition_probabilities() {
    let a = continual_rule_change::sequence(89, 10000, 0);
    let b = continual_rule_change::sequence(89, 10000, 1);
    assert_eq!(a, continual_rule_change::sequence(89, 10000, 0));
    let ar = a.windows(2).filter(|x| x[0] == x[1]).count();
    let bf = b.windows(2).filter(|x| x[0] != x[1]).count();
    assert_eq!(ar, bf);
    assert!((8800..9200).contains(&ar));
}
#[test]
fn probe_does_not_change_live_model_or_reader() {
    let c = config();
    let mut m = Models::new(&c, 23);
    for t in 0..20 {
        m.predict(t % 2, 0, true);
        m.learn(t % 2, t % 2, 0);
    }
    let before = m.clone();
    let seq = continual_rule_change::sequence(99, 80, 0);
    let p = continual_rule_change::probe(&m, &seq, 0, 0);
    assert_eq!(p.trials, 80);
    assert_eq!(
        m.adaptive
            .graph
            .edges
            .iter()
            .map(|e| e.weight)
            .collect::<Vec<_>>(),
        before
            .adaptive
            .graph
            .edges
            .iter()
            .map(|e| e.weight)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        continual_rule_change::probe(&m, &seq, 0, 0),
        continual_rule_change::probe(&before, &seq, 0, 0)
    );
}
#[test]
fn oracle_keeps_a_separate_rule_row() {
    let mut m = Markov::new(0.1, true);
    for _ in 0..100 {
        m.learn(0, 0, 0);
    }
    let a = m.p[0];
    for _ in 0..100 {
        m.learn(0, 1, 1);
    }
    assert_eq!(m.p[0], a);
    assert_eq!(m.predict(0, 0), 0);
    assert_eq!(m.predict(0, 1), 1);
}
#[test]
fn runs_are_paired_deterministic_and_fixed_graph_unchanged() {
    let c = config();
    let a = continual_rule_change::run(&c, 23);
    let b = continual_rule_change::run(&c, 23);
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.probes, b.probes);
    assert_eq!(a.rows.len(), 360);
    assert!(a.weights.iter().all(|phase| phase
        .iter()
        .zip(&a.weights[0])
        .all(|((fixed, _), (init, _))| fixed == init)));
    assert!(a.weights[2]
        .iter()
        .any(|(fixed, adaptive)| fixed != adaptive));
}
#[test]
fn seed_roles_are_disjoint() {
    let a = StreamConfig::parse(include_str!("../configs/continual-rule-change/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/continual-rule-change/test.cfg")).unwrap();
    let offsets = [0u64, 33000, 40000, 42000, 44000, 46000, 48000];
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

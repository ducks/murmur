use murmur_research_runner::{pattern_reweight, temporal_stream::StreamConfig};
use std::collections::HashSet;
fn config() -> StreamConfig {
    StreamConfig {
        seeds: vec![23],
        delays: vec![4],
        train_tokens: 150,
        test_tokens: 100,
        units: 16,
        degree: 3,
        learning_rate: 0.05,
    }
}
#[test]
fn history_reader_is_seeded_and_causal_length() {
    let c = config();
    let a = pattern_reweight::train_and_predict(&c, 23, None, 7);
    let b = pattern_reweight::train_and_predict(&c, 23, None, 7);
    assert_eq!(a, b);
    assert_eq!(a.len(), 98);
    assert!(a.iter().all(|x| *x < 2));
}
#[test]
fn seed_roles_are_disjoint() {
    let a = StreamConfig::parse(include_str!("../configs/pattern-reweight/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/pattern-reweight/test.cfg")).unwrap();
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

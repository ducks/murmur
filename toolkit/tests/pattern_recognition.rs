use murmur_research_runner::{
    pattern_recognition::{self, label, Confusion},
    temporal_stream::StreamConfig,
};
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
fn labels_recognize_current_window_without_future() {
    let a = [1, 0, 1, 1, 0, 1, 0];
    let b = [1, 0, 1, 0, 1, 1, 1];
    assert_eq!(label(&a, 2), 1);
    assert_eq!(label(&a, 3), 0);
    assert_eq!(label(&a, 5), 1);
    assert_eq!(label(&a, 2), label(&b, 2));
}
#[test]
fn metrics_reflect_imbalance_and_absent_class() {
    let mut c = Confusion::default();
    for y in [0, 0, 0, 1] {
        c.add(y, 0);
    }
    assert_eq!(c.accuracy(), 0.75);
    assert_eq!(c.recall(), Some(0.0));
    assert_eq!(c.balanced_accuracy(), Some(0.5));
    assert_eq!(c.f1(), Some(0.0));
    let mut d = Confusion::default();
    d.add(0, 0);
    assert_eq!(d.recall(), None);
    assert_eq!(d.balanced_accuracy(), None);
    assert_eq!(d.f1(), None);
}
#[test]
fn runs_are_paired_and_deterministic() {
    let c = config();
    let a = pattern_recognition::run(&c, 23);
    let b = pattern_recognition::run(&c, 23);
    assert_eq!(a.predictions, b.predictions);
    assert_eq!(a.targets, b.targets);
    assert_eq!(a.targets.len(), 98);
    assert_eq!(a.predictions.len(), 5);
    assert!(a.predictions.iter().all(|x| x.len() == a.targets.len()));
}
#[test]
fn seed_roles_do_not_overlap() {
    let a = StreamConfig::parse(include_str!("../configs/pattern-recognition/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/pattern-recognition/test.cfg")).unwrap();
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
fn seed_collision_is_rejected_before_output() {
    let mut c = config();
    c.seeds = vec![3791001, 3794001];
    let p = std::env::temp_dir().join(format!("murmur-pattern-invalid-{}", std::process::id()));
    assert!(!p.exists());
    assert!(pattern_recognition::execute(&c, &p).is_err());
    assert!(!p.exists());
}

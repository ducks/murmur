use murmur::sequence_engine::{run, EngineConfig, SequenceEngine};
#[test]
fn learns_locally_and_context_can_reset() {
    let mut e = SequenceEngine::new(4, 0.5);
    assert_eq!(e.observe(1, true), None);
    let before = e.weights.clone();
    assert_eq!(e.observe(2, true), Some(0));
    assert_eq!(e.weights[0], before[0]);
    assert_eq!(e.weights[2], before[2]);
    assert_eq!(e.predict_from(1), 2);
    assert!((e.weights[1].iter().sum::<f64>() - 1.0).abs() < 1e-12);
    e.reset_context();
    assert_eq!(e.predict(), None);
    let before = e.weights.clone();
    e.observe(0, true);
    assert_eq!(e.weights, before);
}
#[test]
fn frozen_observation_does_not_change_edges() {
    let mut e = SequenceEngine::new(4, 0.2);
    e.observe(1, true);
    let before = e.weights.clone();
    e.observe(2, false);
    assert_eq!(e.weights, before);
}
#[test]
fn sequence_run_is_reproducible_and_adapts() {
    let c = EngineConfig {
        noise: 0.0,
        ..EngineConfig::default()
    };
    let a = run(&c).unwrap();
    let b = run(&c).unwrap();
    assert_eq!(format!("{:?}", a), format!("{:?}", b));
    assert_eq!(a.new_recall[0], 1.0);
    assert_eq!(a.old_recall[0], 0.0);
    assert_eq!(a.old_recall[1], 1.0);
    assert_eq!(a.new_recall[1], 0.0);
    assert!(a.recovery[0].is_some());
    assert!(run(&EngineConfig { window: 0, ..c }).is_err());
}

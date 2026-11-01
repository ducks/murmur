use murmur::discovery::{run, stream, Discovery};
#[test]
fn frozen_evaluation_does_not_update_weights_or_grow() {
    let mut m = Discovery::new(1, true, 1);
    for e in stream(1, 300, true, 0.1) {
        m.step(e.source, e.target, true);
    }
    let before: Vec<_> = m.models.iter().map(|m| m.weights.clone()).collect();
    let events = m.events.len();
    for e in stream(2, 300, true, 0.1) {
        m.step(e.source, e.target, false);
    }
    assert_eq!(
        before,
        m.models
            .iter()
            .map(|m| m.weights.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(events, m.events.len());
}
#[test]
fn deterministic_normalized_and_capacity_bounded() {
    let a = run(12, true, 0.1);
    let b = run(12, true, 0.1);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert!(a.auto.models.len() <= 2);
    for model in &a.auto.models {
        for row in &model.weights {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-10);
            assert!(row.iter().all(|x| *x > 0.0));
        }
    }
}

#[test]
fn growth_requires_sustained_surprise_after_warmup() {
    let mut m = Discovery::new(1, true, 3);
    for _ in 0..100 {
        m.step(0, 0, true);
    }
    assert!(m.events.is_empty());
    for i in 0..25 {
        m.step(0, i % 4, true);
    }
    assert!(m.events.is_empty());
    for i in 0..25 {
        m.step(0, i % 4, true);
    }
    assert_eq!(m.events.len(), 1);
    assert_eq!(m.events[0].step, 150);
    assert_eq!(m.models.len(), 2);
}

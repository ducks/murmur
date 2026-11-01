use murmur::{
    context_engine::{run, ContextEngine},
    sequence_engine::EngineConfig,
};
#[test]
fn contexts_learn_conflicting_transitions_without_touching_other_context() {
    let mut model = ContextEngine::new(4, 2, 0.5);
    model.observe(0, 0, true);
    model.observe(0, 1, true);
    let saved = model.contexts[0].weights.clone();
    model.observe(1, 3, true);
    assert_eq!(saved, model.contexts[0].weights);
    model.reset_sequence();
    model.observe(1, 0, false);
    model.observe(1, 3, true);
    assert_eq!(model.predict_from(0, 0), 1);
    assert_eq!(model.predict_from(1, 0), 3);
    assert_eq!(model.contexts[0].weights, saved);
}
#[test]
fn prediction_uses_new_context_and_precedes_update() {
    let mut m = ContextEngine::new(4, 2, 1.0);
    m.observe(0, 0, true);
    m.observe(0, 1, true);
    m.reset_sequence();
    m.observe(1, 0, true);
    assert_eq!(m.observe(1, 3, true), Some(0));
    assert_eq!(m.predict_from(1, 0), 3);
    assert_eq!(m.predict_from(0, 0), 1);
}
#[test]
fn experiment_retains_and_is_deterministic() {
    let c = EngineConfig {
        noise: 0.0,
        ..EngineConfig::default()
    };
    let a = run(&c).unwrap();
    let b = run(&c).unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(a.after_b[0], [1.0, 1.0]);
    assert_eq!(a.after_b[1], [0.0, 1.0]);
    assert_eq!(a.after_b[2], [1.0, 1.0]);
    assert_eq!(a.edges_after_a[0], a.edges_after_b[0]);
}

use murmur::{
    hidden_context::{predict, run, Filter},
    sequence_engine::EngineConfig,
};
#[test]
fn posterior_updates_and_mixture_is_normalized() {
    let mut f = Filter::new(0.1);
    assert_eq!(f.prior(), [0.5, 0.5]);
    f.update([0.9, 0.1]);
    assert!((f.posterior[0] - 0.9).abs() < 1e-12);
    assert!((f.posterior[0] + f.posterior[1] - 1.0).abs() < 1e-12);
    assert_eq!(predict([&[0.9, 0.1], &[0.1, 0.9]], f.prior()), 0);
    f.update([0.1, 0.9]);
    assert!(f.posterior[0] < 0.9);
}
#[test]
fn hidden_run_reproducible_and_recovers_context() {
    let c = EngineConfig {
        noise: 0.0,
        ..EngineConfig::default()
    };
    let a = run(&c, 0.02, 50, 4).unwrap();
    let b = run(&c, 0.02, 50, 4).unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert!(a.steps[49].posterior_a > 0.9);
    assert!(a.steps[99].posterior_a < 0.1);
    assert_eq!(a.steps.len(), 200);
    assert!(run(&c, 0.7, 50, 4).is_err());
}

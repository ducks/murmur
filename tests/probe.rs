use murmur::{
    experiment::Config,
    probe::{run_probe, CentroidProbe},
};
#[test]
fn probe_fits_and_requires_both_classes() {
    assert!(CentroidProbe::fit(&[(vec![0.0], 0)]).is_err());
    let p = CentroidProbe::fit(&[(vec![0.0, 1.0], 0), (vec![2.0, 1.0], 1)]).unwrap();
    assert_eq!(p.predict(&[0.1, 1.0]), 0);
    assert_eq!(p.predict(&[1.9, 1.0]), 1);
    assert_eq!(p.predict(&[1.0, 1.0]), 0);
}
#[test]
fn frozen_probe_is_reproducible_and_reads_clean_states() {
    let c = Config {
        delay: 12,
        train_trials: 100,
        test_trials: 100,
        ..Config::default()
    };
    let a = run_probe(&c, &[0.0, 0.01]).unwrap();
    let b = run_probe(&c, &[0.0, 0.01]).unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(a[0].probe_correct, 100);
    assert!(a[0].state_distance > 0.0);
    assert!(run_probe(&c, &[-1.0]).is_err());
}

use murmur::trial_growth::{accept, run, CANDIDATE_EVAL, CANDIDATE_TRAIN};
#[test]
fn candidate_branch_does_not_change_original() {
    let m = murmur::discovery::Discovery::new(1, false, 3);
    let c = m.candidate_branch().unwrap();
    assert_eq!(m.models.len(), 1);
    assert_eq!(c.models.len(), 2);
    assert_eq!(m.models[0].weights, c.models[0].weights);
}
#[test]
fn compare_requires_gain_above_penalty() {
    assert!(!accept(10.0, 10.0));
    assert!(accept(10.01, 10.0));
    assert!(!accept(-1.0, 0.0));
}
#[test]
fn trials_are_reproducible_and_require_later_data() {
    let a = run(12, true, 0.1, 10.0).unwrap();
    let b = run(12, true, 0.1, 10.0).unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    for trial in &a.trials {
        if trial.completed {
            assert_eq!(trial.end - trial.start, CANDIDATE_TRAIN + CANDIDATE_EVAL);
        } else {
            assert!(!trial.accepted);
        }
    }
    let impossible = run(12, true, 0.1, 1e9).unwrap();
    assert!(impossible.trials.iter().all(|t| !t.accepted));
    assert!(run(12, true, 0.1, -1.0).is_err());
}
#[test]
fn scheduled_penalty_tracks_attempts_and_zero_reproduces_baseline() {
    use murmur::trial_growth::{run_custom, run_scheduled, scheduled_penalty};
    assert_eq!(scheduled_penalty(40.0, 20.0, 1), 40.0);
    assert!(scheduled_penalty(40.0, 20.0, 2) > 40.0);
    let a = run_custom(15, false, 0.25, 40.0, 1200, 400, 100).unwrap();
    let b = run_scheduled(15, false, 0.25, 40.0, 1200, 400, 100, 0.0).unwrap();
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    let c = run_scheduled(15, false, 0.25, 40.0, 5000, 400, 100, 40.0).unwrap();
    for (i, t) in c.trials.iter().enumerate() {
        assert_eq!(t.penalty, scheduled_penalty(40.0, 40.0, i + 1));
    }
}

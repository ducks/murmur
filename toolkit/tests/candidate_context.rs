use murmur_research_runner::{candidate_context, temporal_stream::StreamConfig, unlabeled_context};
fn config() -> StreamConfig {
    StreamConfig {
        seeds: vec![23],
        delays: vec![1],
        train_tokens: 120,
        test_tokens: 100,
        units: 16,
        degree: 3,
        learning_rate: 0.05,
    }
}
#[test]
fn reference_models_match_existing_runner() {
    let c = config();
    let a = candidate_context::run(&c, 23, 0, 0.1);
    let b = unlabeled_context::run(&c, 23, 0, 0.1);
    assert_eq!(a.rows.len(), b.rows.len());
    for (x, y) in a.rows.iter().zip(&b.rows) {
        assert_eq!(x.event, y.event);
        assert!((x.predictions[0] - y.p[0]).abs() < 1e-12);
        assert!((x.predictions[1] - y.p[1]).abs() < 1e-12);
        assert!((0.0..=1.0).contains(&x.predictions[2]));
    }
    for (x, y) in a.probes.iter().zip(&b.probes) {
        assert_eq!(x.correct[1], y.correct[1]);
        assert!(x.correct[2] <= x.trials);
    }
}
#[test]
fn evaluator_does_not_pass_context_to_candidate() {
    let mut a = candidate_context::Evaluator::new(23, 0.05, 0.1);
    let mut b = a.clone();
    let x = unlabeled_context::Event {
        input: 0,
        target: 1,
        context: 0,
    };
    let y = unlabeled_context::Event { context: 1, ..x };
    for _ in 0..20 {
        assert_eq!(a.step(x, true)[2], b.step(y, true)[2]);
    }
}

#[test]
fn candidate_frozen_observations_do_not_change_learned_predictions_after_reset() {
    use murmur_research_runner::agent_candidate::Candidate;
    let mut trained = Candidate::new(23, 0.05);
    for i in 0..100 {
        trained.predict(i % 2);
        trained.observe(i % 2, (i / 3) % 2, true);
    }
    let mut control = trained.clone();
    for i in 0..100 {
        trained.predict(i % 2);
        trained.observe(i % 2, (i / 5) % 2, false);
    }
    trained.reset();
    control.reset();
    for x in 0..2 {
        assert!((trained.predict(x) - control.predict(x)).abs() < 1e-12);
    }
}

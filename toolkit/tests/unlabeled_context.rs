use murmur_research_runner::{
    temporal_stream::StreamConfig,
    unlabeled_context::{self, Event, Mixture, Models},
};
use std::collections::HashSet;
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
fn prediction_precedes_target_and_posterior_is_normalized() {
    let base = Mixture::new(23, 0.05);
    let mut a = base.clone();
    let mut b = base;
    let (pa, prior_a, post_a) = a.step(0, 0, true);
    let (pb, prior_b, post_b) = b.step(0, 1, true);
    assert_eq!(pa, pb);
    assert_eq!(prior_a, prior_b);
    assert!((post_a[0] + post_a[1] - 1.0).abs() < 1e-12);
    assert!((post_b[0] + post_b[1] - 1.0).abs() < 1e-12);
}
#[test]
fn unlabeled_mixture_does_not_use_context_labels() {
    let mut a = Models::new(23, 0.05, 0.1);
    let mut b = a.clone();
    let x = Event {
        input: 0,
        target: 1,
        context: 0,
    };
    let y = Event { context: 1, ..x };
    for _ in 0..20 {
        let (pa, _, _) = a.step(x, true);
        let (pb, _, _) = b.step(y, true);
        assert_eq!(pa[0], pb[0]);
        assert_eq!(pa[1], pb[1]);
        assert_eq!(pa[2], pb[2]);
    }
    assert_eq!(a.mixture.p, b.mixture.p);
}
#[test]
fn probe_freezes_model_and_resets_belief() {
    let mut m = Models::new(23, 0.05, 0.1);
    let data = unlabeled_context::stream(80, 120, 1, 0.1);
    for e in &data[..100] {
        m.step(*e, true);
    }
    let copy = m.clone();
    let result = unlabeled_context::probe(&m, &data[..100], 0, 0);
    assert_eq!(result.trials, 100);
    assert_eq!(m.mixture.p, copy.mixture.p);
    assert_eq!(m.mixture.belief, copy.mixture.belief);
    assert_eq!(result, unlabeled_context::probe(&copy, &data[..100], 0, 0));
}
#[test]
fn regimes_and_runs_are_deterministic() {
    let c = config();
    let a = unlabeled_context::run(&c, 23, 0, 0.1);
    let b = unlabeled_context::run(&c, 23, 0, 0.1);
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.probes, b.probes);
    assert_eq!(a.rows.len(), 360);
    assert!(a.rows[..120].iter().all(|x| x.event.context == 0));
    assert!(a.rows[120..240].iter().all(|x| x.event.context == 1));
    assert!(a.rows[240..].iter().all(|x| x.event.context == 0));
}
#[test]
fn seed_roles_do_not_overlap() {
    let a = StreamConfig::parse(include_str!("../configs/unlabeled-context/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/unlabeled-context/test.cfg")).unwrap();
    let offsets = [0u64, 40000, 46000, 48000, 51000];
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

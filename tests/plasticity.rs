use murmur::{
    experiment::{run, Config},
    graph::Graph,
    learning::Readout,
    plasticity::{run_plasticity, trace_step, update_edges, WEIGHT_LIMIT},
};

#[test]
fn eligibility_records_activity_and_zero_feedback_changes_nothing() {
    let mut g = Graph::new(16, 3, 12);
    let mut trace = vec![0.0; g.edges.len()];
    trace_step(&mut g, &mut trace, 0);
    assert!(trace.iter().all(|x| *x == 0.0));
    trace_step(&mut g, &mut trace, 2);
    assert!(trace.iter().any(|x| *x != 0.0));
    let before: Vec<_> = g.edges.iter().map(|e| e.weight).collect();
    let mut r = Readout::new(16, 0.05);
    r.weights[1].fill(1.0);
    update_edges(&mut g, &trace, &r, 0.0);
    assert_eq!(before, g.edges.iter().map(|e| e.weight).collect::<Vec<_>>());
    update_edges(&mut g, &trace, &r, 1.0);
    assert!(g.edges.iter().zip(before).any(|(e, w)| e.weight != w));
    assert!(g.edges.iter().all(|e| e.weight.abs() <= WEIGHT_LIMIT));
}
#[test]
fn matched_baseline_reproducibility_and_evaluation_freeze() {
    let c = Config {
        seed: 2,
        delay: 6,
        train_trials: 150,
        test_trials: 100,
        ..Config::default()
    };
    let (m, learned, shuffled, log) = run_plasticity(&c);
    assert_eq!(m.frozen_correct, run(&c).0.graph_correct);
    assert!(m.learned_mean_abs_change > 0.0 && m.shuffled_mean_abs_change > 0.0);
    let mut second = c.clone();
    second.test_trials = 200;
    let (m2, learned2, shuffled2, log2) = run_plasticity(&second);
    assert_eq!(log, log2);
    assert_eq!(m.learned_mean_abs_change, m2.learned_mean_abs_change);
    assert_eq!(
        learned.edges.iter().map(|e| e.weight).collect::<Vec<_>>(),
        learned2.edges.iter().map(|e| e.weight).collect::<Vec<_>>()
    );
    assert_eq!(
        shuffled.edges.iter().map(|e| e.weight).collect::<Vec<_>>(),
        shuffled2.edges.iter().map(|e| e.weight).collect::<Vec<_>>()
    );
}

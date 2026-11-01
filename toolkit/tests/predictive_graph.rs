use murmur_local_graph::sequence::{Codebook, BITS, N};
use murmur_research_runner::{
    evaluate_seed,
    predictive_graph::{run_seed, PredictiveGraph, LIMIT},
    Config,
};
#[test]
fn updates_are_local_bounded_and_use_target_error() {
    let mut a = PredictiveGraph::new(1, 1.0, true);
    let mut b = a.clone();
    let state = vec![1; N];
    let loss = a.learn_from_state(&state, &vec![1; BITS]);
    b.learn_from_state(&state, &vec![-1; BITS]);
    assert_eq!(loss, 1.0);
    assert!(a.graph.weights[96 * N] > 0.0);
    assert!(b.graph.weights[96 * N] < 0.0);
    assert!(a.graph.weights[..96 * N].iter().all(|w| *w == 0.0));
    for _ in 0..100 {
        a.learn_from_state(&state, &vec![1; BITS]);
    }
    assert!(a
        .graph
        .weights
        .iter()
        .all(|w| w.is_finite() && w.abs() <= LIMIT));
}
#[test]
fn feedback_ablation_and_shared_cues_are_deterministic() {
    let c=Config::parse("schema_version=1\nseeds=72001\npatterns=4\nprefix_noise=0.2\ntrials_per_sequence=2\nsweeps=5\ndensity=0.25\n").unwrap();
    let a = run_seed(&c, 72001);
    let b = run_seed(&c, 72001);
    assert_eq!(a.rows, b.rows);
    let baseline = evaluate_seed(&c, 72001);
    for (x, y) in a.rows.iter().zip(baseline.rows) {
        assert_eq!(x.source, y);
    }
    for i in 96..N {
        for j in 96..N {
            assert!(!a.no_feedback.graph.edges[i * N + j]);
            assert_eq!(a.no_feedback.graph.weights[i * N + j], 0.0);
        }
    }
    let book = Codebook::new(1);
    let empty = PredictiveGraph::new(1, 0.0, true);
    let p = book.encode(&[0, 1, 2, 3]);
    assert!(empty.complete(&p[..96], &vec![1; BITS], &book, 2).converged);
}

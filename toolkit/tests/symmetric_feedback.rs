use murmur_local_graph::sequence::{BITS, N};
use murmur_research_runner::{
    predictive_graph,
    symmetric_feedback::{clamped_energy, run_seed, SymmetricModel},
    Config,
};
#[test]
fn learning_preserves_symmetric_feedback_and_local_sparsity() {
    let mut m = SymmetricModel::new(7, 0.3);
    let input = vec![1; N];
    let target = vec![-1; BITS];
    for _ in 0..10 {
        m.learn_from_state(&input, &target);
    }
    let g = &m.inner.graph;
    for i in 96..N {
        for j in 96..N {
            assert_eq!(g.weights[i * N + j], g.weights[j * N + i]);
            if !g.edges[i * N + j] {
                assert_eq!(g.weights[i * N + j], 0.0);
            }
        }
    }
    assert!(g.weights.iter().all(|w| w.is_finite() && w.abs() <= 1.0));
}
#[test]
fn clamped_energy_does_not_increase_on_asynchronous_sweeps() {
    let mut m = SymmetricModel::new(9, 0.4);
    for p in 0..20 {
        let state: Vec<_> = (0..N)
            .map(|i| if (i * 17 + p * 13) % 7 < 3 { -1 } else { 1 })
            .collect();
        let target: Vec<_> = (0..BITS)
            .map(|i| if (i + p) % 3 == 0 { -1 } else { 1 })
            .collect();
        m.learn_from_state(&state, &target);
    }
    let mut state: Vec<_> = (0..N).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
    let mut clamp = vec![true; N];
    clamp[96..].fill(false);
    for _ in 0..30 {
        let before = clamped_energy(&m.inner.graph, &state);
        m.inner.graph.sweep_clamped(&mut state, &clamp);
        let after = clamped_energy(&m.inner.graph, &state);
        assert!(after <= before + 1e-9);
    }
}
#[test]
fn comparison_uses_same_examples_and_is_reproducible() {
    let c=Config::parse("schema_version=1\nseeds=82001\npatterns=4\nprefix_noise=0.2\ntrials_per_sequence=2\nsweeps=20\ndensity=0.25\n").unwrap();
    let a = run_seed(&c, 82001);
    let b = run_seed(&c, 82001);
    let baseline = predictive_graph::run_seed(&c, 82001);
    assert_eq!(a.rows, b.rows);
    for (x, y) in a.rows.iter().zip(baseline.rows) {
        assert_eq!(x.source, y);
    }
}

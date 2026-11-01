use murmur_local_graph::{evaluate, patterns, Graph};
#[test]
fn edges_are_sparse_symmetric_and_learning_is_local() {
    let mut g = Graph::new(20, 0.4, 1);
    let p = patterns(20, 1, 2).remove(0);
    g.learn(&p);
    for i in 0..g.n {
        for j in 0..g.n {
            assert_eq!(g.edges[i * g.n + j], g.edges[j * g.n + i]);
            assert_eq!(g.weights[i * g.n + j], g.weights[j * g.n + i]);
            if !g.edges[i * g.n + j] {
                assert_eq!(g.weights[i * g.n + j], 0.0);
            }
        }
    }
}
#[test]
fn asynchronous_updates_do_not_increase_energy() {
    let mut g = Graph::new(128, 0.25, 1);
    for p in patterns(128, 4, 2) {
        g.learn(&p);
    }
    let mut x = patterns(128, 1, 3).remove(0);
    for _ in 0..10 {
        let before = g.energy(&x);
        g.sweep(&mut x);
        let after = g.energy(&x);
        assert!(after <= before + 1e-9);
    }
}
#[test]
fn untrained_preserves_cue_and_evaluation_is_reproducible() {
    let g = Graph::new(64, 0.25, 9);
    let p = patterns(64, 3, 8);
    let a = evaluate(&g, &p, 0.2, 10, 7, 10);
    let b = evaluate(&g, &p, 0.2, 10, 7, 10);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(a.graph_correct_bits, a.baseline_correct_bits);
    assert_eq!(a.graph_exact, a.baseline_exact);
}

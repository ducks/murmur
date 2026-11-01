use murmur_local_graph::Graph;
#[test]
fn first_pattern_matches_hebbian_and_preserves_sparsity() {
    let mut h = Graph::new(12, 0.4, 3);
    let mut s = h.clone();
    let p: Vec<_> = (0..12).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
    h.learn(&p);
    s.learn_storkey(&p);
    assert_eq!(h.weights, s.weights);
    for i in 0..s.n {
        for j in 0..s.n {
            assert_eq!(s.weights[i * s.n + j], s.weights[j * s.n + i]);
            if !s.edges[i * s.n + j] {
                assert_eq!(s.weights[i * s.n + j], 0.0);
            }
        }
    }
}
#[test]
fn two_pattern_update_uses_old_fields_and_excludes_direct_edge() {
    let mut g = Graph::new(3, 1.0, 0);
    g.learn_storkey(&[1, 1, 1]);
    g.learn_storkey(&[1, 1, -1]);
    assert!((g.weights[1] - 8.0 / 9.0).abs() < 1e-12);
    assert!(g.weights[2].abs() < 1e-12);
    assert!(g.weights[5].abs() < 1e-12);
}

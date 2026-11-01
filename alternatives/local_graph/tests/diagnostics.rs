use murmur_local_graph::{
    diagnostics::{margin_flips, orthogonal_codebook, run, without_output_feedback},
    sequence::{BITS, N},
    Graph,
};
#[test]
fn orthogonal_codes_and_isolation_are_correct() {
    let b = orthogonal_codebook(3);
    for position in &b.codes {
        for i in 0..4 {
            for j in 0..4 {
                let dot: i32 = position[i]
                    .iter()
                    .zip(&position[j])
                    .map(|(a, b)| *a as i32 * *b as i32)
                    .sum();
                assert_eq!(dot, if i == j { BITS as i32 } else { 0 });
            }
        }
    }
    let mut g = Graph::new(N, 1.0, 1);
    let pattern = b.encode(&[0, 1, 2, 3]);
    g.learn(&pattern);
    assert_eq!(margin_flips(&g, &g, &pattern), 0);
    let h = without_output_feedback(&g);
    for i in 3 * BITS..N {
        for j in 3 * BITS..N {
            assert_eq!(h.weights[i * N + j], 0.0);
        }
    }
    assert_eq!(h.weights[3 * BITS * N], g.weights[3 * BITS * N]);
}
#[test]
fn diagnostic_is_deterministic() {
    assert_eq!(
        format!("{:?}", run(7, 4, 0.2, true, 2)),
        format!("{:?}", run(7, 4, 0.2, true, 2))
    );
}

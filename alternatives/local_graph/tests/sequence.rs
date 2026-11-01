use murmur_local_graph::{
    sequence::{evaluate, graph_predict, lookup, sequences, Codebook, BITS, N},
    Graph,
};
use std::collections::HashSet;
#[test]
fn sequence_prefixes_are_unique_and_lookup_is_correct_on_clean_cues() {
    let book = Codebook::new(1);
    let seqs = sequences(2, 32);
    let mut unique = HashSet::new();
    for s in &seqs {
        assert!(unique.insert([s[0], s[1], s[2]]));
        let p = book.encode(s);
        assert_eq!(lookup(&p[..3 * BITS], &seqs, &book), s[3]);
    }
}
#[test]
fn observed_units_remain_clamped_and_missing_units_are_local() {
    let mut g = Graph::new(N, 0.25, 2);
    let book = Codebook::new(7);
    let s = [0, 1, 2, 3];
    let p = book.encode(&s);
    g.learn(&p);
    let mut state = vec![1; N];
    let mut clamp = vec![true; N];
    clamp[3 * BITS..].fill(false);
    let before = state[..3 * BITS].to_vec();
    g.sweep_clamped(&mut state, &clamp);
    assert_eq!(before, state[..3 * BITS]);
    let missing = vec![1; BITS];
    assert_eq!(
        graph_predict(&g, &p[..3 * BITS], &missing, &book, 10),
        graph_predict(&g, &p[..3 * BITS], &missing, &book, 10)
    );
}
#[test]
fn evaluations_are_reproducible_and_lookup_sees_same_cues() {
    let a = evaluate(4, 8, 0.1, 3, 10);
    let b = evaluate(4, 8, 0.1, 3, 10);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert_eq!(a.trials, 24);
}

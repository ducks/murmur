//! Controlled diagnostics for interference in fixed-length sequence completion.
use crate::{
    sequence::{lookup, sequences, Codebook, BITS, N, POSITIONS, SYMBOLS},
    Graph, Rng,
};

/// Within each position, use balanced orthogonal Walsh codes, multiplied by a
/// seeded position-specific sign mask. Cross-position pairs need not be orthogonal.
pub fn orthogonal_codebook(seed: u64) -> Codebook {
    let mut rng = Rng(seed);
    let mut codes = Vec::new();
    for _ in 0..POSITIONS {
        let mask: Vec<i8> = (0..BITS)
            .map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 })
            .collect();
        let symbols = (1..=SYMBOLS)
            .map(|row| {
                (0..BITS)
                    .map(|bit| {
                        if (row & bit).count_ones() % 2 == 0 {
                            mask[bit]
                        } else {
                            -mask[bit]
                        }
                    })
                    .collect()
            })
            .collect();
        codes.push(symbols);
    }
    Codebook { codes }
}
/// Remove edges among output units but preserve observed-prefix → output edges.
pub fn without_output_feedback(graph: &Graph) -> Graph {
    let mut g = graph.clone();
    let start = (POSITIONS - 1) * BITS;
    for i in start..N {
        for j in start..N {
            g.weights[i * N + j] = 0.0;
            g.edges[i * N + j] = false;
        }
    }
    g
}
/// Fraction of target-output bits whose target-aligned field is positive under
/// the target-only graph but nonpositive under the all-pattern graph. Uses the
/// true full pattern, so this is an oracle diagnostic, not a prediction input.
pub fn margin_flips(full: &Graph, target_only: &Graph, pattern: &[i8]) -> usize {
    let mut flips = 0;
    for i in (POSITIONS - 1) * BITS..N {
        let field = |g: &Graph| -> f64 {
            (0..N)
                .map(|j| g.weights[i * N + j] * pattern[j] as f64)
                .sum::<f64>()
                * pattern[i] as f64
        };
        if field(target_only) > 0.0 && field(full) <= 0.0 {
            flips += 1;
        }
    }
    flips
}
fn state_after(graph: &Graph, prefix: &[i8], missing: &[i8], sweeps: usize) -> Vec<i8> {
    let mut state = prefix.to_vec();
    state.extend_from_slice(missing);
    let mut mask = vec![true; N];
    mask[(POSITIONS - 1) * BITS..].fill(false);
    for _ in 0..sweeps {
        if graph.sweep_clamped(&mut state, &mask) == 0 {
            break;
        }
    }
    state
}
#[derive(Debug)]
pub struct Metrics {
    pub trials: usize,
    pub original: usize,
    pub target_only: usize,
    pub no_feedback: usize,
    pub one_sweep: usize,
    pub lookup: usize,
    pub target_bit_correct: usize,
    pub bit_observations: usize,
    pub flip_bits: usize,
    pub margin_bits: usize,
}
pub fn run(
    seed: u64,
    k: usize,
    noise: f64,
    orthogonal: bool,
    trials_per_sequence: usize,
) -> Metrics {
    assert!((0.0..=1.0).contains(&noise) && trials_per_sequence > 0);
    let book = if orthogonal {
        orthogonal_codebook(seed.wrapping_add(1000))
    } else {
        Codebook::new(seed.wrapping_add(1000))
    };
    let stored = sequences(seed.wrapping_add(2000), k);
    let empty = Graph::new(N, 0.25, seed);
    let mut full = empty.clone();
    for s in &stored {
        full.learn(&book.encode(s));
    }
    let no_feedback = without_output_feedback(&full);
    let mut rng = Rng(seed.wrapping_add(3000));
    let mut m = Metrics {
        trials: 0,
        original: 0,
        target_only: 0,
        no_feedback: 0,
        one_sweep: 0,
        lookup: 0,
        target_bit_correct: 0,
        bit_observations: 0,
        flip_bits: 0,
        margin_bits: 0,
    };
    for seq in &stored {
        let pattern = book.encode(seq);
        let mut oracle = empty.clone();
        oracle.learn(&pattern);
        m.flip_bits += margin_flips(&full, &oracle, &pattern);
        m.margin_bits += BITS;
        for _ in 0..trials_per_sequence {
            let mut prefix = pattern[..(POSITIONS - 1) * BITS].to_vec();
            for x in &mut prefix {
                if rng.uniform() < noise {
                    *x = -*x;
                }
            }
            let missing: Vec<i8> = (0..BITS)
                .map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 })
                .collect();
            let base = state_after(&full, &prefix, &missing, 20);
            let single = state_after(&full, &prefix, &missing, 1);
            let isolated = state_after(&oracle, &prefix, &missing, 20);
            let no_fb = state_after(&no_feedback, &prefix, &missing, 20);
            let start = (POSITIONS - 1) * BITS;
            let answer = seq[POSITIONS - 1];
            m.original += usize::from(book.decode(POSITIONS - 1, &base[start..]) == answer);
            m.target_only += usize::from(book.decode(POSITIONS - 1, &isolated[start..]) == answer);
            m.no_feedback += usize::from(book.decode(POSITIONS - 1, &no_fb[start..]) == answer);
            m.one_sweep += usize::from(book.decode(POSITIONS - 1, &single[start..]) == answer);
            m.lookup += usize::from(lookup(&prefix, &stored, &book) == answer);
            m.target_bit_correct += base[start..]
                .iter()
                .zip(&pattern[start..])
                .filter(|(a, b)| a == b)
                .count();
            m.bit_observations += BITS;
            m.trials += 1;
        }
    }
    m
}

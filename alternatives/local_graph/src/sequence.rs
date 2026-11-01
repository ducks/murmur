//! Fixed-length sequence completion with position-specific symbol codes.
//! This is static associative completion, not a recurrent sequence model.
use crate::{Graph, Rng};
use std::collections::HashSet;
pub const POSITIONS: usize = 4;
pub const SYMBOLS: usize = 4;
pub const BITS: usize = 32;
pub const N: usize = POSITIONS * BITS;

#[derive(Clone)]
pub struct Codebook {
    pub codes: Vec<Vec<Vec<i8>>>,
}
impl Codebook {
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng(seed);
        let codes = (0..POSITIONS)
            .map(|_| {
                (0..SYMBOLS)
                    .map(|_| {
                        (0..BITS)
                            .map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        Self { codes }
    }
    pub fn encode(&self, sequence: &[usize; POSITIONS]) -> Vec<i8> {
        let mut out = Vec::with_capacity(N);
        for (position, &symbol) in sequence.iter().enumerate() {
            out.extend_from_slice(&self.codes[position][symbol]);
        }
        out
    }
    pub fn decode(&self, position: usize, bits: &[i8]) -> usize {
        assert_eq!(bits.len(), BITS);
        let mut best = 0;
        let mut distance = usize::MAX;
        for (symbol, code) in self.codes[position].iter().enumerate() {
            let d = bits.iter().zip(code).filter(|(a, b)| a != b).count();
            if d < distance {
                distance = d;
                best = symbol;
            }
        }
        best
    }
}
/// Distinct three-symbol prefixes guarantee a single correct final symbol per
/// stored sequence. Randomly drawn complete sequences remain the training set.
pub fn sequences(seed: u64, k: usize) -> Vec<[usize; POSITIONS]> {
    assert!(k > 0 && k <= SYMBOLS.pow((POSITIONS - 1) as u32));
    let mut rng = Rng(seed);
    let mut used = HashSet::new();
    let mut result = Vec::new();
    while result.len() < k {
        let seq = std::array::from_fn(|_| (rng.next_u64() as usize) % SYMBOLS);
        if used.insert([seq[0], seq[1], seq[2]]) {
            result.push(seq);
        }
    }
    result
}
/// Exact nearest-neighbor lookup over only the observed prefix bits. Ties
/// choose the first stored sequence; the missing symbol is never inspected.
pub fn lookup(prefix: &[i8], stored: &[[usize; POSITIONS]], book: &Codebook) -> usize {
    assert_eq!(prefix.len(), (POSITIONS - 1) * BITS);
    let mut best = usize::MAX;
    let mut answer = 0;
    for seq in stored {
        let encoded = book.encode(seq);
        let distance = prefix.iter().zip(&encoded).filter(|(a, b)| a != b).count();
        if distance < best {
            best = distance;
            answer = seq[POSITIONS - 1];
        }
    }
    answer
}
/// The final symbol block starts random and observed prefix bits stay clamped.
pub fn graph_predict(
    graph: &Graph,
    prefix: &[i8],
    initial_missing: &[i8],
    book: &Codebook,
    sweeps: usize,
) -> usize {
    let state = graph_complete(graph, prefix, initial_missing, sweeps);
    book.decode(POSITIONS - 1, &state[(POSITIONS - 1) * BITS..])
}

/// Complete the missing block while keeping observed units fixed.
pub fn graph_complete(
    graph: &Graph,
    prefix: &[i8],
    initial_missing: &[i8],
    sweeps: usize,
) -> Vec<i8> {
    assert_eq!(graph.n, N);
    assert_eq!(prefix.len(), (POSITIONS - 1) * BITS);
    assert_eq!(initial_missing.len(), BITS);
    let mut state = prefix.to_vec();
    state.extend_from_slice(initial_missing);
    let mut clamped = vec![true; N];
    clamped[(POSITIONS - 1) * BITS..].fill(false);
    for _ in 0..sweeps {
        if graph.sweep_clamped(&mut state, &clamped) == 0 {
            break;
        }
    }
    state
}
#[derive(Debug)]
pub struct Metrics {
    pub trials: usize,
    pub graph_correct: usize,
    pub lookup_correct: usize,
    pub untrained_correct: usize,
    pub majority_correct: usize,
}
pub fn evaluate(
    seed: u64,
    k: usize,
    noise: f64,
    trials_per_sequence: usize,
    sweeps: usize,
) -> Metrics {
    assert!((0.0..=1.0).contains(&noise) && trials_per_sequence > 0);
    let book = Codebook::new(seed.wrapping_add(1000));
    let stored = sequences(seed.wrapping_add(2000), k);
    let mut graph = Graph::new(N, 0.25, seed);
    let untrained = graph.clone();
    for seq in &stored {
        graph.learn(&book.encode(seq));
    }
    let mut counts = [0usize; SYMBOLS];
    for seq in &stored {
        counts[seq[POSITIONS - 1]] += 1;
    }
    let mut majority = 0;
    for i in 1..SYMBOLS {
        if counts[i] > counts[majority] {
            majority = i;
        }
    }
    let mut rng = Rng(seed.wrapping_add(3000));
    let mut m = Metrics {
        trials: 0,
        graph_correct: 0,
        lookup_correct: 0,
        untrained_correct: 0,
        majority_correct: 0,
    };
    for seq in &stored {
        for _ in 0..trials_per_sequence {
            let mut observed = book.encode(seq)[..(POSITIONS - 1) * BITS].to_vec();
            for bit in &mut observed {
                if rng.uniform() < noise {
                    *bit = -*bit;
                }
            }
            let missing: Vec<i8> = (0..BITS)
                .map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 })
                .collect();
            let target = seq[POSITIONS - 1];
            m.graph_correct +=
                usize::from(graph_predict(&graph, &observed, &missing, &book, sweeps) == target);
            m.untrained_correct += usize::from(
                graph_predict(&untrained, &observed, &missing, &book, sweeps) == target,
            );
            m.lookup_correct += usize::from(lookup(&observed, &stored, &book) == target);
            m.majority_correct += usize::from(majority == target);
            m.trials += 1;
        }
    }
    m
}

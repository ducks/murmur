//! Sparse associative graph: local Hebbian learning and asynchronous dynamics.
pub mod diagnostics;
pub mod sequence;
#[derive(Clone)]
pub struct Rng(pub u64);
impl Rng {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}
#[derive(Clone, Debug)]
pub struct Graph {
    pub n: usize,
    pub edges: Vec<bool>,
    pub weights: Vec<f64>,
}
impl Graph {
    pub fn new(n: usize, density: f64, seed: u64) -> Self {
        assert!(n > 0 && (0.0..=1.0).contains(&density));
        let mut rng = Rng(seed);
        let mut edges = vec![false; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let exists = rng.uniform() < density;
                edges[i * n + j] = exists;
                edges[j * n + i] = exists;
            }
        }
        Self {
            n,
            edges,
            weights: vec![0.0; n * n],
        }
    }
    /// Pairwise Hebbian edge updates; patterns are presented externally.
    pub fn learn(&mut self, pattern: &[i8]) {
        assert_eq!(pattern.len(), self.n);
        assert!(pattern.iter().all(|v| *v == -1 || *v == 1));
        for i in 0..self.n {
            for j in i + 1..self.n {
                if self.edges[i * self.n + j] {
                    let delta = pattern[i] as f64 * pattern[j] as f64 / self.n as f64;
                    self.weights[i * self.n + j] += delta;
                    self.weights[j * self.n + i] += delta;
                }
            }
        }
    }
    /// Sparse-topology Storkey-style update, using pre-update fields and
    /// excluding the direct i-j interaction. All edge updates are simultaneous.
    pub fn learn_storkey(&mut self, pattern: &[i8]) {
        assert_eq!(pattern.len(), self.n);
        assert!(pattern.iter().all(|v| *v == -1 || *v == 1));
        let fields: Vec<f64> = (0..self.n)
            .map(|i| {
                (0..self.n)
                    .map(|j| self.weights[i * self.n + j] * pattern[j] as f64)
                    .sum()
            })
            .collect();
        for i in 0..self.n {
            for j in i + 1..self.n {
                if self.edges[i * self.n + j] {
                    let old = self.weights[i * self.n + j];
                    let xi = pattern[i] as f64;
                    let xj = pattern[j] as f64;
                    let hi_excluding_j = fields[i] - old * xj;
                    let hj_excluding_i = fields[j] - old * xi;
                    let delta =
                        (xi * xj - xi * hj_excluding_i - hi_excluding_j * xj) / self.n as f64;
                    self.weights[i * self.n + j] = old + delta;
                    self.weights[j * self.n + i] = old + delta;
                }
            }
        }
    }
    /// Sequential deterministic local updates; ties preserve the existing state.
    pub fn sweep(&self, state: &mut [i8]) -> usize {
        self.sweep_clamped(state, &vec![false; self.n])
    }
    /// Units marked as clamped preserve their externally supplied values.
    pub fn sweep_clamped(&self, state: &mut [i8], clamped: &[bool]) -> usize {
        assert_eq!(state.len(), self.n);
        assert_eq!(clamped.len(), self.n);
        let mut changes = 0;
        for i in 0..self.n {
            if clamped[i] {
                continue;
            }
            let field: f64 = (0..self.n)
                .map(|j| self.weights[i * self.n + j] * state[j] as f64)
                .sum();
            let next = if field > 0.0 {
                1
            } else if field < 0.0 {
                -1
            } else {
                state[i]
            };
            changes += usize::from(next != state[i]);
            state[i] = next;
        }
        changes
    }
    pub fn energy(&self, state: &[i8]) -> f64 {
        assert_eq!(state.len(), self.n);
        let mut sum = 0.0;
        for i in 0..self.n {
            for j in i + 1..self.n {
                sum -= self.weights[i * self.n + j] * state[i] as f64 * state[j] as f64;
            }
        }
        sum
    }
}
pub fn patterns(n: usize, k: usize, seed: u64) -> Vec<Vec<i8>> {
    let mut rng = Rng(seed);
    (0..k)
        .map(|_| {
            (0..n)
                .map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 })
                .collect()
        })
        .collect()
}
pub fn corrupt(pattern: &[i8], probability: f64, rng: &mut Rng) -> Vec<i8> {
    pattern
        .iter()
        .map(|x| if rng.uniform() < probability { -*x } else { *x })
        .collect()
}
#[derive(Debug)]
pub struct Metrics {
    pub graph_exact: usize,
    pub baseline_exact: usize,
    pub graph_correct_bits: usize,
    pub baseline_correct_bits: usize,
    pub trials: usize,
    pub n: usize,
}
pub fn evaluate(
    graph: &Graph,
    stored: &[Vec<i8>],
    noise: f64,
    trials_per_pattern: usize,
    seed: u64,
    sweeps: usize,
) -> Metrics {
    let mut rng = Rng(seed);
    let mut m = Metrics {
        graph_exact: 0,
        baseline_exact: 0,
        graph_correct_bits: 0,
        baseline_correct_bits: 0,
        trials: stored.len() * trials_per_pattern,
        n: graph.n,
    };
    for target in stored {
        for _ in 0..trials_per_pattern {
            let cue = corrupt(target, noise, &mut rng);
            let mut state = cue.clone();
            for _ in 0..sweeps {
                if graph.sweep(&mut state) == 0 {
                    break;
                }
            }
            let before = cue.iter().zip(target).filter(|(x, y)| x == y).count();
            let after = state.iter().zip(target).filter(|(x, y)| x == y).count();
            m.baseline_correct_bits += before;
            m.graph_correct_bits += after;
            m.baseline_exact += usize::from(before == graph.n);
            m.graph_exact += usize::from(after == graph.n);
        }
    }
    m
}

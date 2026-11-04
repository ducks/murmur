//! Ablation of candidate learning rules with no repeat/flip prior.
//! Initial counts have total strength four per row, matching the candidates.
use murmur::data::Rng;

#[derive(Clone)]
pub struct Candidate {
    counts: [[[f64; 2]; 2]; 2],
    belief: [f64; 2],
    rate: f64,
    predictive_gate: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_slots_stay_equal_but_seeded_offsets_break_symmetry() {
        for predictive in [false, true] {
            let mut equal = Candidate::new(42, 0.05, predictive, false);
            assert_eq!(equal.counts[0], equal.counts[1]);
            for i in 0..100 { equal.observe(i % 2, (i / 3) % 2, true); }
            assert_eq!(equal.counts[0], equal.counts[1]);
            let random = Candidate::new(42, 0.05, predictive, true);
            assert_ne!(random.counts[0], random.counts[1]);
            assert_eq!(random.counts, Candidate::new(42, 0.05, predictive, true).counts);
            for table in random.counts { for row in table { assert!((row[0] + row[1] - 4.0).abs() < 1e-12); } }
        }
    }
}
impl Candidate {
    pub fn new(seed: u64, rate: f64, predictive_gate: bool, break_symmetry: bool) -> Self {
        let mut counts = [[[2.0; 2]; 2]; 2];
        let mut rng = Rng(seed);
        if break_symmetry {
            for table in &mut counts {
                for row in table {
                    // Task-neutral random initialization, independent for each slot and input.
                    let offset = 0.4 * (2.0 * rng.uniform() - 1.0);
                    *row = [2.0 - offset, 2.0 + offset];
                }
            }
        }
        Self { counts, belief: [0.5, 0.5], rate: rate.clamp(0.001, 0.2), predictive_gate }
    }
    fn probability(&self, k: usize, x: usize, y: usize) -> f64 {
        let row = self.counts[k][x]; row[y] / (row[0] + row[1])
    }
    fn context_prior(&self) -> [f64; 2] {
        [0.99 * self.belief[0] + 0.01 * self.belief[1],
         0.99 * self.belief[1] + 0.01 * self.belief[0]]
    }
    pub fn reset(&mut self) { self.belief = [0.5, 0.5]; }
    pub fn predict(&self, x: usize) -> f64 {
        let prior = self.context_prior();
        prior[0] * self.probability(0, x, 1) + prior[1] * self.probability(1, x, 1)
    }
    pub fn observe(&mut self, x: usize, y: usize, learn: bool) {
        let prior = self.context_prior();
        let a = prior[0] * self.probability(0, x, y);
        let b = prior[1] * self.probability(1, x, y);
        self.belief = [a / (a + b), b / (a + b)];
        if learn {
            for k in 0..2 {
                let gate = if self.predictive_gate { prior[k] * prior[k] } else { self.belief[k] };
                let decay = self.rate * 0.02 * if self.predictive_gate { gate } else { 1.0 };
                for input in 0..2 { for target in 0..2 {
                    let c = &mut self.counts[k][input][target];
                    *c += decay * (2.0 - *c);
                }}
                self.counts[k][x][y] += gate;
            }
        }
    }
}

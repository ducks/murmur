//! Two context-specific transition tables with weak, opposing pseudo-count priors.
//! Context is inferred online; neither context labels nor noise are supplied.
#[derive(Clone)]
pub struct Candidate {
    // Beta sufficient statistics, one pair for each context and input symbol.
    counts: [[[f64; 2]; 2]; 2],
    belief: [f64; 2],
    rate: f64,
}

impl Candidate {
    pub fn new(_seed: u64, rate: f64) -> Self {
        let mut counts = [[[0.0; 2]; 2]; 2];
        for k in 0..2 {
            for x in 0..2 {
                for y in 0..2 {
                    counts[k][x][y] = Self::prior(k, x, y);
                }
            }
        }
        Self { counts, belief: [0.5, 0.5], rate: rate.clamp(0.001, 0.2) }
    }

    fn prior(k: usize, x: usize, y: usize) -> f64 {
        if y == (x ^ k) { 3.0 } else { 1.0 }
    }

    fn probability(&self, k: usize, x: usize, y: usize) -> f64 {
        let row = self.counts[k][x];
        row[y] / (row[0] + row[1])
    }

    fn context_prior(&self) -> [f64; 2] {
        let h = 0.01;
        [(1.0 - h) * self.belief[0] + h * self.belief[1],
         (1.0 - h) * self.belief[1] + h * self.belief[0]]
    }

    pub fn reset(&mut self) {
        self.belief = [0.5, 0.5];
    }

    pub fn predict(&self, x: usize) -> f64 {
        let p = self.context_prior();
        p[0] * self.probability(0, x, 1) + p[1] * self.probability(1, x, 1)
    }

    pub fn observe(&mut self, x: usize, y: usize, learn: bool) {
        let prior = self.context_prior();
        let a = prior[0] * self.probability(0, x, y);
        let b = prior[1] * self.probability(1, x, y);
        self.belief = [a / (a + b), b / (a + b)];
        if learn {
            // Discount old evidence towards the prior, then add fractional evidence.
            // Apply discount to both rows so inactivity does not freeze a stale table.
            let decay = self.rate * 0.02;
            for k in 0..2 {
                for input in 0..2 {
                    for target in 0..2 {
                        let c = &mut self.counts[k][input][target];
                        *c += decay * (Self::prior(k, input, target) - *c);
                    }
                }
                self.counts[k][x][y] += self.belief[k];
            }
        }
    }
}

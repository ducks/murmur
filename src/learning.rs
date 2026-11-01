/// Online delta rule: each output synapse uses its presynaptic state and the
/// broadcast prediction error for that output. The graph itself is not trained.
pub struct Readout {
    pub weights: Vec<Vec<f64>>,
    pub rate: f64,
}
impl Readout {
    pub fn new(n: usize, rate: f64) -> Self {
        Self {
            weights: vec![vec![0.0; n + 1]; 2],
            rate,
        }
    }
    pub fn scores(&self, state: &[f64]) -> [f64; 2] {
        let mut result = [0.0; 2];
        for (k, score) in result.iter_mut().enumerate() {
            *score = self.weights[k][state.len()]
                + self.weights[k][..state.len()]
                    .iter()
                    .zip(state)
                    .map(|(w, s)| w * s)
                    .sum::<f64>();
        }
        result
    }
    pub fn predict(&self, state: &[f64]) -> usize {
        let s = self.scores(state);
        usize::from(s[1] > s[0])
    }
    pub fn learn(&mut self, state: &[f64], target: usize) {
        let scores = self.scores(state);
        // Bounded logistic output, with independent binary targets.
        for (k, score) in scores.iter().enumerate() {
            let p = 1.0 / (1.0 + (-score.clamp(-40.0, 40.0)).exp());
            let error = (if k == target { 1.0 } else { 0.0 }) - p;
            for (w, x) in self.weights[k].iter_mut().zip(state) {
                *w += self.rate * error * x;
            }
            self.weights[k][state.len()] += self.rate * error;
        }
    }
}

//! Two latent transition memories with predictive-belief-gated plasticity.
//! The prior context belief gates learning before the current target can
//! influence the assignment, reducing target-dependent cross-contamination.
use crate::unlabeled_context::Mixture;

#[derive(Clone)]
pub struct Candidate {
    inner: Mixture,
}
impl Candidate {
    pub fn new(seed: u64, rate: f64) -> Self {
        Self { inner: Mixture::new(seed, rate) }
    }
    pub fn reset(&mut self) { self.inner.reset(); }
    pub fn predict(&self, x: usize) -> f64 {
        let m = &self.inner;
        let prior = (1.0 - m.hazard) * m.belief[0] + m.hazard * m.belief[1];
        prior * m.p[0][x] + (1.0 - prior) * m.p[1][x]
    }
    pub fn observe(&mut self, x: usize, y: usize, learn: bool) {
        let m = &mut self.inner;
        let prior = [
            (1.0 - m.hazard) * m.belief[0] + m.hazard * m.belief[1],
            (1.0 - m.hazard) * m.belief[1] + m.hazard * m.belief[0],
        ];
        let q = [prior[0] * if y == 1 { m.p[0][x] } else { 1.0 - m.p[0][x] },
                 prior[1] * if y == 1 { m.p[1][x] } else { 1.0 - m.p[1][x] }];
        let total = q[0] + q[1];
        m.belief = [q[0] / total, q[1] / total];
        if learn {
            for k in 0..2 {
                m.p[k][x] = (m.p[k][x] + m.rate * prior[k] * prior[k] * (y as f64 - m.p[k][x])).clamp(0.001, 0.999);
            }
        }
    }
}

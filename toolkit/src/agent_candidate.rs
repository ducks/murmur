//! Candidate model slot for agent-generated experiments.
//!
//! The initial candidate is deliberately identical to the standard mixture.
//! Agents may replace this file while preserving the public interface. The
//! trusted evaluator calls `predict` before `observe` and never passes rule
//! labels or future inputs. `learn=false` must leave learned parameters fixed;
//! transient inference state may still update. `reset` clears inference state.
use crate::unlabeled_context::Mixture;

#[derive(Clone)]
pub struct Candidate {
    inner: Mixture,
}
impl Candidate {
    pub fn new(seed: u64, rate: f64) -> Self {
        Self {
            inner: Mixture::new(seed, rate),
        }
    }
    pub fn reset(&mut self) {
        self.inner.reset();
    }
    pub fn predict(&self, x: usize) -> f64 {
        let m = &self.inner;
        let prior0 = (1.0 - m.hazard) * m.belief[0] + m.hazard * m.belief[1];
        prior0 * m.p[0][x] + (1.0 - prior0) * m.p[1][x]
    }
    pub fn observe(&mut self, x: usize, y: usize, learn: bool) {
        self.inner.step(x, y, learn);
    }
}

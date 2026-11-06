//! Consolidated context memories with evidence-limited cross-row pooling.
use crate::unlabeled_context::Mixture;

#[derive(Clone)]
pub struct Candidate {
    memory: Mixture,
    fast_belief: [f64; 2],
    evidence: [[f64; 2]; 2],
    fast_hazard: f64,
    reversible: bool,
    error_fast: [[f64; 2]; 2],
    error_slow: [[f64; 2]; 2],
    calibrated: bool,
    row_mass: [[f64; 2]; 2],
    pooling: bool,
    relation_log_odds: [f64; 2],
    adaptive_relation: bool,
    alternative_aware_release: bool,
    pending: Option<(usize, usize, [f64; 2])>,
    smoothing: bool,
    switch_guard: bool,
}
impl Candidate {
    pub fn new(seed: u64, rate: f64) -> Self {
        let memory = Mixture::new(seed, rate);
        let fast_hazard = memory.hazard;
        Self { memory, fast_belief: [0.5; 2], evidence: [[0.0; 2]; 2], fast_hazard, reversible: true, error_fast: [[0.5; 2]; 2], error_slow: [[0.5; 2]; 2], calibrated: true, row_mass: [[0.0; 2]; 2], pooling: true, relation_log_odds: [1.0; 2], adaptive_relation: true, alternative_aware_release: true, pending: None, smoothing: true, switch_guard: false }
    }
    /// Seeded correction with switch attenuation, for matched comparison.
    pub fn guarded_smoothing(&self) -> Self {
        let mut model = self.clone();
        model.switch_guard = true;
        model
    }
    /// The preceding candidate's correction, without switch attenuation.
    pub fn unguarded_smoothing(&self) -> Self {
        let mut model = self.clone();
        model.switch_guard = false;
        model
    }
    /// Posterior probability that the hidden state did not change.
    pub fn persistence_probability(posterior: [f64; 2], likelihood: [f64; 2], hazard: f64) -> f64 {
        let stay = (1.0-hazard)*(posterior[0]*likelihood[0]+posterior[1]*likelihood[1]);
        let switch = hazard*(posterior[0]*likelihood[1]+posterior[1]*likelihood[0]);
        stay / (stay + switch)
    }
    /// Matched control without retrospective assignment correction.
    pub fn without_smoothing(&self) -> Self {
        let mut model = self.clone();
        model.smoothing = false;
        model.pending = None;
        model
    }
    /// Matched control: release consolidation without considering the other memory.
    pub fn unguarded_release(&self) -> Self {
        let mut model = self.clone();
        model.alternative_aware_release = false;
        model
    }
    /// Fraction of the release signal that remains when another memory explains y.
    pub fn release_factor(&self, other_likelihood: f64) -> f64 {
        if self.alternative_aware_release { 1.0 - 0.8 * other_likelihood.clamp(0.0, 1.0) } else { 1.0 }
    }
    /// Matched control retaining the fixed opposite-row sharing assumption.
    pub fn fixed_relation(&self) -> Self {
        let mut model = self.clone();
        model.adaptive_relation = false;
        model
    }
    pub fn relation_evidence(&self) -> [f64; 2] { self.relation_log_odds }
    /// Diagnostic snapshot: posterior responsibilities and learned row predictions.
    pub fn diagnostic_state(&self) -> ([f64; 2], [[f64; 2]; 2]) {
        (self.memory.belief, [self.row_predictions(0), self.row_predictions(1)])
    }
    /// Absolute proposed change to the preceding assignment, before learning.
    pub fn diagnostic_correction(&self, x: usize, y: usize) -> f64 {
        if !self.smoothing { return 0.0; }
        let Some((_, _, posterior)) = self.pending else { return 0.0; };
        let p = self.row_predictions(x);
        let l = if y == 1 { p } else { [1.0-p[0], 1.0-p[1]] };
        let h = self.memory.hazard;
        let b = [(1.0-h)*l[0]+h*l[1], h*l[0]+(1.0-h)*l[1]];
        let z = posterior[0]*b[0]+posterior[1]*b[1];
        let guard = if self.switch_guard { Self::persistence_probability(posterior, l, h) } else { 1.0 };
        guard * (posterior[0]*b[0]/z-posterior[0]).abs()
    }
    /// Ablation: retain identical learning machinery but disable cross-row sharing.
    pub fn without_pooling(&self) -> Self {
        let mut model = self.clone();
        model.pooling = false;
        model
    }
    /// Per-slot row predictions after evidence-dependent shrinkage.
    pub fn row_predictions(&self, x: usize) -> [f64; 2] {
        std::array::from_fn(|k| {
            let own = self.memory.p[k][x];
            let other_row = self.memory.p[k][1-x];
            let opposite = if self.adaptive_relation {
                1.0 / (1.0 + (-self.relation_log_odds[k]).exp())
            } else { 1.0 };
            let other = opposite * (1.0-other_row) + (1.0-opposite) * other_row;
            let strength = if self.pooling { 2.0 / (2.0 + self.row_mass[k][x]) } else { 0.0 };
            own + 0.5 * strength * (other - own)
        })
    }
    /// Matched previous release rule, using absolute error directly.
    pub fn absolute_error_control(&self) -> Self {
        let mut model = self.clone();
        model.calibrated = false;
        model
    }
    pub fn error_traces(&self) -> ([[f64; 2]; 2], [[f64; 2]; 2]) {
        (self.error_fast, self.error_slow)
    }
    /// Matched control with irreversible accumulation of consolidation evidence.
    pub fn fixed_consolidation(&self) -> Self {
        let mut model = self.clone();
        model.reversible = false;
        model
    }
    /// Inspect the learned consolidation state in mechanism tests.
    pub fn consolidation(&self) -> [[f64; 2]; 2] { self.evidence }
    /// Control: both filters use the ordinary context switch rate.
    pub fn single_timescale(&self) -> Self {
        let mut model = self.clone();
        model.fast_hazard = model.memory.hazard;
        model
    }
    /// Restore parallel routing on the same learned memories for frozen comparisons.
    pub fn parallel_timescale(&self) -> Self {
        let mut model = self.clone();
        model.fast_hazard = 0.05;
        model
    }
    pub fn reset(&mut self) {
        self.memory.reset();
        self.fast_belief = [0.5; 2];
        self.pending = None;
    }
    fn prior(b: [f64; 2], hazard: f64) -> [f64; 2] {
        [(1.0-hazard)*b[0]+hazard*b[1], (1.0-hazard)*b[1]+hazard*b[0]]
    }
    pub fn predict(&self, x: usize) -> f64 {
        let slow = Self::prior(self.memory.belief, self.memory.hazard);
        let fast = Self::prior(self.fast_belief, self.fast_hazard);
        let w = 0.75*slow[0] + 0.25*fast[0];
        let p = self.row_predictions(x);
        w*p[0] + (1.0-w)*p[1]
    }
    pub fn observe(&mut self, x: usize, y: usize, learn: bool) {
        let predictions = self.row_predictions(x);
        let factors = [self.release_factor(if y == 1 { predictions[1] } else { 1.0-predictions[1] }),
                       self.release_factor(if y == 1 { predictions[0] } else { 1.0-predictions[0] })];
        let m = &mut self.memory;
        let slow = Self::prior(m.belief, m.hazard);
        let fast = Self::prior(self.fast_belief, self.fast_hazard);
        let likelihood = |k: usize| if y == 1 { predictions[k] } else { 1.0-predictions[k] };
        let update = |prior: [f64; 2]| {
            let q = [prior[0]*likelihood(0), prior[1]*likelihood(1)];
            [q[0]/(q[0]+q[1]), q[1]/(q[0]+q[1])]
        };
        m.belief = update(slow);
        self.fast_belief = update(fast);
        if learn {
            // One-step HMM backward message: how well would each previous
            // state explain the current observation after a possible switch?
            // Correct only the previous row update; do not replay inference.
            if self.smoothing {
                if let Some((px, py, posterior)) = self.pending {
                    let h = m.hazard;
                    let l = [likelihood(0), likelihood(1)];
                    let b = [(1.0-h)*l[0]+h*l[1], h*l[0]+(1.0-h)*l[1]];
                    let z = posterior[0]*b[0]+posterior[1]*b[1];
                    let guard = if self.switch_guard { Self::persistence_probability(posterior, l, h) } else { 1.0 };
                    for k in 0..2 {
                        let smoothed = posterior[k]*b[k]/z;
                        let delta = guard * (smoothed - posterior[k]);
                        let scale = 1.0/(1.0+0.004*self.evidence[k][px]);
                        m.p[k][px] = (m.p[k][px] + 0.5*m.rate*scale*delta*(py as f64-m.p[k][px])).clamp(0.001, 0.999);
                    }
                }
            }
            self.pending = Some((x, y, m.belief));
            for k in 0..2 {
                let responsibility = 0.75*m.belief[k] + 0.25*self.fast_belief[k];
                let weight = responsibility * responsibility.sqrt().sqrt();
                // Compare same-row and opposite-row hypotheses using the other
                // row's pre-update prediction. Discount old evidence so a stale
                // relationship can change; do not learn from unexposed rows.
                if self.adaptive_relation {
                    let other = m.p[k][1-x].clamp(0.01, 0.99);
                    let same_likelihood = if y == 1 { other } else { 1.0-other };
                    let opposite_likelihood = 1.0-same_likelihood;
                    let exposure = self.row_mass[k][1-x] / (10.0+self.row_mass[k][1-x]);
                    self.relation_log_odds[k] = (self.relation_log_odds[k] * (1.0-0.005*weight)
                        + 0.05*weight*exposure*(opposite_likelihood/same_likelihood).ln()).clamp(-4.0, 4.0);
                }
                self.row_mass[k][x] = (self.row_mass[k][x] + weight).min(10000.0);
                let scale = 1.0/(1.0+0.004*self.evidence[k][x]);
                m.p[k][x] = (m.p[k][x]+m.rate*weight*scale*(y as f64-m.p[k][x])).clamp(0.001,0.999);
                // Persistent surprising observations assigned to this slot gradually
                // release its consolidation, allowing a stale rule to be revised.
                let surprise = (y as f64 - m.p[k][x]).abs();
                let fast = &mut self.error_fast[k][x];
                let slow = &mut self.error_slow[k][x];
                *fast += 0.08 * weight * (surprise - *fast);
                *slow += 0.004 * weight * (surprise - *slow);
                let excess = (*fast - *slow).max(0.0);
                let signal = if self.calibrated { excess } else { surprise };
                let release = if self.reversible { 0.02 * weight * signal * factors[k] } else { 0.0 };
                self.evidence[k][x] = (self.evidence[k][x] * (1.0-release)
                    + weight*responsibility * if self.reversible { 1.0-surprise } else { 1.0 }).min(250.0);
            }
        } else {
            self.pending = None;
        }
    }
}

//! A separate one-sided CUSUM-style change alarm against a uniform alternative.
//! It reports an alarm only and does not create rule memories.
#[derive(Debug, Clone)]
pub struct Cusum {
    counts: [[u64; 4]; 4],
    pub evidence: f64,
    steps: usize,
    pub threshold: f64,
    pub first_alarm: Option<usize>,
}
impl Cusum {
    pub fn new(threshold: f64) -> Self {
        assert!(threshold.is_finite() && threshold > 0.0);
        Self {
            counts: [[1; 4]; 4],
            evidence: 0.0,
            steps: 0,
            threshold,
            first_alarm: None,
        }
    }
    /// Score before adding the observation. Cumulative counts define the null;
    /// a uniform next-symbol distribution is the alternative. Suppress alarms
    /// for the first 100 observations, then accumulate log likelihood ratios.
    pub fn observe(&mut self, source: usize, target: usize) {
        assert!(source < 4 && target < 4);
        let row = &self.counts[source];
        let p = row[target] as f64 / row.iter().sum::<u64>() as f64;
        self.steps += 1;
        if self.steps > 100 {
            self.evidence = (self.evidence + (0.25 / p).ln()).max(0.0);
            if self.evidence > self.threshold && self.first_alarm.is_none() {
                self.first_alarm = Some(self.steps);
            }
        }
        self.counts[source][target] += 1;
    }
}

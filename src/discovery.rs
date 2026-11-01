//! Exploratory online mixture of transition models, with surprise-gated growth.
//! This is a standard hidden-state mixture extended by a simple growth trigger,
//! not an implementation of locally emerging graph dynamics.
use crate::{data::Rng, sequence_engine::SequenceEngine};

pub const SYMBOLS: usize = 4;
pub const HAZARD: f64 = 0.02;
pub const RATE: f64 = 0.15;
pub const WINDOW: usize = 25;
pub const WARMUP: usize = 100;
pub const THRESHOLD: f64 = 0.85;
pub const SUSTAINED: usize = 2;
pub const MAX_MODELS: usize = 2;

#[derive(Clone, Debug)]
pub struct Event {
    pub step: usize,
    pub mean_surprise: f64,
    pub new_count: usize,
}
#[derive(Clone, Debug)]
pub struct Discovery {
    pub models: Vec<SequenceEngine>,
    pub posterior: Vec<f64>,
    pub events: Vec<Event>,
    pub allow_growth: bool,
    rng: RngState,
    steps: usize,
    window_sum: f64,
    window_count: usize,
    high_windows: usize,
}
// Keep the generator's state inspectable and cloneable, without exposing labels.
#[derive(Clone, Debug)]
struct RngState(u64);
impl RngState {
    fn next(&mut self) -> u64 {
        let mut r = Rng(self.0);
        let v = r.next_u64();
        self.0 = r.0;
        v
    }
}
#[derive(Debug)]
pub struct Observation {
    pub prediction: usize,
    pub surprise: f64,
    pub inferred: usize,
    pub grown: bool,
}
impl Discovery {
    pub fn new(initial_models: usize, allow_growth: bool, seed: u64) -> Self {
        assert!((1..=MAX_MODELS).contains(&initial_models));
        let mut this = Self {
            models: vec![],
            posterior: vec![],
            events: vec![],
            allow_growth,
            rng: RngState(seed),
            steps: 0,
            window_sum: 0.0,
            window_count: 0,
            high_windows: 0,
        };
        for i in 0..initial_models {
            let model = if initial_models == 1 && i == 0 {
                SequenceEngine::new(SYMBOLS, RATE)
            } else {
                this.random_model()
            };
            this.models.push(model);
        }
        this.posterior = vec![1.0 / initial_models as f64; initial_models];
        this
    }
    fn random_model(&mut self) -> SequenceEngine {
        let mut m = SequenceEngine::new(SYMBOLS, RATE);
        for row in &mut m.weights {
            let favored = (self.rng.next() as usize) % SYMBOLS;
            for (j, w) in row.iter_mut().enumerate() {
                *w = 0.7 * (1.0 / SYMBOLS as f64) + if j == favored { 0.3 } else { 0.0 };
            }
        }
        m
    }
    pub fn reset_belief(&mut self) {
        let n = self.models.len();
        self.posterior = vec![1.0 / n as f64; n];
    }
    /// Make a separate experimental branch with one additional candidate table.
    /// The original model is unchanged; callers decide whether to commit it.
    pub fn candidate_branch(&self) -> Option<Self> {
        if self.models.len() >= MAX_MODELS {
            return None;
        }
        let mut branch = self.clone();
        let new_model = branch.random_model();
        branch.models.push(new_model);
        branch.reset_belief();
        Some(branch)
    }
    /// Predict before observing target. When learning is false only the belief
    /// changes; weights and topology are frozen and no surprise windows accrue.
    pub fn step(&mut self, source: usize, target: usize, learn: bool) -> Observation {
        assert!(source < SYMBOLS && target < SYMBOLS);
        let n = self.models.len();
        let prior: Vec<f64> = if n == 1 {
            vec![1.0]
        } else {
            (0..n)
                .map(|i| {
                    (1.0 - HAZARD) * self.posterior[i]
                        + HAZARD * (1.0 - self.posterior[i]) / (n - 1) as f64
                })
                .collect()
        };
        let mut scores = [0.0; SYMBOLS];
        for (model, p) in self.models.iter().zip(&prior) {
            for (s, w) in scores.iter_mut().zip(&model.weights[source]) {
                *s += p * w;
            }
        }
        let mut prediction = 0;
        for i in 1..SYMBOLS {
            if scores[i] > scores[prediction] {
                prediction = i;
            }
        }
        let likelihood = scores[target].max(1e-15);
        let surprise = -likelihood.ln();
        self.posterior = (0..n)
            .map(|i| prior[i] * self.models[i].weights[source][target] / likelihood)
            .collect();
        let mut inferred = 0;
        for i in 1..n {
            if self.posterior[i] > self.posterior[inferred] {
                inferred = i;
            }
        }
        let mut grown = false;
        if learn {
            for (model, responsibility) in self.models.iter_mut().zip(&self.posterior) {
                for (dest, w) in model.weights[source].iter_mut().enumerate() {
                    let label = if dest == target { 1.0 } else { 0.0 };
                    *w += RATE * responsibility * (label - *w);
                }
            }
            self.steps += 1;
            self.window_sum += surprise;
            self.window_count += 1;
            if self.window_count == WINDOW {
                let mean = self.window_sum / WINDOW as f64;
                if self.steps >= WARMUP && mean > THRESHOLD {
                    self.high_windows += 1;
                } else {
                    self.high_windows = 0;
                }
                if self.allow_growth
                    && self.models.len() < MAX_MODELS
                    && self.high_windows >= SUSTAINED
                {
                    let model = self.random_model();
                    self.models.push(model);
                    self.reset_belief();
                    self.events.push(Event {
                        step: self.steps,
                        mean_surprise: mean,
                        new_count: self.models.len(),
                    });
                    self.high_windows = 0;
                    grown = true;
                }
                self.window_sum = 0.0;
                self.window_count = 0;
            }
        }
        Observation {
            prediction,
            surprise,
            inferred,
            grown,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Example {
    pub source: usize,
    pub target: usize,
    pub context: usize,
}
/// Labels are only for scoring. Switching streams alternate every 100 steps.
pub fn stream(seed: u64, n: usize, switching: bool, noise: f64) -> Vec<Example> {
    stream_with_block(seed, n, switching, noise, 100)
}

pub fn stream_with_block(
    seed: u64,
    n: usize,
    switching: bool,
    noise: f64,
    block: usize,
) -> Vec<Example> {
    assert!((0.0..=1.0).contains(&noise));
    assert!(block > 0);
    let mut rng = Rng(seed);
    let mut source = (rng.next_u64() as usize) % SYMBOLS;
    let mut out = Vec::new();
    for i in 0..n {
        let context = if switching { (i / block) % 2 } else { 0 };
        let desired = if context == 0 {
            (source + 1) % SYMBOLS
        } else {
            (source + SYMBOLS - 1) % SYMBOLS
        };
        let target = if rng.uniform() < noise {
            let a = (rng.next_u64() as usize) % (SYMBOLS - 1);
            if a >= desired {
                a + 1
            } else {
                a
            }
        } else {
            desired
        };
        out.push(Example {
            source,
            target,
            context,
        });
        source = target;
    }
    out
}
#[derive(Debug)]
pub struct Experiment {
    pub auto: Discovery,
    pub fixed: Discovery,
    pub single: Discovery,
    pub rows: Vec<(Example, usize, usize, usize, usize, f64)>,
    pub evaluation: Vec<(Example, usize, usize, usize, usize, usize)>,
}
pub fn run(seed: u64, switching: bool, noise: f64) -> Experiment {
    let mut auto = Discovery::new(1, true, seed.wrapping_add(401));
    let mut fixed = Discovery::new(2, false, seed.wrapping_add(401));
    let mut single = Discovery::new(1, false, seed.wrapping_add(401));
    let mut rows = Vec::new();
    for e in stream(seed.wrapping_add(402), 1200, switching, noise) {
        let a = auto.step(e.source, e.target, true);
        let f = fixed.step(e.source, e.target, true);
        let s = single.step(e.source, e.target, true);
        rows.push((
            e,
            a.prediction,
            f.prediction,
            s.prediction,
            auto.models.len(),
            a.surprise,
        ));
    }
    auto.reset_belief();
    fixed.reset_belief();
    single.reset_belief();
    let mut evaluation = Vec::new();
    for e in stream(seed.wrapping_add(403), 400, switching, noise) {
        let a = auto.step(e.source, e.target, false);
        let f = fixed.step(e.source, e.target, false);
        let s = single.step(e.source, e.target, false);
        evaluation.push((
            e,
            a.prediction,
            f.prediction,
            s.prediction,
            a.inferred,
            f.inferred,
        ));
    }
    Experiment {
        auto,
        fixed,
        single,
        rows,
        evaluation,
    }
}

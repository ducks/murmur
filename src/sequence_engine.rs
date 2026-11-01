//! A deliberately minimal, inspectable online sequence engine.
//!
//! Each symbol is a unit; its outgoing edges represent predictions of the next
//! symbol. This is an exponentially weighted first-order Markov model expressed
//! as a graph, not yet the recurrent substrate or self-organizing architecture.
use crate::data::Rng;

#[derive(Clone, Debug)]
pub struct SequenceEngine {
    pub weights: Vec<Vec<f64>>,
    pub previous: Option<usize>,
    pub rate: f64,
}
impl SequenceEngine {
    pub fn new(symbols: usize, rate: f64) -> Self {
        assert!(symbols > 1 && rate.is_finite() && rate > 0.0 && rate <= 1.0);
        Self {
            weights: vec![vec![1.0 / symbols as f64; symbols]; symbols],
            previous: None,
            rate,
        }
    }
    pub fn reset_context(&mut self) {
        self.previous = None;
    }
    pub fn predict_from(&self, symbol: usize) -> usize {
        argmax(&self.weights[symbol])
    }
    pub fn predict(&self) -> Option<usize> {
        self.previous.map(|s| self.predict_from(s))
    }
    pub fn learn_transition(&mut self, from: usize, to: usize) {
        assert!(from < self.weights.len() && to < self.weights.len());
        for (destination, weight) in self.weights[from].iter_mut().enumerate() {
            let target = if destination == to { 1.0 } else { 0.0 };
            *weight += self.rate * (target - *weight);
        }
    }
    /// Returns the prediction made before seeing `symbol`, then learns the
    /// transition if enabled. The first symbol establishes context only.
    pub fn observe(&mut self, symbol: usize, learn: bool) -> Option<usize> {
        assert!(symbol < self.weights.len());
        let predicted = self.predict();
        if let (Some(previous), true) = (self.previous, learn) {
            self.learn_transition(previous, symbol);
        }
        self.previous = Some(symbol);
        predicted
    }
}
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    best
}
/// Stationary count baseline with a uniform one-count prior, no forgetting.
#[derive(Clone)]
pub struct Markov {
    pub counts: Vec<Vec<u64>>,
    pub previous: Option<usize>,
}
impl Markov {
    pub fn new(n: usize) -> Self {
        assert!(n > 1);
        Self {
            counts: vec![vec![1; n]; n],
            previous: None,
        }
    }
    pub fn predict_from(&self, symbol: usize) -> usize {
        let row = &self.counts[symbol];
        let mut best = 0;
        for i in 1..row.len() {
            if row[i] > row[best] {
                best = i;
            }
        }
        best
    }
    pub fn observe(&mut self, symbol: usize) -> Option<usize> {
        assert!(symbol < self.counts.len());
        let prediction = self.previous.map(|s| self.predict_from(s));
        if let Some(s) = self.previous {
            self.counts[s][symbol] += 1;
        }
        self.previous = Some(symbol);
        prediction
    }
}

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub seed: u64,
    pub symbols: usize,
    pub train_steps: usize,
    pub change_steps: usize,
    pub rate: f64,
    pub noise: f64,
    pub window: usize,
    pub recovery_threshold: f64,
}
impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            symbols: 4,
            train_steps: 600,
            change_steps: 400,
            rate: 0.15,
            noise: 0.1,
            window: 50,
            recovery_threshold: 0.8,
        }
    }
}
#[derive(Debug)]
pub struct Step {
    pub phase: &'static str,
    pub index: usize,
    pub from: usize,
    pub target: usize,
    pub engine: usize,
    pub frozen: usize,
    pub markov: usize,
}
#[derive(Debug)]
pub struct ResultSet {
    pub steps: Vec<Step>,
    pub train_weights: Vec<Vec<f64>>,
    pub final_weights: Vec<Vec<f64>>,
    pub old_recall: [f64; 3],
    pub new_recall: [f64; 3],
    pub recovery: [Option<usize>; 3],
}
/// Phase A goes clockwise, phase B counterclockwise. With probability `noise`,
/// the next symbol is sampled uniformly among the other destinations. The
/// stream is shared by all models and scored prequentially (predict, then learn).
pub fn run(c: &EngineConfig) -> Result<ResultSet, &'static str> {
    if c.symbols < 3
        || c.train_steps == 0
        || c.change_steps == 0
        || c.window == 0
        || c.window > c.change_steps
        || !c.rate.is_finite()
        || c.rate <= 0.0
        || c.rate > 1.0
        || !c.noise.is_finite()
        || !(0.0..=1.0).contains(&c.noise)
        || !c.recovery_threshold.is_finite()
        || !(0.0..=1.0).contains(&c.recovery_threshold)
    {
        return Err("invalid configuration");
    }
    let mut rng = Rng(c.seed);
    let mut engine = SequenceEngine::new(c.symbols, c.rate);
    let mut frozen = engine.clone();
    let mut markov = Markov::new(c.symbols);
    let mut current = (rng.next_u64() as usize) % c.symbols;
    engine.observe(current, true);
    frozen.observe(current, true);
    markov.observe(current);
    let mut steps = Vec::with_capacity(c.train_steps + c.change_steps);
    let mut train_weights = Vec::new();
    for i in 0..(c.train_steps + c.change_steps) {
        let changed = i >= c.train_steps;
        let desired = if changed {
            (current + c.symbols - 1) % c.symbols
        } else {
            (current + 1) % c.symbols
        };
        let target = if rng.uniform() < c.noise {
            let alt = (rng.next_u64() as usize) % (c.symbols - 1);
            if alt >= desired {
                alt + 1
            } else {
                alt
            }
        } else {
            desired
        };
        let ep = engine.observe(target, true).unwrap();
        let fp = frozen.observe(target, !changed).unwrap();
        let mp = markov.observe(target).unwrap();
        steps.push(Step {
            phase: if changed { "changed" } else { "train" },
            index: if changed {
                i - c.train_steps + 1
            } else {
                i + 1
            },
            from: current,
            target,
            engine: ep,
            frozen: fp,
            markov: mp,
        });
        current = target;
        if i + 1 == c.train_steps {
            train_weights = engine.weights.clone();
        }
    }
    let mut old = [0.0; 3];
    let mut new = [0.0; 3];
    for from in 0..c.symbols {
        let preds = [
            engine.predict_from(from),
            frozen.predict_from(from),
            markov.predict_from(from),
        ];
        for k in 0..3 {
            old[k] += f64::from(u8::from(preds[k] == (from + 1) % c.symbols));
            new[k] += f64::from(u8::from(preds[k] == (from + c.symbols - 1) % c.symbols));
        }
    }
    for x in old.iter_mut().chain(new.iter_mut()) {
        *x /= c.symbols as f64;
    }
    let changed = &steps[c.train_steps..];
    let mut recovery = [None; 3];
    for k in 0..3 {
        for end in c.window..=changed.len() {
            let count = changed[end - c.window..end]
                .iter()
                .filter(|s| [s.engine, s.frozen, s.markov][k] == s.target)
                .count();
            if count as f64 / c.window as f64 >= c.recovery_threshold {
                recovery[k] = Some(end);
                break;
            }
        }
    }
    Ok(ResultSet {
        steps,
        train_weights,
        final_weights: engine.weights,
        old_recall: old,
        new_recall: new,
        recovery,
    })
}

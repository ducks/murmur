//! Explicitly cued context: separate predictive edges per (context, source).
//! This is a context-conditioned Markov model; context is supplied externally.
use crate::{
    data::Rng,
    sequence_engine::{EngineConfig, Markov, SequenceEngine},
};

#[derive(Clone, Debug)]
pub struct ContextEngine {
    pub contexts: Vec<SequenceEngine>,
    pub previous: Option<usize>,
}
impl ContextEngine {
    pub fn new(symbols: usize, contexts: usize, rate: f64) -> Self {
        assert!(contexts > 0);
        Self {
            contexts: (0..contexts)
                .map(|_| SequenceEngine::new(symbols, rate))
                .collect(),
            previous: None,
        }
    }
    /// Context describes the transition into `symbol` and is known before prediction.
    pub fn predict_from(&self, context: usize, source: usize) -> usize {
        self.contexts[context].predict_from(source)
    }
    pub fn predict(&self, context: usize) -> Option<usize> {
        self.previous
            .map(|source| self.predict_from(context, source))
    }
    pub fn observe(&mut self, context: usize, symbol: usize, learn: bool) -> Option<usize> {
        assert!(context < self.contexts.len() && symbol < self.contexts[0].weights.len());
        let prediction = self.predict(context);
        if let (Some(from), true) = (self.previous, learn) {
            self.contexts[context].learn_transition(from, symbol);
        }
        self.previous = Some(symbol);
        prediction
    }
    pub fn reset_sequence(&mut self) {
        self.previous = None;
    }
}
/// Cumulative first-order counts with the same externally supplied context.
struct ContextCounts {
    contexts: Vec<Markov>,
    previous: Option<usize>,
}
impl ContextCounts {
    fn new(symbols: usize, contexts: usize) -> Self {
        Self {
            contexts: (0..contexts).map(|_| Markov::new(symbols)).collect(),
            previous: None,
        }
    }
    fn predict_from(&self, context: usize, source: usize) -> usize {
        self.contexts[context].predict_from(source)
    }
    fn observe(&mut self, context: usize, symbol: usize) -> Option<usize> {
        let prediction = self
            .previous
            .map(|source| self.predict_from(context, source));
        if let Some(source) = self.previous {
            self.contexts[context].counts[source][symbol] += 1;
        }
        self.previous = Some(symbol);
        prediction
    }
}
#[derive(Debug)]
pub struct ContextStep {
    pub phase: &'static str,
    pub index: usize,
    pub context: usize,
    pub source: usize,
    pub target: usize,
    pub context_prediction: usize,
    pub plain_prediction: usize,
    pub counts_prediction: usize,
}
#[derive(Debug)]
pub struct ContextResult {
    pub steps: Vec<ContextStep>,
    /// Rows: model, columns: original rule and new rule after phase B, before replay.
    pub after_b: [[f64; 2]; 3],
    pub final_recall: [[f64; 2]; 3],
    pub edges_after_a: Vec<Vec<Vec<f64>>>,
    pub edges_after_b: Vec<Vec<Vec<f64>>>,
}
fn recall(
    graph: &ContextEngine,
    plain: &SequenceEngine,
    counts: &ContextCounts,
    n: usize,
) -> [[f64; 2]; 3] {
    let mut values = [[0.0; 2]; 3];
    for context in 0..2 {
        for source in 0..n {
            let target = if context == 0 {
                (source + 1) % n
            } else {
                (source + n - 1) % n
            };
            let pred = [
                graph.predict_from(context, source),
                plain.predict_from(source),
                counts.predict_from(context, source),
            ];
            for model in 0..3 {
                values[model][context] += f64::from(u8::from(pred[model] == target)) / n as f64;
            }
        }
    }
    values
}
fn edges(model: &ContextEngine) -> Vec<Vec<Vec<f64>>> {
    model.contexts.iter().map(|g| g.weights.clone()).collect()
}
/// Phase A, phase B, then four alternating 50-transition blocks. The context
/// cue is available before each prediction; it is never inferred by the model.
pub fn run(c: &EngineConfig) -> Result<ContextResult, &'static str> {
    if c.symbols < 3
        || c.train_steps == 0
        || c.change_steps == 0
        || !c.rate.is_finite()
        || c.rate <= 0.0
        || c.rate > 1.0
        || !c.noise.is_finite()
        || !(0.0..=1.0).contains(&c.noise)
    {
        return Err("invalid configuration");
    }
    let mut rng = Rng(c.seed);
    let mut graph = ContextEngine::new(c.symbols, 2, c.rate);
    let mut plain = SequenceEngine::new(c.symbols, c.rate);
    let mut counts = ContextCounts::new(c.symbols, 2);
    let mut current = (rng.next_u64() as usize) % c.symbols;
    graph.observe(0, current, true);
    plain.observe(current, true);
    counts.observe(0, current);
    let mut steps = Vec::new();
    let mut after_b = [[0.0; 2]; 3];
    let mut edges_after_a = Vec::new();
    let mut edges_after_b = Vec::new();
    for i in 0..c.train_steps + c.change_steps + 200 {
        let (phase, index, context) = if i < c.train_steps {
            ("A", i + 1, 0)
        } else if i < c.train_steps + c.change_steps {
            ("B", i - c.train_steps + 1, 1)
        } else {
            let j = i - c.train_steps - c.change_steps;
            ("alternating", j + 1, (j / 50) % 2)
        };
        let desired = if context == 0 {
            (current + 1) % c.symbols
        } else {
            (current + c.symbols - 1) % c.symbols
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
        let cp = graph.observe(context, target, true).unwrap();
        let pp = plain.observe(target, true).unwrap();
        let mp = counts.observe(context, target).unwrap();
        steps.push(ContextStep {
            phase,
            index,
            context,
            source: current,
            target,
            context_prediction: cp,
            plain_prediction: pp,
            counts_prediction: mp,
        });
        current = target;
        if i + 1 == c.train_steps {
            edges_after_a = edges(&graph);
        }
        if i + 1 == c.train_steps + c.change_steps {
            after_b = recall(&graph, &plain, &counts, c.symbols);
            edges_after_b = edges(&graph);
        }
    }
    Ok(ContextResult {
        steps,
        after_b,
        final_recall: recall(&graph, &plain, &counts, c.symbols),
        edges_after_a,
        edges_after_b,
    })
}

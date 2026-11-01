//! Inference over two pre-trained, externally labelled transition models.
//! The Bayesian filter receives symbols only during evaluation, never labels.
use crate::{
    context_engine::ContextEngine,
    data::Rng,
    sequence_engine::{EngineConfig, SequenceEngine},
};

#[derive(Clone, Debug)]
pub struct Filter {
    pub posterior: [f64; 2],
    pub switch_probability: f64,
}
impl Filter {
    pub fn new(switch_probability: f64) -> Self {
        assert!(switch_probability.is_finite() && (0.0..=0.5).contains(&switch_probability));
        Self {
            posterior: [0.5, 0.5],
            switch_probability,
        }
    }
    pub fn prior(&self) -> [f64; 2] {
        let h = self.switch_probability;
        [
            (1.0 - h) * self.posterior[0] + h * self.posterior[1],
            (1.0 - h) * self.posterior[1] + h * self.posterior[0],
        ]
    }
    /// Multiply the predicted context probabilities by P(observed symbol | context).
    pub fn update(&mut self, likelihood: [f64; 2]) {
        assert!(likelihood.iter().all(|x| x.is_finite() && *x > 0.0));
        let p = self.prior();
        let a = p[0] * likelihood[0];
        let b = p[1] * likelihood[1];
        self.posterior = [a / (a + b), b / (a + b)];
    }
}
fn argmax(xs: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best
}
/// Mixture prediction before the next symbol is observed.
pub fn predict(rows: [&[f64]; 2], prior: [f64; 2]) -> usize {
    assert_eq!(rows[0].len(), rows[1].len());
    let scores: Vec<_> = (0..rows[0].len())
        .map(|i| prior[0] * rows[0][i] + prior[1] * rows[1][i])
        .collect();
    argmax(&scores)
}
fn next(rng: &mut Rng, source: usize, context: usize, n: usize, noise: f64) -> usize {
    let expected = if context == 0 {
        (source + 1) % n
    } else {
        (source + n - 1) % n
    };
    if rng.uniform() < noise {
        let alt = (rng.next_u64() as usize) % (n - 1);
        if alt >= expected {
            alt + 1
        } else {
            alt
        }
    } else {
        expected
    }
}
#[derive(Debug)]
pub struct HiddenStep {
    pub index: usize,
    pub context: usize,
    pub source: usize,
    pub target: usize,
    pub graph_prediction: usize,
    pub count_prediction: usize,
    pub oracle_prediction: usize,
    pub plain_prediction: usize,
    pub prior_a: f64,
    pub posterior_a: f64,
    pub counts_posterior_a: f64,
}
#[derive(Debug)]
pub struct HiddenResult {
    pub steps: Vec<HiddenStep>,
    pub training_weights: Vec<Vec<Vec<f64>>>,
}
/// Labelled training of both emission tables, then fresh evaluation with labels
/// withheld from both filters. Oracle labels are used only by oracle prediction
/// and for scoring/logging. The no-context reader keeps learning online.
pub fn run(
    c: &EngineConfig,
    hazard: f64,
    block: usize,
    blocks: usize,
) -> Result<HiddenResult, &'static str> {
    if c.symbols < 3
        || c.train_steps == 0
        || c.change_steps == 0
        || block == 0
        || blocks == 0
        || !c.rate.is_finite()
        || c.rate <= 0.0
        || c.rate > 1.0
        || !c.noise.is_finite()
        || !(0.0..=1.0).contains(&c.noise)
        || !hazard.is_finite()
        || !(0.0..=0.5).contains(&hazard)
    {
        return Err("invalid configuration");
    }
    let mut train_rng = Rng(c.seed.wrapping_add(201));
    let mut graph = ContextEngine::new(c.symbols, 2, c.rate);
    let mut counts = vec![vec![vec![1u64; c.symbols]; c.symbols]; 2];
    let mut source = (train_rng.next_u64() as usize) % c.symbols;
    graph.observe(0, source, true);
    let mut plain = SequenceEngine::new(c.symbols, c.rate);
    plain.observe(source, true);
    for i in 0..c.train_steps + c.change_steps {
        let context = usize::from(i >= c.train_steps);
        let target = next(&mut train_rng, source, context, c.symbols, c.noise);
        graph.observe(context, target, true);
        counts[context][source][target] += 1;
        plain.observe(target, true);
        source = target;
    }
    let training_weights: Vec<_> = graph.contexts.iter().map(|g| g.weights.clone()).collect();
    let count_probs: Vec<Vec<Vec<f64>>> = counts
        .iter()
        .map(|g| {
            g.iter()
                .map(|row| {
                    let sum = row.iter().sum::<u64>() as f64;
                    row.iter().map(|v| *v as f64 / sum).collect()
                })
                .collect()
        })
        .collect();
    let mut graph_filter = Filter::new(hazard);
    let mut count_filter = Filter::new(hazard);
    let mut eval_rng = Rng(c.seed.wrapping_add(202));
    source = (eval_rng.next_u64() as usize) % c.symbols;
    plain.reset_context();
    plain.observe(source, false);
    let mut steps = Vec::new();
    for i in 0..block * blocks {
        let context = (i / block) % 2;
        let prior = graph_filter.prior();
        let graph_rows = [
            &training_weights[0][source][..],
            &training_weights[1][source][..],
        ];
        let count_rows = [&count_probs[0][source][..], &count_probs[1][source][..]];
        let graph_prediction = predict(graph_rows, prior);
        let count_prediction = predict(count_rows, count_filter.prior());
        let oracle_prediction = argmax(graph_rows[context]);
        let target = next(&mut eval_rng, source, context, c.symbols, c.noise);
        let plain_prediction = plain.observe(target, true).unwrap();
        graph_filter.update([graph_rows[0][target], graph_rows[1][target]]);
        count_filter.update([count_rows[0][target], count_rows[1][target]]);
        steps.push(HiddenStep {
            index: i + 1,
            context,
            source,
            target,
            graph_prediction,
            count_prediction,
            oracle_prediction,
            plain_prediction,
            prior_a: prior[0],
            posterior_a: graph_filter.posterior[0],
            counts_posterior_a: count_filter.posterior[0],
        });
        source = target;
    }
    Ok(HiddenResult {
        steps,
        training_weights,
    })
}

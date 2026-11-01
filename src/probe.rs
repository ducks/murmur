//! Compare the online readout with a separately fitted linear centroid probe.
use crate::{
    data::{trials, Rng},
    experiment::Config,
    graph::Graph,
    learning::Readout,
};

pub const NOISE_LEVELS: [f64; 4] = [0.0, 0.0001, 0.001, 0.01];

/// A linear nearest-centroid classifier, fitted to labeled examples.
/// Equal-distance ties select class 0. Features are not standardized.
pub struct CentroidProbe {
    pub centers: [Vec<f64>; 2],
}
impl CentroidProbe {
    pub fn fit(examples: &[(Vec<f64>, usize)]) -> Result<Self, &'static str> {
        let n = examples.first().ok_or("empty training set")?.0.len();
        if n == 0 {
            return Err("empty state");
        }
        let mut centers = [vec![0.0; n], vec![0.0; n]];
        let mut counts = [0usize; 2];
        for (x, label) in examples {
            if *label > 1 || x.len() != n || x.iter().any(|x| !x.is_finite()) {
                return Err("invalid example");
            }
            counts[*label] += 1;
            for (sum, value) in centers[*label].iter_mut().zip(x) {
                *sum += value;
            }
        }
        if counts.contains(&0) {
            return Err("both cues required");
        }
        for k in 0..2 {
            for x in &mut centers[k] {
                *x /= counts[k] as f64;
            }
        }
        Ok(Self { centers })
    }
    pub fn predict(&self, x: &[f64]) -> usize {
        assert_eq!(x.len(), self.centers[0].len());
        let distance = |k: usize| {
            x.iter()
                .zip(&self.centers[k])
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
        };
        usize::from(distance(1) < distance(0))
    }
}
fn present(graph: &mut Graph, cue: usize, delay: usize) {
    graph.reset();
    graph.step(cue);
    for _ in 0..delay {
        graph.step(2);
    }
}
#[derive(Debug)]
pub struct ProbeResult {
    pub delay: usize,
    pub noise: f64,
    pub online_correct: usize,
    pub probe_correct: usize,
    pub trials: usize,
    pub state_distance: f64,
}

/// The online reader is trained first on clean training trials. The probe is
/// fitted separately on clean states from its own training stream. Evaluation
/// uses a third cue stream, and the same noisy observation for both readers.
pub fn run_probe(c: &Config, noise_levels: &[f64]) -> Result<Vec<ProbeResult>, &'static str> {
    if c.units == 0
        || c.degree == 0
        || c.delay == 0
        || c.train_trials == 0
        || c.test_trials == 0
        || !c.learning_rate.is_finite()
        || c.learning_rate <= 0.0
        || noise_levels.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("invalid experiment configuration");
    }
    let mut graph = Graph::new(c.units, c.degree, c.seed);
    let mut online = Readout::new(c.units, c.learning_rate);
    for cue in trials(c.seed.wrapping_add(1), c.train_trials) {
        present(&mut graph, cue, c.delay);
        online.learn(&graph.state, cue);
    }
    let mut examples = Vec::new();
    for cue in trials(c.seed.wrapping_add(3), c.train_trials) {
        present(&mut graph, cue, c.delay);
        examples.push((graph.state.clone(), cue));
    }
    let probe = CentroidProbe::fit(&examples)?;
    present(&mut graph, 0, c.delay);
    let state0 = graph.state.clone();
    present(&mut graph, 1, c.delay);
    let distance = state0
        .iter()
        .zip(&graph.state)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let cues = trials(c.seed.wrapping_add(2), c.test_trials);
    let mut results = Vec::new();
    for &noise in noise_levels {
        // Same underlying random perturbations at each noise level.
        let mut rng = Rng(c.seed.wrapping_add(4));
        let mut result = ProbeResult {
            delay: c.delay,
            noise,
            online_correct: 0,
            probe_correct: 0,
            trials: c.test_trials,
            state_distance: distance,
        };
        for &cue in &cues {
            present(&mut graph, cue, c.delay);
            let observed: Vec<_> = graph
                .state
                .iter()
                .map(|s| s + noise * (rng.uniform() * 2.0 - 1.0))
                .collect();
            result.online_correct += usize::from(online.predict(&observed) == cue);
            result.probe_correct += usize::from(probe.predict(&observed) == cue);
        }
        results.push(result);
    }
    Ok(results)
}

//! Varied distractor sequences and disturbances inside the dynamical system.
use crate::{data::Rng, experiment::Config, graph::Graph, learning::Readout, probe::CentroidProbe};

#[derive(Clone, Copy, Debug)]
pub struct Task {
    pub varied: bool,
    pub noise: f64,
}
#[derive(Clone, Debug)]
pub struct Trial {
    pub cue: usize,
    pub distractors: Vec<usize>,
}

/// Hash assigns a whole distractor sequence to exactly one split regardless of cue.
/// This is a deterministic partition, not a cryptographic hash.
pub fn test_sequence(seq: &[usize]) -> bool {
    let mut hash = 14695981039346656037u64;
    for &symbol in seq {
        hash = (hash ^ symbol as u64).wrapping_mul(1099511628211);
    }
    hash % 5 == 0
}
pub fn generate(seed: u64, count: usize, delay: usize, varied: bool, test: bool) -> Vec<Trial> {
    assert!(delay >= 4);
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let cue = (rng.next_u64() & 1) as usize;
        let distractors: Vec<_> = (0..delay)
            .map(|_| {
                if varied {
                    2 + (rng.next_u64() % 3) as usize
                } else {
                    2
                }
            })
            .collect();
        if !varied || test_sequence(&distractors) == test {
            out.push(Trial { cue, distractors });
        }
    }
    out
}
fn present(graph: &mut Graph, trial: &Trial, noise: f64, rng: &mut Rng) {
    graph.reset();
    for symbol in std::iter::once(trial.cue).chain(trial.distractors.iter().copied()) {
        graph.step(symbol);
        // Fresh internal disturbance after each transition, including cue step.
        for x in &mut graph.state {
            *x = (*x + noise * (2.0 * rng.uniform() - 1.0)).clamp(-1.0, 1.0);
        }
    }
}
#[derive(Debug)]
pub struct RobustnessResult {
    pub online_correct: usize,
    pub probe_correct: usize,
    pub test_trials: usize,
}
/// Separate training and test streams; within a run both readers see identical
/// states. Internal disturbances are sampled independently for each trial.
pub fn run(c: &Config, task: Task) -> Result<RobustnessResult, &'static str> {
    if c.units == 0
        || c.degree == 0
        || c.delay < 4
        || c.train_trials == 0
        || c.test_trials == 0
        || !c.learning_rate.is_finite()
        || c.learning_rate <= 0.0
        || !task.noise.is_finite()
        || task.noise < 0.0
    {
        return Err("invalid configuration");
    }
    let training = generate(
        c.seed.wrapping_add(101),
        c.train_trials,
        c.delay,
        task.varied,
        false,
    );
    let testing = generate(
        c.seed.wrapping_add(102),
        c.test_trials,
        c.delay,
        task.varied,
        true,
    );
    let mut train_noise = Rng(c.seed.wrapping_add(103));
    let mut test_noise = Rng(c.seed.wrapping_add(104));
    let mut graph = Graph::with_symbols(c.units, c.degree, c.seed, 5);
    let mut readout = Readout::new(c.units, c.learning_rate);
    let mut examples = Vec::with_capacity(c.train_trials);
    for trial in &training {
        present(&mut graph, trial, task.noise, &mut train_noise);
        readout.learn(&graph.state, trial.cue);
        examples.push((graph.state.clone(), trial.cue));
    }
    let probe = CentroidProbe::fit(&examples)?;
    let mut result = RobustnessResult {
        online_correct: 0,
        probe_correct: 0,
        test_trials: c.test_trials,
    };
    for trial in &testing {
        present(&mut graph, trial, task.noise, &mut test_noise);
        result.online_correct += usize::from(readout.predict(&graph.state) == trial.cue);
        result.probe_correct += usize::from(probe.predict(&graph.state) == trial.cue);
    }
    Ok(result)
}

use crate::{data::trials, graph::Graph, learning::Readout};
#[derive(Clone, Debug)]
pub struct Config {
    pub seed: u64,
    pub units: usize,
    pub degree: usize,
    pub train_trials: usize,
    pub test_trials: usize,
    pub learning_rate: f64,
    pub delay: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            seed: 42,
            units: 128,
            degree: 4,
            train_trials: 1000,
            test_trials: 1000,
            learning_rate: 0.05,
            delay: 1,
        }
    }
}
#[derive(Debug)]
pub struct Metrics {
    pub graph_correct: usize,
    pub reset_correct: usize,
    pub edge_free_correct: usize,
    pub markov_correct: usize,
    pub test_trials: usize,
}
/// Each independent trial resets state before its cue. The reset ablation also
/// resets between cue and marker; both readouts are trained on their own states.
pub fn run(c: &Config) -> (Metrics, Graph, Vec<[f64; 3]>) {
    assert!(
        c.units > 0
            && c.degree > 0
            && c.train_trials > 0
            && c.test_trials > 0
            && c.delay > 0
            && c.learning_rate.is_finite()
            && c.learning_rate > 0.0
    );
    let mut graph = Graph::new(c.units, c.degree, c.seed);
    let mut reset = graph.clone();
    let mut edge_free = graph.clone();
    edge_free.edges.clear();
    let mut model = Readout::new(c.units, c.learning_rate);
    let mut ablation = Readout::new(c.units, c.learning_rate);
    let mut edge_free_model = Readout::new(c.units, c.learning_rate);
    let mut counts = [0usize; 2];
    for cue in trials(c.seed.wrapping_add(1), c.train_trials) {
        graph.reset();
        graph.step(cue);
        for _ in 0..c.delay {
            graph.step(2);
        }
        model.learn(&graph.state, cue);
        edge_free.reset();
        edge_free.step(cue);
        for _ in 0..c.delay {
            edge_free.step(2);
        }
        edge_free_model.learn(&edge_free.state, cue);
        reset.reset();
        reset.step(cue);
        reset.reset();
        for _ in 0..c.delay {
            reset.step(2);
        }
        ablation.learn(&reset.state, cue);
        counts[cue] += 1;
    }
    let markov = usize::from(counts[1] > counts[0]);
    let mut m = Metrics {
        graph_correct: 0,
        reset_correct: 0,
        edge_free_correct: 0,
        markov_correct: 0,
        test_trials: c.test_trials,
    };
    let mut activations = Vec::new();
    for (i, cue) in trials(c.seed.wrapping_add(2), c.test_trials)
        .into_iter()
        .enumerate()
    {
        graph.reset();
        graph.step(cue);
        for _ in 0..c.delay {
            graph.step(2);
        }
        edge_free.reset();
        edge_free.step(cue);
        for _ in 0..c.delay {
            edge_free.step(2);
        }
        reset.reset();
        for _ in 0..c.delay {
            reset.step(2);
        }
        m.graph_correct += usize::from(model.predict(&graph.state) == cue);
        m.reset_correct += usize::from(ablation.predict(&reset.state) == cue);
        m.edge_free_correct += usize::from(edge_free_model.predict(&edge_free.state) == cue);
        m.markov_correct += usize::from(markov == cue);
        if i < 16 {
            activations.push([i as f64, cue as f64, graph.state[0]]);
        }
    }
    (m, graph, activations)
}

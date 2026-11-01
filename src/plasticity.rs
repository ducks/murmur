//! A deliberately simple, experimental three-factor edge rule.
use crate::{data::trials, experiment::Config, graph::Graph, learning::Readout};

pub const EDGE_RATE: f64 = 0.01;
pub const TRACE_DECAY: f64 = 0.8;
pub const WEIGHT_LIMIT: f64 = 0.5;

/// Traces are trial-local. Each step records source activity before propagation
/// times destination activity after propagation, with exponential decay.
pub fn trace_step(graph: &mut Graph, trace: &mut [f64], symbol: usize) {
    assert_eq!(trace.len(), graph.edges.len());
    let previous = graph.state.clone();
    graph.step(symbol);
    for (e, t) in graph.edges.iter().zip(trace) {
        *t = TRACE_DECAY * *t + previous[e.from] * graph.state[e.to];
    }
}

/// Feedback is a signed output error. Each destination receives that scalar
/// multiplied by its current class-1 readout weight. This is supervised global
/// feedback, not strictly local credit assignment or a true gradient.
pub fn update_edges(graph: &mut Graph, trace: &[f64], readout: &Readout, feedback: f64) {
    assert_eq!(graph.edges.len(), trace.len());
    for (edge, eligibility) in graph.edges.iter_mut().zip(trace) {
        let change = EDGE_RATE * feedback * readout.weights[1][edge.to] * eligibility;
        edge.weight = (edge.weight + change).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
    }
}

fn output_error(readout: &Readout, state: &[f64], target: usize) -> f64 {
    let score = readout.scores(state)[1].clamp(-40.0, 40.0);
    target as f64 - 1.0 / (1.0 + (-score).exp())
}

struct Model {
    graph: Graph,
    readout: Readout,
}
impl Model {
    fn new(graph: &Graph, rate: f64) -> Self {
        Self {
            graph: graph.clone(),
            readout: Readout::new(graph.state.len(), rate),
        }
    }
    fn present(&mut self, cue: usize, delay: usize) -> Vec<f64> {
        self.graph.reset();
        let mut trace = vec![0.0; self.graph.edges.len()];
        trace_step(&mut self.graph, &mut trace, cue);
        for _ in 0..delay {
            trace_step(&mut self.graph, &mut trace, 2);
        }
        trace
    }
}

#[derive(Debug)]
pub struct PlasticityMetrics {
    pub frozen_correct: usize,
    pub learned_correct: usize,
    pub shuffled_correct: usize,
    pub test_trials: usize,
    pub learned_mean_abs_change: f64,
    pub shuffled_mean_abs_change: f64,
    pub diagnostics: [Diagnostics; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct Diagnostics {
    pub cue_state_distance: f64,
    pub cue0_margin: f64,
    pub cue1_margin: f64,
}

fn diagnose(model: &mut Model, delay: usize) -> Diagnostics {
    model.present(0, delay);
    let state0 = model.graph.state.clone();
    let scores0 = model.readout.scores(&state0);
    model.present(1, delay);
    let scores1 = model.readout.scores(&model.graph.state);
    Diagnostics {
        cue_state_distance: state0
            .iter()
            .zip(&model.graph.state)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt(),
        cue0_margin: scores0[1] - scores0[0],
        cue1_margin: scores1[1] - scores1[0],
    }
}
/// Three independent models share initial topology, data, and readout settings.
/// The control uses its own previous trial's error as mismatched feedback; its
/// first trial gets zero. Both feedback conditions use pre-readout-update error.
pub fn run_plasticity(c: &Config) -> (PlasticityMetrics, Graph, Graph, Vec<(usize, f64, f64)>) {
    assert!(
        c.units > 0
            && c.degree > 0
            && c.delay > 0
            && c.train_trials > 0
            && c.test_trials > 0
            && c.learning_rate.is_finite()
            && c.learning_rate > 0.0
    );
    let initial = Graph::new(c.units, c.degree, c.seed);
    let mut frozen = Model::new(&initial, c.learning_rate);
    let mut learned = Model::new(&initial, c.learning_rate);
    let mut shuffled = Model::new(&initial, c.learning_rate);
    let mut previous_error = 0.0;
    let mut log = Vec::new();
    for (i, cue) in trials(c.seed.wrapping_add(1), c.train_trials)
        .into_iter()
        .enumerate()
    {
        frozen.present(cue, c.delay);
        frozen.readout.learn(&frozen.graph.state, cue);
        let trace = learned.present(cue, c.delay);
        let error = output_error(&learned.readout, &learned.graph.state, cue);
        update_edges(&mut learned.graph, &trace, &learned.readout, error);
        learned.readout.learn(&learned.graph.state, cue);
        let trace = shuffled.present(cue, c.delay);
        let error = output_error(&shuffled.readout, &shuffled.graph.state, cue);
        update_edges(
            &mut shuffled.graph,
            &trace,
            &shuffled.readout,
            previous_error,
        );
        shuffled.readout.learn(&shuffled.graph.state, cue);
        previous_error = error;
        if (i + 1) % 100 == 0 || i + 1 == c.train_trials {
            log.push((
                i + 1,
                mean_change(&initial, &learned.graph),
                mean_change(&initial, &shuffled.graph),
            ));
        }
    }
    let mut m = PlasticityMetrics {
        frozen_correct: 0,
        learned_correct: 0,
        shuffled_correct: 0,
        test_trials: c.test_trials,
        learned_mean_abs_change: mean_change(&initial, &learned.graph),
        shuffled_mean_abs_change: mean_change(&initial, &shuffled.graph),
        diagnostics: [Diagnostics {
            cue_state_distance: 0.0,
            cue0_margin: 0.0,
            cue1_margin: 0.0,
        }; 3],
    };
    for cue in trials(c.seed.wrapping_add(2), c.test_trials) {
        frozen.present(cue, c.delay);
        learned.present(cue, c.delay);
        shuffled.present(cue, c.delay);
        m.frozen_correct += usize::from(frozen.readout.predict(&frozen.graph.state) == cue);
        m.learned_correct += usize::from(learned.readout.predict(&learned.graph.state) == cue);
        m.shuffled_correct += usize::from(shuffled.readout.predict(&shuffled.graph.state) == cue);
    }
    m.diagnostics = [
        diagnose(&mut frozen, c.delay),
        diagnose(&mut learned, c.delay),
        diagnose(&mut shuffled, c.delay),
    ];
    (m, learned.graph, shuffled.graph, log)
}
fn mean_change(a: &Graph, b: &Graph) -> f64 {
    a.edges
        .iter()
        .zip(&b.edges)
        .map(|(x, y)| (x.weight - y.weight).abs())
        .sum::<f64>()
        / a.edges.len() as f64
}

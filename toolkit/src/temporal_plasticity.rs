//! Online, error-modulated recurrent edge learning on a delayed-bit stream.
//! Eligibility plus fixed random feedback is an approximate heuristic, not BPTT.
use crate::temporal_stream::{self, StreamConfig};
use murmur::{data::Rng, graph::Graph, learning::Readout};
use std::{collections::HashSet, fs, io, path::Path};
pub const TRACE_DECAY: f64 = 0.8;
pub const EDGE_RATE: f64 = 0.02;
pub const WEIGHT_LIMIT: f64 = 0.5;

pub struct PlasticModel {
    pub graph: Graph,
    pub readout_weights_marker: usize,
    readout: Readout,
    pub eligibility: Vec<f64>,
    pub feedback: Vec<f64>,
}
impl PlasticModel {
    pub fn new(c: &StreamConfig, seed: u64) -> Self {
        let graph = Graph::new(c.units, c.degree, seed);
        let mut rng = Rng(seed.wrapping_add(33000));
        let scale = 1.0 / (c.units as f64).sqrt();
        let feedback = (0..c.units)
            .map(|_| {
                if rng.next_u64() & 1 == 0 {
                    -scale
                } else {
                    scale
                }
            })
            .collect();
        let eligibility = vec![0.0; graph.edges.len()];
        Self {
            graph,
            readout_weights_marker: 2 * (c.units + 1),
            readout: Readout::new(c.units, c.learning_rate),
            eligibility,
            feedback,
        }
    }
    /// Compute the synchronous transition and, during training only, update
    /// each edge's local eligibility from previous source activity and the
    /// destination's local tanh derivative.
    pub fn advance(&mut self, input: usize, record_trace: bool) -> usize {
        let old = if record_trace {
            Some(self.graph.state.clone())
        } else {
            None
        };
        self.graph.step(input);
        if let Some(old) = old {
            let leak = self.graph.leak;
            for (edge, trace) in self.graph.edges.iter().zip(&mut self.eligibility) {
                let activated = ((self.graph.state[edge.to] - (1.0 - leak) * old[edge.to]) / leak)
                    .clamp(-1.0, 1.0);
                *trace =
                    TRACE_DECAY * *trace + leak * (1.0 - activated * activated) * old[edge.from];
            }
        }
        self.readout.predict(&self.graph.state)
    }
    /// The reader always learns from the true target. Only the recurrent-edge
    /// feedback target differs in the control; it may coincide by chance.
    pub fn learn(&mut self, true_target: usize, feedback_target: usize) {
        assert!(true_target < 2 && feedback_target < 2);
        let score = self.readout.scores(&self.graph.state)[1].clamp(-40.0, 40.0);
        let probability = 1.0 / (1.0 + (-score).exp());
        let error = feedback_target as f64 - probability;
        for (edge, trace) in self.graph.edges.iter_mut().zip(&self.eligibility) {
            edge.weight = (edge.weight + EDGE_RATE * error * self.feedback[edge.to] * trace)
                .clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
        }
        self.readout.learn(&self.graph.state, true_target);
    }
    /// Control intervention: normalize the proposed edge-update vector to a
    /// supplied L2 norm. Clipping may make the realized norm smaller. This
    /// cross-model matching is experimental instrumentation, not a local rule.
    pub fn learn_with_update_norm(
        &mut self,
        true_target: usize,
        feedback_target: usize,
        desired_norm: f64,
    ) -> f64 {
        assert!(
            true_target < 2
                && feedback_target < 2
                && desired_norm.is_finite()
                && desired_norm >= 0.0
        );
        let score = self.readout.scores(&self.graph.state)[1].clamp(-40.0, 40.0);
        let error = feedback_target as f64 - 1.0 / (1.0 + (-score).exp());
        let proposal: Vec<f64> = self
            .graph
            .edges
            .iter()
            .zip(&self.eligibility)
            .map(|(edge, trace)| error * self.feedback[edge.to] * trace)
            .collect();
        let norm = proposal.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = if norm > 0.0 { desired_norm / norm } else { 0.0 };
        let mut actual_squared = 0.0;
        for (edge, value) in self.graph.edges.iter_mut().zip(proposal) {
            let old = edge.weight;
            edge.weight = (old + scale * value).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
            actual_squared += (edge.weight - old).powi(2);
        }
        self.readout.learn(&self.graph.state, true_target);
        actual_squared.sqrt()
    }
    pub fn reset_state(&mut self) {
        self.graph.reset();
        self.eligibility.fill(0.0);
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub baseline: temporal_stream::Row,
    pub adaptive: usize,
    pub random_feedback: usize,
}
#[derive(Debug)]
pub struct Checkpoint {
    pub step: usize,
    pub scored: usize,
    pub adaptive_online_correct: usize,
    pub random_online_correct: usize,
    pub adaptive_mean_abs_change: f64,
    pub random_mean_abs_change: f64,
    pub adaptive_saturated: usize,
    pub random_saturated: usize,
}
pub struct Run {
    pub rows: Vec<Row>,
    pub initial: Graph,
    pub adaptive: Graph,
    pub random_feedback: Graph,
    pub feedback: Vec<f64>,
    pub checkpoints: Vec<Checkpoint>,
    pub readout_parameters: usize,
}
fn drift(graph: &Graph, initial: &Graph) -> (f64, usize) {
    let mean = graph
        .edges
        .iter()
        .zip(&initial.edges)
        .map(|(a, b)| (a.weight - b.weight).abs())
        .sum::<f64>()
        / graph.edges.len() as f64;
    let clipped = graph
        .edges
        .iter()
        .filter(|e| e.weight.abs() >= WEIGHT_LIMIT - 1e-12)
        .count();
    (mean, clipped)
}
/// True and control learners share input, initial graph, feedback vector, trace
/// rule and reader targets. The control's graph receives a separately seeded
/// random target instead of the actual one; the reader still sees true labels.
pub fn run_delay(c: &StreamConfig, seed: u64, delay: usize) -> Run {
    let fixed = temporal_stream::run_delay(c, seed, delay);
    let train = temporal_stream::stream(seed.wrapping_add(31000), c.train_tokens);
    let fake = temporal_stream::stream(seed.wrapping_add(34000), c.train_tokens);
    let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
    let mut adaptive = PlasticModel::new(c, seed);
    let mut control = PlasticModel::new(c, seed);
    let initial = adaptive.graph.clone();
    let mut checkpoints = Vec::new();
    let (mut scored, mut ac, mut rc) = (0usize, 0usize, 0usize);
    for (t, &input) in train.iter().enumerate() {
        let a = adaptive.advance(input, true);
        let b = control.advance(input, true);
        if t >= delay {
            let target = train[t - delay];
            scored += 1;
            ac += usize::from(a == target);
            rc += usize::from(b == target);
            adaptive.learn(target, target);
            control.learn(target, fake[t]);
        }
        if (t + 1) % 1000 == 0 || t + 1 == train.len() {
            let (ad, asat) = drift(&adaptive.graph, &initial);
            let (rd, rsat) = drift(&control.graph, &initial);
            checkpoints.push(Checkpoint {
                step: t + 1,
                scored,
                adaptive_online_correct: ac,
                random_online_correct: rc,
                adaptive_mean_abs_change: ad,
                random_mean_abs_change: rd,
                adaptive_saturated: asat,
                random_saturated: rsat,
            });
        }
    }
    adaptive.reset_state();
    control.reset_state();
    let mut rows = Vec::with_capacity(fixed.rows.len());
    for (t, &input) in test.iter().enumerate() {
        let a = adaptive.advance(input, false);
        let b = control.advance(input, false);
        if t >= delay {
            let source = fixed.rows[t - delay].clone();
            debug_assert_eq!(source.index, t);
            debug_assert_eq!(source.target, test[t - delay]);
            rows.push(Row {
                baseline: source,
                adaptive: a,
                random_feedback: b,
            });
        }
    }
    Run {
        rows,
        initial,
        adaptive: adaptive.graph,
        random_feedback: control.graph,
        feedback: adaptive.feedback,
        checkpoints,
        readout_parameters: adaptive.readout_weights_marker,
    }
}
fn validate(c: &StreamConfig) -> Result<(), String> {
    StreamConfig::parse(&c.resolved())?;
    let mut seeds = HashSet::new();
    for &s in &c.seeds {
        for offset in [0, 31000, 32000, 33000, 34000] {
            if !seeds.insert(s.wrapping_add(offset)) {
                return Err("graph, stream or feedback seeds overlap across runs".into());
            }
        }
    }
    Ok(())
}
fn outputs(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),format!("runner_version={}\ntask=predict x[t-delay] after observing x[t]\nlocal_trace=0.8*trace+leak*(1-tanh_field^2)*previous_source\nedge_update=0.02*feedback_error*fixed_random_destination_sign*trace\nweight_limit=0.5\nfeedback_sign_scale=1/sqrt(units)\ntrue_reader_target=actual delayed bit for both learners\ncontrol_edge_target=separately seeded random bit; can match actual by chance\nseed_scheme=graph:s;train:s+31000;test:s+32000;feedback:s+33000;random_targets:s+34000 (wrapping u64)\nstate=reset at train/test boundary; graph and readout weights frozen in evaluation\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n",env!("CARGO_PKG_VERSION")))?;
    let mut pred=String::from("seed,delay,index,input,target,fixed_graph,adaptive_graph,random_feedback,edge_free,reset,shift\n");
    let mut metrics=String::from("seed,delay,model,correct,trials,accuracy,graph_edge_count,trainable_edge_count,trainable_readout_parameters\n");
    let mut diagnostics=String::from("seed,delay,step,scored,adaptive_online_correct,random_online_correct,adaptive_mean_abs_change,random_mean_abs_change,adaptive_saturated,random_saturated\n");
    let mut weights = String::from("seed,delay,edge_id,from,to,initial,adaptive,random_feedback\n");
    let mut feedback = String::from("seed,delay,unit,fixed_feedback\n");
    let names = [
        "fixed_graph",
        "adaptive_graph",
        "random_feedback",
        "edge_free",
        "reset",
        "shift",
    ];
    let mut totals = vec![[0usize; 6]; c.delays.len()];
    let mut counts = vec![0usize; c.delays.len()];
    for &seed in &c.seeds {
        for (didx, &delay) in c.delays.iter().enumerate() {
            let r = run_delay(c, seed, delay);
            let n = r.rows.len();
            let mut correct = [0usize; 6];
            for x in &r.rows {
                let p = &x.baseline;
                for (k, v) in [
                    p.graph,
                    x.adaptive,
                    x.random_feedback,
                    p.edge_free,
                    p.reset,
                    p.shift,
                ]
                .iter()
                .enumerate()
                {
                    correct[k] += usize::from(*v == p.target);
                }
                pred.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{},{}\n",
                    p.index,
                    p.input,
                    p.target,
                    p.graph,
                    x.adaptive,
                    x.random_feedback,
                    p.edge_free,
                    p.reset,
                    p.shift
                ));
            }
            counts[didx] += n;
            for k in 0..6 {
                totals[didx][k] += correct[k];
                let fixed_edges = if k == 3 || k == 5 {
                    0
                } else {
                    r.initial.edges.len()
                };
                let learn_edges = if k == 1 || k == 2 {
                    r.initial.edges.len()
                } else {
                    0
                };
                let params = if k < 5 { r.readout_parameters } else { 0 };
                metrics.push_str(&format!(
                    "{seed},{delay},{},{},{n},{},{fixed_edges},{learn_edges},{params}\n",
                    names[k],
                    correct[k],
                    correct[k] as f64 / n as f64
                ));
            }
            for d in &r.checkpoints {
                diagnostics.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{}\n",
                    d.step,
                    d.scored,
                    d.adaptive_online_correct,
                    d.random_online_correct,
                    d.adaptive_mean_abs_change,
                    d.random_mean_abs_change,
                    d.adaptive_saturated,
                    d.random_saturated
                ));
            }
            for (i, ((initial, adaptive), fake)) in r
                .initial
                .edges
                .iter()
                .zip(&r.adaptive.edges)
                .zip(&r.random_feedback.edges)
                .enumerate()
            {
                weights.push_str(&format!(
                    "{seed},{delay},{i},{},{},{:.17},{:.17},{:.17}\n",
                    initial.from, initial.to, initial.weight, adaptive.weight, fake.weight
                ));
            }
            for (i, w) in r.feedback.iter().enumerate() {
                feedback.push_str(&format!("{seed},{delay},{i},{w:.17}\n"));
            }
        }
    }
    fs::write(dir.join("predictions.csv"), pred)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("training_diagnostics.csv"), diagnostics)?;
    fs::write(dir.join("weights.csv"), weights)?;
    fs::write(dir.join("feedback.csv"), feedback)?;
    let mut summary=format!("# Temporal graph plasticity\n\n{} seeds; true-target and random-target feedback learners use the same streams and fixed feedback signs. Evaluation resets state and freezes all weights.\n\n| Delay | Fixed | Adaptive | Random feedback | Edge-free | Reset | Shift register |\n|---:|---:|---:|---:|---:|---:|---:|\n",c.seeds.len());
    for (i, d) in c.delays.iter().enumerate() {
        let n = counts[i] as f64;
        summary.push_str(&format!(
            "| {d} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            totals[i][0] as f64 / n,
            totals[i][1] as f64 / n,
            totals[i][2] as f64 / n,
            totals[i][3] as f64 / n,
            totals[i][4] as f64 / n,
            totals[i][5] as f64 / n
        ));
    }
    summary.push_str("\nRandom feedback is an independent pseudo-target, not an exact error permutation. Both readers learn true labels; only graph feedback differs. The shift register stores history and is not resource-matched.\n");
    fs::write(dir.join("summary.md"), summary)?;
    Ok(())
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    validate(c).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    if let Some(p) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(p)?;
    }
    fs::create_dir(dir)?;
    fs::write(dir.join("status.txt"), "running\n")?;
    match outputs(c, dir) {
        Ok(()) => fs::write(dir.join("status.txt"), "complete\n"),
        Err(e) => {
            let _ = fs::write(dir.join("status.txt"), "failed\n");
            Err(e)
        }
    }
}

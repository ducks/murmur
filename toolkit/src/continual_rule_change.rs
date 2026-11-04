//! A -> B -> A online transition learning and frozen counterfactual probes.
use crate::{temporal_plasticity::PlasticModel, temporal_stream::StreamConfig};
use murmur::data::Rng;
use std::{collections::HashSet, fs, io, path::Path};
pub const NOISE: f64 = 0.1;
pub const WINDOW: usize = 200;

/// Rule A normally repeats the current bit; B normally flips it. Noise and
/// initial state come from the supplied deterministic RNG seed.
pub fn sequence(seed: u64, transitions: usize, rule: usize) -> Vec<usize> {
    assert!(rule < 2);
    let mut rng = Rng(seed);
    let mut x = (rng.next_u64() & 1) as usize;
    let mut out = vec![x];
    for _ in 0..transitions {
        let flip = (rule == 1) ^ (rng.uniform() < NOISE);
        if flip {
            x ^= 1;
        }
        out.push(x);
    }
    out
}
#[derive(Clone)]
pub struct Markov {
    pub p: [[f64; 2]; 2],
    pub rate: f64,
    pub oracle: bool,
}
impl Markov {
    pub fn new(rate: f64, oracle: bool) -> Self {
        Self {
            p: [[0.5; 2]; 2],
            rate,
            oracle,
        }
    }
    pub fn predict(&self, x: usize, rule: usize) -> usize {
        usize::from(self.p[if self.oracle { rule } else { 0 }][x] > 0.5)
    }
    pub fn learn(&mut self, x: usize, y: usize, rule: usize) {
        let p = &mut self.p[if self.oracle { rule } else { 0 }][x];
        *p += self.rate * (y as f64 - *p);
    }
}
#[derive(Clone)]
pub struct Models {
    pub fixed: PlasticModel,
    pub adaptive: PlasticModel,
    pub markov: Markov,
    pub oracle: Markov,
}
impl Models {
    pub fn new(c: &StreamConfig, seed: u64) -> Self {
        Self {
            fixed: PlasticModel::new(c, seed),
            adaptive: PlasticModel::new(c, seed),
            markov: Markov::new(c.learning_rate, false),
            oracle: Markov::new(c.learning_rate, true),
        }
    }
    pub fn reset(&mut self) {
        self.fixed.reset_state();
        self.adaptive.reset_state();
    }
    pub fn predict(&mut self, x: usize, rule: usize, train: bool) -> [usize; 4] {
        [
            self.fixed.advance(x, false),
            self.adaptive.advance(x, train),
            self.markov.predict(x, rule),
            self.oracle.predict(x, rule),
        ]
    }
    pub fn learn(&mut self, x: usize, y: usize, rule: usize) {
        let signal = 0.5 - self.adaptive.class_one_probability();
        self.fixed.learn_with_error(y, 0.0);
        self.adaptive.learn_with_error(y, signal);
        self.markov.learn(x, y, rule);
        self.oracle.learn(x, y, rule);
    }
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub phase: usize,
    pub step: usize,
    pub input: usize,
    pub target: usize,
    pub predicted: [usize; 4],
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probe {
    pub checkpoint: usize,
    pub rule: usize,
    pub correct: [usize; 4],
    pub trials: usize,
}
pub struct Run {
    pub rows: Vec<Row>,
    pub probes: Vec<Probe>,
    pub training_probes: Vec<Probe>,
    pub weights: Vec<Vec<(f64, f64)>>,
}
/// Probing clones every reader and graph, resets its state, and never learns.
pub fn probe(m: &Models, input: &[usize], rule: usize, checkpoint: usize) -> Probe {
    let mut frozen = m.clone();
    frozen.reset();
    let mut correct = [0usize; 4];
    for i in 0..input.len() - 1 {
        let y = input[i + 1];
        let p = frozen.predict(input[i], rule, false);
        for k in 0..4 {
            correct[k] += usize::from(p[k] == y);
        }
    }
    Probe {
        checkpoint,
        rule,
        correct,
        trials: input.len() - 1,
    }
}
pub fn run(c: &StreamConfig, seed: u64) -> Run {
    let sequences = [
        sequence(seed.wrapping_add(40000), c.train_tokens, 0),
        sequence(seed.wrapping_add(42000), c.train_tokens, 1),
        sequence(seed.wrapping_add(44000), c.train_tokens, 0),
    ];
    let probes = [
        sequence(seed.wrapping_add(46000), c.test_tokens, 0),
        sequence(seed.wrapping_add(48000), c.test_tokens, 1),
    ];
    let mut models = Models::new(c, seed);
    let mut rows = Vec::new();
    let mut results = Vec::new();
    let mut training_probes = Vec::new();
    let mut weights = Vec::new();
    for phase in 0..3 {
        let rule = usize::from(phase == 1);
        models.reset();
        for t in 0..c.train_tokens {
            let x = sequences[phase][t];
            let y = sequences[phase][t + 1];
            let predicted = models.predict(x, rule, true);
            rows.push(Row {
                phase,
                step: t,
                input: x,
                target: y,
                predicted,
            });
            models.learn(x, y, rule);
        }
        for rule in 0..2 {
            results.push(probe(&models, &probes[rule], rule, phase));
        }
        let tail_len = c.test_tokens.min(c.train_tokens);
        let tail = &sequences[phase][c.train_tokens - tail_len..];
        training_probes.push(probe(&models, tail, rule, phase));
        weights.push(
            models
                .fixed
                .graph
                .edges
                .iter()
                .zip(&models.adaptive.graph.edges)
                .map(|(f, a)| (f.weight, a.weight))
                .collect(),
        );
    }
    Run {
        rows,
        probes: results,
        training_probes,
        weights,
    }
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let offsets = [0, 33000, 40000, 42000, 44000, 46000, 48000];
    let mut seeds = HashSet::new();
    for s in &c.seeds {
        for o in offsets {
            if !seeds.insert(s.wrapping_add(o)) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "seed roles overlap",
                ));
            }
        }
    }
    if let Some(p) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(p)?;
    }
    fs::create_dir(dir)?;
    fs::write(dir.join("status.txt"), "running\n")?;
    let result = write_outputs(c, dir);
    fs::write(
        dir.join("status.txt"),
        if result.is_ok() {
            "complete\n"
        } else {
            "failed\n"
        },
    )?;
    result
}
fn write_outputs(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),"task=online predict next binary symbol; A repeats and B flips with probability 0.9; A->B->A, three separately seeded phases\nmodels=fixed graph true-label reader, reader-feedback graph, recency Markov, oracle-context Markov\noracle=provided true phase/rule, separate Markov parameters; other models never receive rule identity\nlearning=prediction before true-label update; graph feedback uses 0.5-class1_probability; fixed graph gets zero edge update\nstate=graph activations and traces reset at phase boundaries; learned readers and edges persist\nprobes=after each phase, evaluate cloned frozen models on same independent A and B streams; model state reset for each probe; no learning\nseeds=graph:s feedback:s+33000 phaseA:s+40000 phaseB:s+42000 returnA:s+44000 probeA:s+46000 probeB:s+48000; wrapping u64\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let names = ["fixed_graph", "adaptive_graph", "markov", "oracle_markov"];
    let phase_names = ["A", "B", "A_return"];
    let mut pred = String::from(
        "seed,phase,step,input,target,fixed_graph,adaptive_graph,markov,oracle_markov\n",
    );
    let mut windows =
        String::from("seed,phase,window_start,window_end,model,correct,trials,accuracy\n");
    let mut probes = String::from("seed,checkpoint,probe_rule,model,correct,trials,accuracy\n");
    let mut training_probes = String::from("seed,checkpoint,model,correct,trials,accuracy\n");
    let mut weights = String::from("seed,checkpoint,edge_id,initial,fixed,adaptive\n");
    let mut sum = [[[0usize; 4]; 2]; 3];
    let mut count = [[0usize; 2]; 3];
    for &seed in &c.seeds {
        let r = run(c, seed);
        for row in &r.rows {
            pred.push_str(&format!(
                "{seed},{},{},{},{},{},{},{},{}\n",
                phase_names[row.phase],
                row.step,
                row.input,
                row.target,
                row.predicted[0],
                row.predicted[1],
                row.predicted[2],
                row.predicted[3]
            ));
        }
        for phase in 0..3 {
            for start in (0..c.train_tokens).step_by(WINDOW) {
                let end = (start + WINDOW).min(c.train_tokens);
                let slice = &r.rows[phase * c.train_tokens + start..phase * c.train_tokens + end];
                for (k, name) in names.iter().enumerate() {
                    let correct = slice
                        .iter()
                        .filter(|row| row.predicted[k] == row.target)
                        .count();
                    windows.push_str(&format!(
                        "{seed},{},{start},{end},{name},{correct},{},{}\n",
                        phase_names[phase],
                        slice.len(),
                        correct as f64 / slice.len() as f64
                    ));
                }
            }
        }
        for p in &r.probes {
            count[p.checkpoint][p.rule] += p.trials;
            for (k, name) in names.iter().enumerate() {
                sum[p.checkpoint][p.rule][k] += p.correct[k];
                probes.push_str(&format!(
                    "{seed},{},{},{name},{},{},{}\n",
                    phase_names[p.checkpoint],
                    phase_names[p.rule],
                    p.correct[k],
                    p.trials,
                    p.correct[k] as f64 / p.trials as f64
                ));
            }
        }
        for p in &r.training_probes {
            for (k, name) in names.iter().enumerate() {
                training_probes.push_str(&format!(
                    "{seed},{},{name},{},{},{}\n",
                    phase_names[p.checkpoint],
                    p.correct[k],
                    p.trials,
                    p.correct[k] as f64 / p.trials as f64
                ));
            }
        }
        let initial = PlasticModel::new(c, seed).graph;
        for (checkpoint, edges) in r.weights.iter().enumerate() {
            for (i, (fixed, adaptive)) in edges.iter().enumerate() {
                weights.push_str(&format!(
                    "{seed},{},{i},{:.17},{fixed:.17},{adaptive:.17}\n",
                    phase_names[checkpoint], initial.edges[i].weight
                ));
            }
        }
    }
    fs::write(dir.join("predictions.csv"), pred)?;
    fs::write(dir.join("windows.csv"), windows)?;
    fs::write(dir.join("probes.csv"), probes)?;
    fs::write(dir.join("training_tail_probes.csv"), training_probes)?;
    fs::write(dir.join("weights.csv"), weights)?;
    let mut summary=String::from("# Rule-change frozen probes\n\n| After phase | Probe rule | Fixed graph | Adaptive graph | Markov | Oracle-context Markov |\n|---|---|---:|---:|---:|---:|\n");
    for phase in 0..3 {
        for rule in 0..2 {
            let n = count[phase][rule] as f64;
            summary.push_str(&format!(
                "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                phase_names[phase],
                phase_names[rule],
                sum[phase][rule][0] as f64 / n,
                sum[phase][rule][1] as f64 / n,
                sum[phase][rule][2] as f64 / n,
                sum[phase][rule][3] as f64 / n
            ));
        }
    }
    fs::write(dir.join("summary.md"), summary)
}

//! Trusted evaluator for an agent-editable candidate on hidden-context streams.
use crate::{
    agent_candidate::Candidate,
    temporal_stream::StreamConfig,
    unlabeled_context::{self, Event, Models, WINDOW},
};
use murmur::data::Rng;
use std::{collections::HashSet, fs, io, path::Path};
const NAMES: [&str; 5] = ["single", "mixture", "candidate", "known_filter", "oracle"];
#[derive(Clone)]
pub struct Evaluator {
    pub controls: Models,
    pub candidate: Candidate,
}
impl Evaluator {
    pub fn new(seed: u64, rate: f64, noise: f64) -> Self {
        Self {
            controls: Models::new(seed, rate, noise),
            candidate: Candidate::new(seed.wrapping_add(51000), rate),
        }
    }
    pub fn step(&mut self, e: Event, learn: bool) -> [f64; 5] {
        let prediction = self.candidate.predict(e.input);
        assert!(
            prediction.is_finite() && (0.0..=1.0).contains(&prediction),
            "candidate returned invalid probability"
        );
        let (p, _, _) = self.controls.step(e, learn);
        self.candidate.observe(e.input, e.target, learn);
        [p[0], p[1], prediction, p[2], p[3]]
    }
    pub fn reset(&mut self) {
        self.controls.mixture.reset();
        self.controls.known.reset();
        self.candidate.reset();
    }
}
fn loss(p: f64, y: usize) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    if y == 1 {
        -p.ln()
    } else {
        -(1.0 - p).ln()
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub event: Event,
    pub predictions: [f64; 5],
}
#[derive(Clone, Debug, PartialEq)]
pub struct Probe {
    pub phase: usize,
    pub rule: usize,
    pub correct: [usize; 5],
    pub logloss: [f64; 5],
    pub trials: usize,
}
pub struct Run {
    pub rows: Vec<Row>,
    pub probes: Vec<Probe>,
}
fn probe(model: &Evaluator, data: &[Event], rule: usize, phase: usize) -> Probe {
    let mut frozen = model.clone();
    frozen.reset();
    let mut correct = [0; 5];
    let mut logloss = [0.0; 5];
    for &e in data {
        let p = frozen.step(e, false);
        for k in 0..5 {
            correct[k] += usize::from(usize::from(p[k] > 0.5) == e.target);
            logloss[k] += loss(p[k], e.target);
        }
    }
    Probe {
        phase,
        rule,
        correct,
        logloss,
        trials: data.len(),
    }
}
pub fn run(c: &StreamConfig, seed: u64, regime: usize, noise: f64) -> Run {
    let data = unlabeled_context::stream(seed.wrapping_add(40000), c.train_tokens, regime, noise);
    let aa = unlabeled_context::stream(seed.wrapping_add(46000), c.test_tokens, 1, noise);
    let a = &aa[..c.test_tokens];
    let mut rng = Rng(seed.wrapping_add(48000));
    let mut x = (rng.next_u64() & 1) as usize;
    let mut b = Vec::new();
    for _ in 0..c.test_tokens {
        let y = if rng.uniform() < noise { x } else { x ^ 1 };
        b.push(Event {
            input: x,
            target: y,
            context: 1,
        });
        x = y;
    }
    let mut model = Evaluator::new(seed, c.learning_rate, noise);
    let mut rows = Vec::new();
    let mut probes = Vec::new();
    for (i, e) in data.into_iter().enumerate() {
        let predictions = model.step(e, true);
        rows.push(Row {
            event: e,
            predictions,
        });
        if (i + 1) % c.train_tokens == 0 {
            let phase = i / c.train_tokens;
            probes.push(probe(&model, a, 0, phase));
            probes.push(probe(&model, &b, 1, phase));
        }
    }
    Run { rows, probes }
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let mut used = HashSet::new();
    for s in &c.seeds {
        for o in [0, 40000, 46000, 48000, 51000] {
            if !used.insert(s.wrapping_add(o)) {
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
    fs::write(dir.join("protocol.txt"),"models=single Markov, original two-slot mixture, agent candidate, known-rule filter, labelled-context oracle\ncandidate receives only seed, rate, current input and observed target; prediction occurs before target; rule and noise not passed\nregimes=ABA, stationary, switching every 200; noise=0.1,0.25; shared paired streams\nprobes=cloned parameters, candidate reset and frozen learning, inference updates from observed targets\nseeds=stream:s+40000 probeA:s+46000 probeB:s+48000 init:s+51000 wrapping u64\nsource_revision=recorded by coordinator\nnixpkgs_revision=unrecorded\n")?;
    let regimes = ["aba", "stationary", "switching"];
    let mut predictions=String::from("seed,regime,noise,step,context,input,target,single_p1,mixture_p1,candidate_p1,known_p1,oracle_p1\n");
    let mut windows = String::from(
        "seed,regime,noise,window_start,window_end,model,correct,trials,accuracy,mean_logloss\n",
    );
    let mut probes = String::from(
        "seed,regime,noise,checkpoint,rule,model,correct,trials,accuracy,mean_logloss\n",
    );
    for &seed in &c.seeds {
        for (regime, name) in regimes.iter().enumerate() {
            for &noise in &[0.1, 0.25] {
                let r = run(c, seed, regime, noise);
                for (i, row) in r.rows.iter().enumerate() {
                    predictions.push_str(&format!(
                        "{seed},{name},{noise},{i},{},{},{},{},{},{},{},{}\n",
                        row.event.context,
                        row.event.input,
                        row.event.target,
                        row.predictions[0],
                        row.predictions[1],
                        row.predictions[2],
                        row.predictions[3],
                        row.predictions[4]
                    ));
                }
                for start in (0..r.rows.len()).step_by(WINDOW) {
                    let end = (start + WINDOW).min(r.rows.len());
                    let data = &r.rows[start..end];
                    for (k, model) in NAMES.iter().enumerate() {
                        let correct = data
                            .iter()
                            .filter(|x| usize::from(x.predictions[k] > 0.5) == x.event.target)
                            .count();
                        let l = data
                            .iter()
                            .map(|x| loss(x.predictions[k], x.event.target))
                            .sum::<f64>()
                            / data.len() as f64;
                        windows.push_str(&format!(
                            "{seed},{name},{noise},{start},{end},{model},{correct},{},{},{l}\n",
                            data.len(),
                            correct as f64 / data.len() as f64
                        ));
                    }
                }
                for p in &r.probes {
                    for (k, model) in NAMES.iter().enumerate() {
                        probes.push_str(&format!(
                            "{seed},{name},{noise},{},{},{model},{},{},{},{}\n",
                            p.phase + 1,
                            p.rule,
                            p.correct[k],
                            p.trials,
                            p.correct[k] as f64 / p.trials as f64,
                            p.logloss[k] / p.trials as f64
                        ));
                    }
                }
            }
        }
    }
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("windows.csv"), windows)?;
    fs::write(dir.join("probes.csv"), probes)?;
    fs::write(dir.join("summary.md"),"# Candidate context experiment\n\nSee probes.csv and windows.csv for per-seed comparisons.\n")
}

//! Fixed-capacity, unlabeled online hidden-state transition models.
//! This is a standard mixture baseline, not a learned graph or growth rule.
use crate::{continual_rule_change::Markov, temporal_stream::StreamConfig};
use murmur::data::Rng;
use std::{collections::HashSet, fs, io, path::Path};
pub const HAZARD: f64 = 0.01;
pub const BLOCK: usize = 200;
pub const WINDOW: usize = 200;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub input: usize,
    pub target: usize,
    pub context: usize,
}
/// Context is returned solely for generating/scoring and the labelled oracle.
pub fn stream(seed: u64, phase_len: usize, regime: usize, noise: f64) -> Vec<Event> {
    assert!(regime < 3 && (0.0..0.5).contains(&noise));
    let mut rng = Rng(seed);
    let mut x = (rng.next_u64() & 1) as usize;
    let mut out = Vec::new();
    for t in 0..3 * phase_len {
        let context = match regime {
            0 => usize::from(t >= phase_len && t < 2 * phase_len),
            1 => 0,
            _ => (t / BLOCK) % 2,
        };
        let flip = (context == 1) ^ (rng.uniform() < noise);
        let y = if flip { x ^ 1 } else { x };
        out.push(Event {
            input: x,
            target: y,
            context,
        });
        x = y;
    }
    out
}
#[derive(Clone)]
pub struct Mixture {
    pub p: [[f64; 2]; 2],
    pub belief: [f64; 2],
    pub rate: f64,
    pub hazard: f64,
    pub frozen: bool,
}
impl Mixture {
    pub fn new(seed: u64, rate: f64) -> Self {
        let mut r = Rng(seed);
        let mut p = [[0.0; 2]; 2];
        for row in &mut p {
            for x in row {
                *x = 0.2 + 0.6 * r.uniform();
            }
        }
        Self {
            p,
            belief: [0.5, 0.5],
            rate,
            hazard: HAZARD,
            frozen: false,
        }
    }
    pub fn known(noise: f64) -> Self {
        Self {
            p: [[noise, 1.0 - noise], [1.0 - noise, noise]],
            belief: [0.5, 0.5],
            rate: 0.0,
            hazard: HAZARD,
            frozen: true,
        }
    }
    pub fn reset(&mut self) {
        self.belief = [0.5, 0.5];
    }
    /// Predict from the prior; learn the posterior only after the target arrives.
    pub fn step(&mut self, x: usize, y: usize, learn: bool) -> (f64, [f64; 2], [f64; 2]) {
        let prior = [
            (1.0 - self.hazard) * self.belief[0] + self.hazard * self.belief[1],
            (1.0 - self.hazard) * self.belief[1] + self.hazard * self.belief[0],
        ];
        let p1 = prior[0] * self.p[0][x] + prior[1] * self.p[1][x];
        let likelihood = |k: usize| {
            if y == 1 {
                self.p[k][x]
            } else {
                1.0 - self.p[k][x]
            }
        };
        let q = [prior[0] * likelihood(0), prior[1] * likelihood(1)];
        let total = q[0] + q[1];
        self.belief = [q[0] / total, q[1] / total];
        if learn && !self.frozen {
            for k in 0..2 {
                self.p[k][x] = (self.p[k][x]
                    + self.rate * self.belief[k] * (y as f64 - self.p[k][x]))
                    .clamp(0.001, 0.999);
            }
        }
        (p1, prior, self.belief)
    }
}
#[derive(Clone)]
pub struct Models {
    pub single: Markov,
    pub mixture: Mixture,
    pub known: Mixture,
    pub oracle: Markov,
}
impl Models {
    pub fn new(seed: u64, rate: f64, noise: f64) -> Self {
        Self {
            single: Markov::new(rate, false),
            mixture: Mixture::new(seed.wrapping_add(51000), rate),
            known: Mixture::known(noise),
            oracle: Markov::new(rate, true),
        }
    }
    pub fn step(&mut self, e: Event, learn: bool) -> ([f64; 4], [f64; 2], [f64; 2]) {
        let prior_single = self.single.p[0][e.input];
        let prior_oracle = self.oracle.p[e.context][e.input];
        let (p, prior, posterior) = self.mixture.step(e.input, e.target, learn);
        let (known, _, _) = self.known.step(e.input, e.target, false);
        if learn {
            self.single.learn(e.input, e.target, e.context);
            self.oracle.learn(e.input, e.target, e.context);
        }
        ([prior_single, p, known, prior_oracle], prior, posterior)
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub index: usize,
    pub event: Event,
    pub p: [f64; 4],
    pub prior: [f64; 2],
    pub posterior: [f64; 2],
}
#[derive(Clone, Debug, PartialEq)]
pub struct Probe {
    pub phase: usize,
    pub rule: usize,
    pub correct: [usize; 4],
    pub logloss: [f64; 4],
    pub trials: usize,
}
pub struct Run {
    pub rows: Vec<Row>,
    pub probes: Vec<Probe>,
    pub slots: Vec<[[f64; 2]; 2]>,
}
fn loss(p: f64, y: usize) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    if y == 1 {
        -p.ln()
    } else {
        -(1.0 - p).ln()
    }
}
/// Probe cloned, frozen parameters; beliefs update from observations but no
/// rule label is supplied except to the explicitly labelled oracle.
pub fn probe(models: &Models, data: &[Event], rule: usize, phase: usize) -> Probe {
    let mut m = models.clone();
    m.mixture.reset();
    m.known.reset();
    let mut correct = [0; 4];
    let mut logloss = [0.0; 4];
    for e in data {
        let (p, _, _) = m.step(*e, false);
        for k in 0..4 {
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
    let data = stream(seed.wrapping_add(40000), c.train_tokens, regime, noise);
    let a = stream(seed.wrapping_add(46000), c.test_tokens, 1, noise);
    let mut b = a[..c.test_tokens].to_vec();
    let mut rng = Rng(seed.wrapping_add(48000));
    let mut x = (rng.next_u64() & 1) as usize;
    for e in &mut b {
        let y = if rng.uniform() < noise { x } else { x ^ 1 };
        *e = Event {
            input: x,
            target: y,
            context: 1,
        };
        x = y;
    }
    // The helper stream emits 3*phase_len; use exactly test_tokens probe events.
    let a = &a[..c.test_tokens];
    let b = &b[..c.test_tokens];
    let mut m = Models::new(seed, c.learning_rate, noise);
    let mut rows = Vec::new();
    let mut probes = Vec::new();
    let mut slots = Vec::new();
    for (i, e) in data.into_iter().enumerate() {
        let (p, prior, posterior) = m.step(e, true);
        rows.push(Row {
            index: i,
            event: e,
            p,
            prior,
            posterior,
        });
        if (i + 1) % c.train_tokens == 0 {
            let phase = i / c.train_tokens;
            probes.push(probe(&m, a, 0, phase));
            probes.push(probe(&m, b, 1, phase));
            slots.push(m.mixture.p);
        }
    }
    Run {
        rows,
        probes,
        slots,
    }
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
    fs::write(dir.join("protocol.txt"),"models=single recency Markov, fixed-two-slot online hidden-state mixture, known-rule hidden-state filter, oracle context Markov\nmixture=random seeded transition tables, hazard=0.01; posterior responsibilities update active input row; predict before observing target\nknown_filter=two exact data-generating rules and true noise supplied, no context labels; oracle_markov=true context supplied; other models receive no context labels\nregimes=ABA fixed phase lengths; stationary A; switching every 200; noise=0.1,0.25\nprobes=cloned frozen transition tables, uniform initial beliefs, beliefs updated from observed transitions; same separate A and B probe streams at checkpoints\nseed_roles=stream:s+40000 probeA:s+46000 probeB:s+48000 model_init:s+51000; wrapping u64\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let names = ["single", "mixture", "known_filter", "oracle"];
    let regimes = ["aba", "stationary", "switching"];
    let mut pred=String::from("seed,regime,noise,step,context,input,target,single_p1,mixture_p1,known_p1,oracle_p1,mixture_prior0,mixture_posterior0\n");
    let mut windows = String::from(
        "seed,regime,noise,window_start,window_end,model,correct,trials,accuracy,mean_logloss\n",
    );
    let mut probes = String::from(
        "seed,regime,noise,checkpoint,rule,model,correct,trials,accuracy,mean_logloss\n",
    );
    let mut slots = String::from("seed,regime,noise,checkpoint,slot,p1_if_0,p1_if_1\n");
    let mut aggregates = vec![[[0usize; 4]; 2]; 6];
    let mut counts = vec![[0usize; 2]; 6];
    for &seed in &c.seeds {
        for (regime, rname) in regimes.iter().enumerate() {
            for (ni, &noise) in [0.1, 0.25].iter().enumerate() {
                let r = run(c, seed, regime, noise);
                for x in &r.rows {
                    pred.push_str(&format!(
                        "{seed},{rname},{noise},{},{},{},{},{},{},{},{},{},{}\n",
                        x.index,
                        x.event.context,
                        x.event.input,
                        x.event.target,
                        x.p[0],
                        x.p[1],
                        x.p[2],
                        x.p[3],
                        x.prior[0],
                        x.posterior[0]
                    ));
                }
                for start in (0..r.rows.len()).step_by(WINDOW) {
                    let end = (start + WINDOW).min(r.rows.len());
                    let data = &r.rows[start..end];
                    for (k, name) in names.iter().enumerate() {
                        let correct = data
                            .iter()
                            .filter(|x| usize::from(x.p[k] > 0.5) == x.event.target)
                            .count();
                        let l = data
                            .iter()
                            .map(|x| loss(x.p[k], x.event.target))
                            .sum::<f64>()
                            / data.len() as f64;
                        windows.push_str(&format!(
                            "{seed},{rname},{noise},{start},{end},{name},{correct},{},{},{l}\n",
                            data.len(),
                            correct as f64 / data.len() as f64
                        ));
                    }
                }
                for p in &r.probes {
                    if p.phase == 2 {
                        counts[regime * 2 + ni][p.rule] += p.trials;
                        for k in 0..4 {
                            aggregates[regime * 2 + ni][p.rule][k] += p.correct[k];
                        }
                    }
                    for (k, name) in names.iter().enumerate() {
                        probes.push_str(&format!(
                            "{seed},{rname},{noise},{},{},{name},{},{},{},{}\n",
                            p.phase + 1,
                            p.rule,
                            p.correct[k],
                            p.trials,
                            p.correct[k] as f64 / p.trials as f64,
                            p.logloss[k] / p.trials as f64
                        ));
                    }
                }
                for (phase, tables) in r.slots.iter().enumerate() {
                    for (k, row) in tables.iter().enumerate() {
                        slots.push_str(&format!(
                            "{seed},{rname},{noise},{},{k},{},{}\n",
                            phase + 1,
                            row[0],
                            row[1]
                        ));
                    }
                }
            }
        }
    }
    fs::write(dir.join("predictions.csv"), pred)?;
    fs::write(dir.join("windows.csv"), windows)?;
    fs::write(dir.join("probes.csv"), probes)?;
    fs::write(dir.join("slots.csv"), slots)?;
    let mut summary=String::from("# Unlabeled context reuse: frozen probes after three phases\n\n| Regime | Noise | Probe | Single | Two-slot mixture | Known-rule filter | Oracle |\n|---|---:|---|---:|---:|---:|---:|\n");
    for (regime, name) in regimes.iter().enumerate() {
        for (ni, noise) in [0.1, 0.25].iter().enumerate() {
            for rule in 0..2 {
                let n = counts[regime * 2 + ni][rule] as f64;
                let v = aggregates[regime * 2 + ni][rule];
                summary.push_str(&format!(
                    "| {name} | {noise} | {} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                    if rule == 0 { "A" } else { "B" },
                    v[0] as f64 / n,
                    v[1] as f64 / n,
                    v[2] as f64 / n,
                    v[3] as f64 / n
                ));
            }
        }
    }
    fs::write(dir.join("summary.md"), summary)
}

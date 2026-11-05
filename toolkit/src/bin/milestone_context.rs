//! Matched candidate scorecard against fixed baselines on four task families.
use murmur_research_runner::agent_candidate;
use murmur::data::Rng;
use murmur_research_runner::{continual_rule_change::Markov, temporal_stream::StreamConfig, unlabeled_context::Mixture};
use std::{env, fs, io, path::Path};
use std::time::Instant;

const NAMES: [&str; 4] = ["single", "mixture", "candidate", "oracle"];
// Each row is P(y=1 | x=0), P(y=1 | x=1). Contexts are never passed
// to the candidate, mixture, or single baseline; only the oracle sees them.
const FAMILIES: [(&str, [[f64; 2]; 2]); 4] = [
    ("complementary", [[0.25, 0.75], [0.75, 0.25]]),
    ("asymmetric", [[0.15, 0.70], [0.65, 0.25]]),
    ("constant_bias", [[0.10, 0.10], [0.90, 0.90]]),
    ("unequal_noise", [[0.05, 0.90], [0.70, 0.30]]),
];
const EVALUATION_FAMILIES: [(&str, [[f64; 2]; 2]); 4] = [
    ("shifted_bias", [[0.20, 0.20], [0.80, 0.80]]),
    ("mixed_rules", [[0.10, 0.20], [0.75, 0.35]]),
    ("similar_rules", [[0.10, 0.70], [0.30, 0.90]]),
    ("reversed_asym", [[0.80, 0.30], [0.20, 0.75]]),
];
#[derive(Clone, Copy, Debug, PartialEq)]
struct Event { x: usize, y: usize, context: usize }
fn events(seed: u64, contexts: &[usize], rules: [[f64; 2]; 2]) -> Vec<Event> {
    let mut rng = Rng(seed);
    let mut x = (rng.next_u64() & 1) as usize;
    contexts.iter().map(|&context| {
        let y = usize::from(rng.uniform() < rules[context][x]);
        let event = Event { x, y, context };
        x = y;
        event
    }).collect()
}
#[derive(Clone)]
struct Models { single: Markov, mixture: Mixture, candidate: agent_candidate::Candidate, oracle: Markov }
impl Models {
    fn new(seed: u64, rate: f64) -> Self {
        Self { single: Markov::new(rate, false), mixture: Mixture::new(seed, rate),
               candidate: agent_candidate::Candidate::new(seed, rate),
               oracle: Markov::new(rate, true) }
    }
    fn reset(&mut self) { self.mixture.reset(); self.candidate.reset(); }
    fn step(&mut self, e: Event, learn: bool) -> [f64; 4] {
        let p = [self.single.p[0][e.x], self.mixture.step(e.x, e.y, learn).0,
                 self.candidate.predict(e.x), self.oracle.p[e.context][e.x]];
        assert!(p.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)));
        self.candidate.observe(e.x, e.y, learn);
        if learn { self.single.learn(e.x, e.y, e.context); self.oracle.learn(e.x, e.y, e.context); }
        p
    }
    fn timed_training_step(&mut self, e: Event, nanos: &mut [u128; 4]) -> [f64; 4] {
        let start = Instant::now();
        let single = self.single.p[0][e.x];
        self.single.learn(e.x, e.y, e.context);
        nanos[0] += start.elapsed().as_nanos();
        let start = Instant::now();
        let mixture = self.mixture.step(e.x, e.y, true).0;
        nanos[1] += start.elapsed().as_nanos();
        let start = Instant::now();
        let candidate = self.candidate.predict(e.x);
        self.candidate.observe(e.x, e.y, true);
        nanos[2] += start.elapsed().as_nanos();
        let start = Instant::now();
        let oracle = self.oracle.p[e.context][e.x];
        self.oracle.learn(e.x, e.y, e.context);
        nanos[3] += start.elapsed().as_nanos();
        let p = [single, mixture, candidate, oracle];
        assert!(p.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)));
        p
    }
}
#[derive(Default)]
struct Scores { correct: [usize; 4], loss: [f64; 4], count: usize }
impl Scores {
    fn add(&mut self, p: [f64; 4], y: usize) {
        self.count += 1;
        for k in 0..4 {
            self.correct[k] += usize::from(usize::from(p[k] > 0.5) == y);
            let prob = if y == 1 { p[k] } else { 1.0-p[k] };
            self.loss[k] -= prob.clamp(1e-12, 1.0).ln();
        }
    }
    fn rows(&self, out: &mut String, seed: u64, family: &str, phase: usize, kind: &str) {
        for (k, name) in NAMES.iter().enumerate() {
            out.push_str(&format!("{seed},{family},{phase},{kind},{name},{},{},{:.8},{:.8}\n", self.correct[k], self.count,
                self.correct[k] as f64/self.count as f64, self.loss[k]/self.count as f64));
        }
    }
}
fn run(c: &StreamConfig, seed: u64, family: &str, rules: [[f64; 2]; 2], out: &mut String, resources: &mut String) {
    let contexts: Vec<_> = (0..3*c.train_tokens).map(|t| usize::from(t >= c.train_tokens && t < 2*c.train_tokens)).collect();
    let train = events(seed.wrapping_add(40000), &contexts, rules);
    let probes = [events(seed.wrapping_add(46000), &vec![0; c.test_tokens], rules),
                  events(seed.wrapping_add(48000), &vec![1; c.test_tokens], rules)];
    let mut model = Models::new(seed.wrapping_add(51000), c.learning_rate);
    let mut nanos = [0u128; 4];
    for phase in 0..3 {
        let mut score = Scores::default();
        let mut window = Scores::default();
        let phase_events = &train[phase*c.train_tokens..(phase+1)*c.train_tokens];
        for (i, &e) in phase_events.iter().enumerate() {
            let p = model.timed_training_step(e, &mut nanos);
            score.add(p, e.y); window.add(p, e.y);
            if (i + 1) % 100 == 0 || i + 1 == phase_events.len() {
                window.rows(out, seed, family, phase+1, &format!("window_{:04}", (i / 100) * 100));
                window = Scores::default();
            }
        }
        score.rows(out, seed, family, phase+1, "online");
        for (k, probe) in probes.iter().enumerate() {
            let mut frozen = model.clone();
            frozen.reset();
            let mut score = Scores::default();
            for &e in probe { score.add(frozen.step(e, false), e.y); }
            score.rows(out, seed, family, phase+1, if k == 0 { "probe_a" } else { "probe_b" });
        }
    }
    let inline = [std::mem::size_of::<Markov>(), std::mem::size_of::<Mixture>(),
                  std::mem::size_of::<agent_candidate::Candidate>(), std::mem::size_of::<Markov>()];
    for k in 0..4 {
        resources.push_str(&format!("{seed},{family},{},{},{},{}\n", NAMES[k], inline[k], nanos[k], 3*c.train_tokens));
    }
}
fn execute(c: &StreamConfig, dir: &Path, families: [(&str, [[f64; 2]; 2]); 4]) -> io::Result<()> {
    fs::create_dir(dir)?; // Never overwrite an earlier experiment.
    let mut csv = String::from("seed,family,phase,measurement,model,correct,trials,accuracy,logloss\n");
    let mut resources = String::from("seed,family,model,inline_bytes,training_ns,training_steps\n");
    for &seed in &c.seeds { for (name, rules) in families { run(c, seed, name, rules, &mut csv, &mut resources); } }
    fs::write(dir.join("metrics.csv"), csv)?;
    fs::write(dir.join("resources.csv"), resources)?;
    fs::write(dir.join("status.txt"), "complete\n")?;
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len()!=3 { return Err("usage: milestone_context <config> <new-output-directory> <development|evaluation>".into()); }
    let c = StreamConfig::parse(&fs::read_to_string(&args[0])?)?;
    if c.train_tokens < 100 { return Err("train_tokens must be at least 100".into()); }
    let families = match args[2].as_str() { "development" => FAMILIES, "evaluation" => EVALUATION_FAMILIES,
                                             _ => return Err("unknown split".into()) };
    execute(&c, Path::new(&args[1]), families)?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn streams_are_deterministic_and_obey_extreme_rules() {
        let contexts = [0,1,0,1,1];
        let rules = [[0.0,0.0],[1.0,1.0]];
        let a = events(99, &contexts, rules);
        assert_eq!(a, events(99, &contexts, rules));
        for (e, &context) in a.iter().zip(&contexts) { assert_eq!(e.y, context); }
        for pair in a.windows(2) { assert_eq!(pair[0].y, pair[1].x); }
    }
    #[test]
    fn probes_do_not_train_models_and_use_identical_events() {
        let mut original = Models::new(17, 0.05);
        for e in events(7, &vec![0; 100], FAMILIES[0].1) { original.step(e, true); }
        let mut frozen = original.clone(); frozen.reset();
        for e in events(8, &vec![1; 100], FAMILIES[0].1) { frozen.step(e, false); }
        original.reset(); frozen.reset();
        for x in 0..2 {
            assert_eq!(original.candidate.predict(x), frozen.candidate.predict(x));
            assert_eq!(original.mixture.p, frozen.mixture.p);
            assert_eq!(original.single.p, frozen.single.p);
            assert_eq!(original.oracle.p, frozen.oracle.p);
        }
    }
    #[test]
    fn predictions_precede_target_and_comparison_is_reproducible() {
        let model = Models::new(42, 0.05);
        let mut a = model.clone();
        let mut b = model;
        assert_eq!(a.step(Event { x: 0, y: 0, context: 0 }, false),
                   b.step(Event { x: 0, y: 1, context: 0 }, false));
        let c = StreamConfig { seeds: vec![42], delays: vec![1], train_tokens: 25, test_tokens: 20,
                               units: 128, degree: 4, learning_rate: 0.05 };
        let mut first = String::new(); let mut second = String::new();
        run(&c, 42, FAMILIES[1].0, FAMILIES[1].1, &mut first, &mut String::new());
        run(&c, 42, FAMILIES[1].0, FAMILIES[1].1, &mut second, &mut String::new());
        assert_eq!(first, second);
        assert_eq!(first.lines().count(), 48);
    }
}

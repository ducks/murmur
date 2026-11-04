//! Matched, exploratory comparison of the two committed campaign candidates.
#[path = "../candidate_transfer/posterior.rs"]
mod posterior;
#[path = "../candidate_transfer/predictive.rs"]
mod predictive;
#[path = "../candidate_transfer/neutral.rs"]
mod neutral;
use murmur::data::Rng;
use murmur_research_runner::{continual_rule_change::Markov, temporal_stream::StreamConfig, unlabeled_context::Mixture};
use std::{env, fs, io, path::Path};

const NAMES: [&str; 9] = ["single", "mixture", "posterior", "predictive", "posterior_neutral", "predictive_neutral", "posterior_random", "predictive_random", "oracle"];
// Each row is P(y=1 | x=0), P(y=1 | x=1). Contexts are never passed
// to the candidate, mixture, or single baseline; only the oracle sees them.
const FAMILIES: [(&str, [[f64; 2]; 2]); 4] = [
    ("complementary", [[0.25, 0.75], [0.75, 0.25]]),
    ("asymmetric", [[0.15, 0.70], [0.65, 0.25]]),
    ("constant_bias", [[0.10, 0.10], [0.90, 0.90]]),
    ("unequal_noise", [[0.05, 0.90], [0.70, 0.30]]),
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
struct Models { single: Markov, mixture: Mixture, posterior: posterior::Candidate, predictive: predictive::Candidate, neutral: [neutral::Candidate; 4], oracle: Markov }
impl Models {
    fn new(seed: u64, rate: f64) -> Self {
        Self { single: Markov::new(rate, false), mixture: Mixture::new(seed, rate),
               posterior: posterior::Candidate::new(seed, rate), predictive: predictive::Candidate::new(seed, rate),
               neutral: [neutral::Candidate::new(seed, rate, false, false), neutral::Candidate::new(seed, rate, true, false),
                         neutral::Candidate::new(seed, rate, false, true), neutral::Candidate::new(seed, rate, true, true)],
               oracle: Markov::new(rate, true) }
    }
    fn reset(&mut self) { self.mixture.reset(); self.posterior.reset(); self.predictive.reset(); for m in &mut self.neutral { m.reset(); } }
    fn step(&mut self, e: Event, learn: bool) -> [f64; 9] {
        let p = [self.single.p[0][e.x], self.mixture.step(e.x, e.y, learn).0,
                 self.posterior.predict(e.x), self.predictive.predict(e.x),
                 self.neutral[0].predict(e.x), self.neutral[1].predict(e.x),
                 self.neutral[2].predict(e.x), self.neutral[3].predict(e.x), self.oracle.p[e.context][e.x]];
        assert!(p.iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x)));
        self.posterior.observe(e.x, e.y, learn);
        self.predictive.observe(e.x, e.y, learn);
        for model in &mut self.neutral { model.observe(e.x, e.y, learn); }
        if learn { self.single.learn(e.x, e.y, e.context); self.oracle.learn(e.x, e.y, e.context); }
        p
    }
}
#[derive(Default)]
struct Scores { correct: [usize; 9], loss: [f64; 9], count: usize }
impl Scores {
    fn add(&mut self, p: [f64; 9], y: usize) {
        self.count += 1;
        for k in 0..9 {
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
fn run(c: &StreamConfig, seed: u64, family: &str, rules: [[f64; 2]; 2], out: &mut String) {
    let contexts: Vec<_> = (0..3*c.train_tokens).map(|t| usize::from(t >= c.train_tokens && t < 2*c.train_tokens)).collect();
    let train = events(seed.wrapping_add(40000), &contexts, rules);
    let probes = [events(seed.wrapping_add(46000), &vec![0; c.test_tokens], rules),
                  events(seed.wrapping_add(48000), &vec![1; c.test_tokens], rules)];
    let mut model = Models::new(seed.wrapping_add(51000), c.learning_rate);
    for phase in 0..3 {
        let mut score = Scores::default();
        let mut window = Scores::default();
        let phase_events = &train[phase*c.train_tokens..(phase+1)*c.train_tokens];
        for (i, &e) in phase_events.iter().enumerate() {
            let p = model.step(e, true);
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
}
fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    fs::create_dir(dir)?; // Never overwrite an earlier experiment.
    let mut csv = String::from("seed,family,phase,measurement,model,correct,trials,accuracy,logloss\n");
    for &seed in &c.seeds { for (name, rules) in FAMILIES { run(c, seed, name, rules, &mut csv); } }
    fs::write(dir.join("metrics.csv"), csv)?;
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len()!=2 { return Err("usage: candidate_ablation <config> <new-output-directory>".into()); }
    let c = StreamConfig::parse(&fs::read_to_string(&args[0])?)?;
    if c.train_tokens < 100 { return Err("train_tokens must be at least 100 for switch windows".into()); }
    execute(&c, Path::new(&args[1]))?;
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
            assert_eq!(original.posterior.predict(x), frozen.posterior.predict(x));
            assert_eq!(original.predictive.predict(x), frozen.predictive.predict(x));
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
        run(&c, 42, FAMILIES[1].0, FAMILIES[1].1, &mut first);
        run(&c, 42, FAMILIES[1].0, FAMILIES[1].1, &mut second);
        assert_eq!(first, second);
        assert_eq!(first.lines().count(), 108);
    }
}

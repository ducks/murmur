//! Candidate memories must outperform a control on later, frozen evaluation data.
use crate::discovery::{
    stream_with_block, Discovery, Example, MAX_MODELS, SUSTAINED, THRESHOLD, WARMUP, WINDOW,
};

pub const CANDIDATE_TRAIN: usize = 100;
pub const CANDIDATE_EVAL: usize = 100;
pub const COOLDOWN: usize = 100;
pub const TRAIN_STEPS: usize = 1200;
pub const TEST_STEPS: usize = 400;

#[derive(Debug)]
pub struct Trial {
    pub start: usize,
    pub end: usize,
    pub completed: bool,
    pub gain: f64,
    pub penalty: f64,
    pub accepted: bool,
}
#[derive(Debug)]
pub struct Trace {
    pub step: usize,
    pub context: usize,
    pub source: usize,
    pub target: usize,
    pub prediction: usize,
    pub surprise: f64,
    pub mode: &'static str,
    pub model_count: usize,
}
#[derive(Debug)]
pub struct ResultSet {
    pub model: Discovery,
    pub fixed: Discovery,
    pub single: Discovery,
    pub trials: Vec<Trial>,
    pub trace: Vec<Trace>,
    pub test: Vec<(Example, usize, usize, usize)>,
}
struct Active {
    candidate: Discovery,
    start: usize,
    train: usize,
    eval: usize,
    gain: f64,
}
pub fn accept(gain: f64, penalty: f64) -> bool {
    gain > penalty
}
/// Stream labels are passed through for analysis only; neither trial opening,
/// candidate learning, nor selection receives labels or switch boundaries.
pub fn run(
    seed: u64,
    switching: bool,
    noise: f64,
    penalty: f64,
) -> Result<ResultSet, &'static str> {
    run_custom(
        seed,
        switching,
        noise,
        penalty,
        TRAIN_STEPS,
        TEST_STEPS,
        100,
    )
}

pub fn run_custom(
    seed: u64,
    switching: bool,
    noise: f64,
    penalty: f64,
    train_steps: usize,
    test_steps: usize,
    block: usize,
) -> Result<ResultSet, &'static str> {
    run_scheduled(
        seed,
        switching,
        noise,
        penalty,
        train_steps,
        test_steps,
        block,
        0.0,
    )
}

/// Raise the penalty on later completed attempts. This is a heuristic schedule,
/// not a statistically calibrated sequential test.
pub fn scheduled_penalty(base: f64, scale: f64, attempt: usize) -> f64 {
    assert!(attempt > 0);
    base + scale * (attempt as f64).ln()
}

pub fn run_scheduled(
    seed: u64,
    switching: bool,
    noise: f64,
    penalty: f64,
    train_steps: usize,
    test_steps: usize,
    block: usize,
    scale: f64,
) -> Result<ResultSet, &'static str> {
    if !noise.is_finite()
        || !(0.0..=1.0).contains(&noise)
        || !penalty.is_finite()
        || penalty < 0.0
        || train_steps == 0
        || test_steps == 0
        || block == 0
        || !scale.is_finite()
        || scale < 0.0
    {
        return Err("invalid configuration");
    }
    let mut model = Discovery::new(1, false, seed.wrapping_add(501));
    let mut fixed = Discovery::new(2, false, seed.wrapping_add(501));
    let mut single = Discovery::new(1, false, seed.wrapping_add(501));
    let mut active: Option<Active> = None;
    let mut records = Vec::new();
    let mut trace = Vec::new();
    let (mut sum, mut count, mut high, mut cooldown) = (0.0, 0usize, 0usize, 0usize);
    for (i, e) in stream_with_block(seed.wrapping_add(502), train_steps, switching, noise, block)
        .into_iter()
        .enumerate()
    {
        let step = i + 1;
        fixed.step(e.source, e.target, true);
        single.step(e.source, e.target, true);
        let mut completed = None;
        let (obs, mode) = if let Some(ref mut t) = active {
            if t.train < CANDIDATE_TRAIN {
                let o = model.step(e.source, e.target, true);
                t.candidate.step(e.source, e.target, true);
                t.train += 1;
                (o, "candidate_train")
            } else {
                let o = model.step(e.source, e.target, false);
                let candidate = t.candidate.step(e.source, e.target, false);
                t.gain += o.surprise - candidate.surprise;
                t.eval += 1;
                if t.eval == CANDIDATE_EVAL {
                    completed = Some((t.start, t.gain));
                }
                (o, "candidate_eval")
            }
        } else {
            let o = model.step(e.source, e.target, true);
            if cooldown > 0 {
                cooldown -= 1;
            } else if model.models.len() < MAX_MODELS {
                sum += o.surprise;
                count += 1;
                if count == WINDOW {
                    let mean = sum / WINDOW as f64;
                    if step >= WARMUP && mean > THRESHOLD {
                        high += 1;
                    } else {
                        high = 0;
                    }
                    if high >= SUSTAINED {
                        if let Some(candidate) = model.candidate_branch() {
                            active = Some(Active {
                                candidate,
                                start: step,
                                train: 0,
                                eval: 0,
                                gain: 0.0,
                            });
                        }
                        high = 0;
                    }
                    sum = 0.0;
                    count = 0;
                }
            }
            (o, "normal")
        };
        trace.push(Trace {
            step,
            context: e.context,
            source: e.source,
            target: e.target,
            prediction: obs.prediction,
            surprise: obs.surprise,
            mode,
            model_count: model.models.len(),
        });
        if let Some((start, gain)) = completed {
            let t = active.take().unwrap();
            let threshold = scheduled_penalty(penalty, scale, records.len() + 1);
            let accepted = accept(gain, threshold);
            if accepted {
                model = t.candidate;
            }
            records.push(Trial {
                start,
                end: step,
                completed: true,
                gain,
                penalty: threshold,
                accepted,
            });
            sum = 0.0;
            count = 0;
            high = 0;
            cooldown = COOLDOWN;
        }
    }
    if let Some(t) = active {
        records.push(Trial {
            start: t.start,
            end: train_steps,
            completed: false,
            gain: t.gain,
            penalty: scheduled_penalty(penalty, scale, records.len() + 1),
            accepted: false,
        });
    }
    model.reset_belief();
    fixed.reset_belief();
    single.reset_belief();
    let mut test = Vec::new();
    for e in stream_with_block(seed.wrapping_add(503), test_steps, switching, noise, block) {
        let a = model.step(e.source, e.target, false);
        let b = fixed.step(e.source, e.target, false);
        let c = single.step(e.source, e.target, false);
        test.push((e, a.prediction, b.prediction, c.prediction));
    }
    Ok(ResultSet {
        model,
        fixed,
        single,
        trials: records,
        trace,
        test,
    })
}

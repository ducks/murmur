//! Online feedback control matched to the task model's per-step L2 edge update.
use crate::{
    temporal_plasticity::{self, PlasticModel, WEIGHT_LIMIT},
    temporal_stream::{self, StreamConfig},
};
use std::{collections::HashSet, fs, io, path::Path};

pub struct Checkpoint {
    pub step: usize,
    pub reference_norm_sum: f64,
    pub control_norm_sum: f64,
    pub max_norm_difference: f64,
    pub task_drift: f64,
    pub control_drift: f64,
    pub task_clipped: usize,
    pub control_clipped: usize,
}
pub struct Run {
    pub rows: Vec<temporal_plasticity::Row>,
    pub matched: Vec<usize>,
    pub checkpoints: Vec<Checkpoint>,
}
pub fn run(c: &StreamConfig, seed: u64, delay: usize) -> Run {
    let baseline = temporal_plasticity::run_delay(c, seed, delay);
    let train = temporal_stream::stream(seed.wrapping_add(31000), c.train_tokens);
    let targets = temporal_stream::stream(seed.wrapping_add(34000), c.train_tokens);
    let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
    let mut task = PlasticModel::new(c, seed);
    let mut control = PlasticModel::new(c, seed);
    let initial: Vec<f64> = task.graph.edges.iter().map(|e| e.weight).collect();
    let mut ref_sum = 0.0;
    let mut ctrl_sum = 0.0;
    let mut max_diff: f64 = 0.0;
    let mut checkpoints = Vec::new();
    for (t, &symbol) in train.iter().enumerate() {
        task.advance(symbol, true);
        control.advance(symbol, true);
        if t >= delay {
            let before: Vec<f64> = task.graph.edges.iter().map(|e| e.weight).collect();
            task.learn(train[t - delay], train[t - delay]);
            let norm = task
                .graph
                .edges
                .iter()
                .zip(before)
                .map(|(e, w)| (e.weight - w).powi(2))
                .sum::<f64>()
                .sqrt();
            let actual = control.learn_with_update_norm(train[t - delay], targets[t], norm);
            ref_sum += norm;
            ctrl_sum += actual;
            max_diff = max_diff.max((norm - actual).abs());
        }
        if (t + 1) % 1000 == 0 || t + 1 == train.len() {
            let stats = |m: &PlasticModel| {
                let drift = m
                    .graph
                    .edges
                    .iter()
                    .zip(&initial)
                    .map(|(e, w)| (e.weight - w).abs())
                    .sum::<f64>()
                    / initial.len() as f64;
                let clipped = m
                    .graph
                    .edges
                    .iter()
                    .filter(|e| e.weight.abs() >= WEIGHT_LIMIT - 1e-12)
                    .count();
                (drift, clipped)
            };
            let (task_drift, task_clipped) = stats(&task);
            let (control_drift, control_clipped) = stats(&control);
            checkpoints.push(Checkpoint {
                step: t + 1,
                reference_norm_sum: ref_sum,
                control_norm_sum: ctrl_sum,
                max_norm_difference: max_diff,
                task_drift,
                control_drift,
                task_clipped,
                control_clipped,
            });
        }
    }
    task.reset_state();
    control.reset_state();
    let mut matched = Vec::new();
    for (t, &symbol) in test.iter().enumerate() {
        let task_prediction = task.advance(symbol, false);
        let prediction = control.advance(symbol, false);
        if t >= delay {
            assert_eq!(task_prediction, baseline.rows[t - delay].adaptive);
            matched.push(prediction);
        }
    }
    Run {
        rows: baseline.rows,
        matched,
        checkpoints,
    }
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let mut seen = HashSet::new();
    for s in &c.seeds {
        for o in [0, 31000, 32000, 33000, 34000] {
            if !seen.insert(s.wrapping_add(o)) {
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
    fs::write(dir.join("protocol.txt"),"task=delayed bit recall\nmatched_control=unrelated target feedback normalized each step to L2 norm of task graph realized edge update; clipping may reduce realized norm\nmatching=nonlocal cross-model intervention; task and control have separate traces, weights, states and true-label-trained readouts\nseeds=graph:s train:s+31000 test:s+32000 feedback:s+33000 unrelated:s+34000 (wrapping u64)\nevaluation=reset state; freeze weights and readers\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let mut metrics = String::from("seed,delay,model,correct,trials,accuracy\n");
    let mut predictions = String::from(
        "seed,delay,index,input,target,fixed,task,unrelated,matched_unrelated,shift\n",
    );
    let mut diagnostics=String::from("seed,delay,step,reference_update_norm_sum,control_update_norm_sum,max_per_step_norm_difference,task_mean_absolute_drift,control_mean_absolute_drift,task_clipped,control_clipped\n");
    let names = ["fixed", "task", "unrelated", "matched_unrelated", "shift"];
    let mut sums = vec![[0usize; 5]; c.delays.len()];
    let mut counts = vec![0usize; c.delays.len()];
    for &seed in &c.seeds {
        for (di, &delay) in c.delays.iter().enumerate() {
            let r = run(c, seed, delay);
            let mut correct = [0usize; 5];
            for (i, row) in r.rows.iter().enumerate() {
                let b = &row.baseline;
                let v = [
                    b.graph,
                    row.adaptive,
                    row.random_feedback,
                    r.matched[i],
                    b.shift,
                ];
                for k in 0..5 {
                    correct[k] += usize::from(v[k] == b.target);
                }
                predictions.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{}\n",
                    b.index, b.input, b.target, v[0], v[1], v[2], v[3], v[4]
                ));
            }
            let n = r.rows.len();
            counts[di] += n;
            for k in 0..5 {
                sums[di][k] += correct[k];
                metrics.push_str(&format!(
                    "{seed},{delay},{},{},{n},{}\n",
                    names[k],
                    correct[k],
                    correct[k] as f64 / n as f64
                ));
            }
            for x in r.checkpoints {
                diagnostics.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{}\n",
                    x.step,
                    x.reference_norm_sum,
                    x.control_norm_sum,
                    x.max_norm_difference,
                    x.task_drift,
                    x.control_drift,
                    x.task_clipped,
                    x.control_clipped
                ));
            }
        }
    }
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("diagnostics.csv"), diagnostics)?;
    let mut summary=String::from("# Online update-magnitude matching\n\n| Delay | Fixed | Task | Unrelated | Matched unrelated | Shift |\n|---:|---:|---:|---:|---:|---:|\n");
    for (i, d) in c.delays.iter().enumerate() {
        summary.push_str(&format!(
            "| {d} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            sums[i][0] as f64 / counts[i] as f64,
            sums[i][1] as f64 / counts[i] as f64,
            sums[i][2] as f64 / counts[i] as f64,
            sums[i][3] as f64 / counts[i] as f64,
            sums[i][4] as f64 / counts[i] as f64
        ));
    }
    fs::write(dir.join("summary.md"), summary)
}

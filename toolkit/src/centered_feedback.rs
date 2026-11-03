//! Remove the reader-dependent component from random-target edge feedback.
use crate::{
    connection_strength::{impulse, HORIZON, PROBES},
    temporal_plasticity::{self, PlasticModel, WEIGHT_LIMIT},
    temporal_stream::{self, StreamConfig},
    trained_strength,
};
use murmur::graph::Graph;
use std::{collections::HashSet, fs, io, path::Path};

pub struct Checkpoint {
    pub step: usize,
    pub scored: usize,
    pub mean_probability: f64,
    pub mean_pseudo_error: f64,
    pub mean_centered_error: f64,
    pub pseudo_drift: f64,
    pub centered_drift: f64,
    pub pseudo_mean_abs: f64,
    pub centered_mean_abs: f64,
    pub pseudo_clipped: usize,
    pub centered_clipped: usize,
}
pub struct Run {
    pub baseline: temporal_plasticity::Run,
    pub graphs: Vec<Graph>,
    pub frozen: Vec<Vec<usize>>,
    pub centered_online: Vec<usize>,
    pub checkpoints: Vec<Checkpoint>,
}
pub fn run(c: &StreamConfig, seed: u64, delay: usize) -> Run {
    let baseline = temporal_plasticity::run_delay(c, seed, delay);
    let train = temporal_stream::stream(seed.wrapping_add(31000), c.train_tokens);
    let fake = temporal_stream::stream(seed.wrapping_add(34000), c.train_tokens);
    let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
    let mut pseudo = PlasticModel::new(c, seed);
    let mut centered = PlasticModel::new(c, seed);
    let initial: Vec<_> = pseudo.graph.edges.iter().map(|e| e.weight).collect();
    let (mut prob_sum, mut pseudo_sum, mut centered_sum, mut scored) = (0.0, 0.0, 0.0, 0usize);
    let mut checkpoints = Vec::new();
    for (t, &x) in train.iter().enumerate() {
        pseudo.advance(x, true);
        centered.advance(x, true);
        if t >= delay {
            let p = pseudo.class_one_probability();
            let a = fake[t] as f64 - p;
            let b = fake[t] as f64 - 0.5;
            prob_sum += p;
            pseudo_sum += a;
            centered_sum += b;
            scored += 1;
            pseudo.learn(train[t - delay], fake[t]);
            centered.learn_with_error(train[t - delay], b);
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
                (drift, trained_strength::mean_abs(&m.graph), clipped)
            };
            let (pd, pm, pc) = stats(&pseudo);
            let (cd, cm, cc) = stats(&centered);
            let n = scored.max(1) as f64;
            checkpoints.push(Checkpoint {
                step: t + 1,
                scored,
                mean_probability: prob_sum / n,
                mean_pseudo_error: pseudo_sum / n,
                mean_centered_error: centered_sum / n,
                pseudo_drift: pd,
                centered_drift: cd,
                pseudo_mean_abs: pm,
                centered_mean_abs: cm,
                pseudo_clipped: pc,
                centered_clipped: cc,
            });
        }
    }
    assert!(pseudo
        .graph
        .edges
        .iter()
        .zip(&baseline.random_feedback.edges)
        .all(|(a, b)| a.weight == b.weight));
    let graphs = vec![
        baseline.initial.clone(),
        baseline.random_feedback.clone(),
        centered.graph.clone(),
        baseline.adaptive.clone(),
    ];
    let frozen = graphs
        .iter()
        .map(|g| trained_strength::predict(c, seed, delay, g.clone()))
        .collect();
    pseudo.reset_state();
    centered.reset_state();
    let mut centered_online = Vec::new();
    for (t, &x) in test.iter().enumerate() {
        let p = pseudo.advance(x, false);
        let y = centered.advance(x, false);
        if t >= delay {
            assert_eq!(p, baseline.rows[t - delay].random_feedback);
            centered_online.push(y);
        }
    }
    Run {
        baseline,
        graphs,
        frozen,
        centered_online,
        checkpoints,
    }
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let mut used = HashSet::new();
    for s in &c.seeds {
        for o in [0, 31000, 32000, 33000, 34000, 35000] {
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
    fs::write(dir.join("protocol.txt"),"task=delayed bit recall\ncomparison=pseudo edge scalar random_bit-sigmoid(class1 score) vs centered edge scalar random_bit-0.5; same feedback signs, initial graph, random bits, input and true-label readers\nreadout=online true labels; frozen graph comparisons retrain reader on the same train stream\nseed_roles=graph:s train:s+31000 test:s+32000 feedback:s+33000 pseudo:s+34000 probe:s+35000 wrapping u64\nprobe=64 reset trajectory pairs, first bit differs followed by common input, RMS distance lags 0..16\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let names = [
        "initial_frozen",
        "pseudo_frozen",
        "centered_frozen",
        "task_frozen",
        "pseudo_online",
        "centered_online",
        "shift",
    ];
    let mut metrics = String::from("seed,delay,model,correct,trials,accuracy\n");
    let mut predictions=String::from("seed,delay,index,input,target,initial_frozen,pseudo_frozen,centered_frozen,task_frozen,pseudo_online,centered_online,shift\n");
    let mut diagnostics=String::from("seed,delay,step,scored,mean_probability,mean_pseudo_error,mean_centered_error,pseudo_mean_abs_drift,centered_mean_abs_drift,pseudo_mean_abs_weight,centered_mean_abs_weight,pseudo_clipped,centered_clipped\n");
    let mut probes = String::from("seed,delay,model,lag,mean_rms_distance\n");
    let mut weights = String::from("seed,delay,edge_id,from,to,initial,pseudo,centered,task\n");
    let mut sums = vec![[0usize; 7]; c.delays.len()];
    let mut counts = vec![0usize; c.delays.len()];
    for &seed in &c.seeds {
        for (di, &delay) in c.delays.iter().enumerate() {
            let r = run(c, seed, delay);
            let mut correct = [0usize; 7];
            for (i, row) in r.baseline.rows.iter().enumerate() {
                let b = &row.baseline;
                let v = [
                    r.frozen[0][i],
                    r.frozen[1][i],
                    r.frozen[2][i],
                    r.frozen[3][i],
                    row.random_feedback,
                    r.centered_online[i],
                    b.shift,
                ];
                for k in 0..7 {
                    correct[k] += usize::from(v[k] == b.target);
                }
                predictions.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{},{},{}\n",
                    b.index, b.input, b.target, v[0], v[1], v[2], v[3], v[4], v[5], v[6]
                ));
            }
            let n = r.baseline.rows.len();
            counts[di] += n;
            for k in 0..7 {
                sums[di][k] += correct[k];
                metrics.push_str(&format!(
                    "{seed},{delay},{},{},{n},{}\n",
                    names[k],
                    correct[k],
                    correct[k] as f64 / n as f64
                ));
            }
            for x in &r.checkpoints {
                diagnostics.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{},{},{},{}\n",
                    x.step,
                    x.scored,
                    x.mean_probability,
                    x.mean_pseudo_error,
                    x.mean_centered_error,
                    x.pseudo_drift,
                    x.centered_drift,
                    x.pseudo_mean_abs,
                    x.centered_mean_abs,
                    x.pseudo_clipped,
                    x.centered_clipped
                ));
            }
            for (k, g) in r.graphs.iter().enumerate() {
                for (lag, x) in impulse(g, seed.wrapping_add(35000), PROBES, HORIZON)
                    .iter()
                    .enumerate()
                {
                    probes.push_str(&format!("{seed},{delay},{},{lag},{x}\n", names[k]));
                }
            }
            for i in 0..r.graphs[0].edges.len() {
                let e = &r.graphs[0].edges[i];
                weights.push_str(&format!(
                    "{seed},{delay},{i},{},{},{:.17},{:.17},{:.17},{:.17}\n",
                    e.from,
                    e.to,
                    e.weight,
                    r.graphs[1].edges[i].weight,
                    r.graphs[2].edges[i].weight,
                    r.graphs[3].edges[i].weight
                ));
            }
        }
    }
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("diagnostics.csv"), diagnostics)?;
    fs::write(dir.join("impulse.csv"), probes)?;
    fs::write(dir.join("weights.csv"), weights)?;
    let mut summary=String::from("# Centered random feedback\n\n| Delay | Fixed | Pseudo frozen | Centered frozen | Task frozen | Pseudo online | Centered online | Shift |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (i, d) in c.delays.iter().enumerate() {
        let n = counts[i] as f64;
        summary.push_str(&format!(
            "| {d} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            sums[i][0] as f64 / n,
            sums[i][1] as f64 / n,
            sums[i][2] as f64 / n,
            sums[i][3] as f64 / n,
            sums[i][4] as f64 / n,
            sums[i][5] as f64 / n,
            sums[i][6] as f64 / n
        ));
    }
    fs::write(dir.join("summary.md"), summary)
}

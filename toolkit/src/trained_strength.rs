//! Intervene on the average absolute strength of trained and initial graphs.
use crate::{
    connection_strength::{impulse, HORIZON, PROBES},
    temporal_plasticity::{self, WEIGHT_LIMIT},
    temporal_stream::{self, StreamConfig},
};
use murmur::{graph::Graph, learning::Readout};
use std::{collections::HashSet, fs, io, path::Path};

pub fn mean_abs(g: &Graph) -> f64 {
    g.edges.iter().map(|e| e.weight.abs()).sum::<f64>() / g.edges.len() as f64
}
/// Uniform positive rescaling, with clipping, preserves each edge's sign. The
/// target mean is found by bisection; zero-valued edges cannot be made nonzero.
pub fn match_strength(g: &Graph, target: f64) -> Result<(Graph, f64), String> {
    if g.edges.is_empty() || !target.is_finite() || target < 0.0 {
        return Err("invalid strength target".into());
    }
    let max = WEIGHT_LIMIT * g.edges.iter().filter(|e| e.weight != 0.0).count() as f64
        / g.edges.len() as f64;
    if target > max + 1e-12 {
        return Err("strength target is unreachable".into());
    }
    let mean = |scale: f64| {
        g.edges
            .iter()
            .map(|e| (e.weight * scale).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT).abs())
            .sum::<f64>()
            / g.edges.len() as f64
    };
    let (mut low, mut high) = (0.0, 1.0);
    while mean(high) < target - 1e-14 {
        high *= 2.0;
        if !high.is_finite() {
            return Err("could not reach target".into());
        }
    }
    for _ in 0..100 {
        let mid = (low + high) / 2.0;
        if mean(mid) < target {
            low = mid;
        } else {
            high = mid;
        }
    }
    let factor = (low + high) / 2.0;
    let mut result = g.clone();
    for e in &mut result.edges {
        e.weight = (e.weight * factor).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
    }
    Ok((result, factor))
}
fn predict(c: &StreamConfig, seed: u64, delay: usize, mut graph: Graph) -> Vec<usize> {
    let train = temporal_stream::stream(seed.wrapping_add(31000), c.train_tokens);
    let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
    let mut reader = Readout::new(c.units, c.learning_rate);
    for (t, &x) in train.iter().enumerate() {
        graph.step(x);
        if t >= delay {
            reader.learn(&graph.state, train[t - delay]);
        }
    }
    graph.reset();
    test.iter()
        .enumerate()
        .filter_map(|(t, &x)| {
            graph.step(x);
            (t >= delay).then(|| reader.predict(&graph.state))
        })
        .collect()
}
pub struct Run {
    pub baseline: temporal_plasticity::Run,
    pub graphs: Vec<Graph>,
    pub predictions: Vec<Vec<usize>>,
    pub scaled_initial_factor: f64,
    pub normalized_random_factor: f64,
}
pub fn run(c: &StreamConfig, seed: u64, delay: usize) -> Run {
    let baseline = temporal_plasticity::run_delay(c, seed, delay);
    let init_strength = mean_abs(&baseline.initial);
    let learned_strength = mean_abs(&baseline.random_feedback);
    let (scaled_initial, scaled_initial_factor) =
        match_strength(&baseline.initial, learned_strength)
            .expect("learned target reachable by initial graph");
    let (normalized_random, normalized_random_factor) =
        match_strength(&baseline.random_feedback, init_strength)
            .expect("initial target reachable by learned graph");
    let graphs = vec![
        baseline.initial.clone(),
        scaled_initial,
        baseline.random_feedback.clone(),
        normalized_random,
        baseline.adaptive.clone(),
    ];
    let predictions: Vec<_> = graphs
        .iter()
        .map(|g| predict(c, seed, delay, g.clone()))
        .collect();
    assert!(predictions[0]
        .iter()
        .zip(&baseline.rows)
        .all(|(x, r)| *x == r.baseline.graph));
    Run {
        baseline,
        graphs,
        predictions,
        scaled_initial_factor,
        normalized_random_factor,
    }
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let mut seeds = HashSet::new();
    for s in &c.seeds {
        for o in [0, 31000, 32000, 33000, 34000, 35000] {
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
    fs::write(dir.join("protocol.txt"),"task=delayed bit recall\ntrained_graphs=online true-target reader and either task or unrelated-target recurrent feedback\nintervention=uniform nonnegative weight scaling with clipping; bisection matches per-run mean absolute edge weight\nfrozen_readers=separately initialized and trained on the same training stream after graphs are fixed; state reset and weights frozen for evaluation\nimpulse=64 pairs start in zero state, first inputs 0 vs 1, then same random inputs; RMS state distance at lags 0..16\nseeds=graph:s train:s+31000 test:s+32000 feedback:s+33000 random_target:s+34000 impulse:s+35000, wrapping u64\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let names = [
        "initial",
        "initial_strength_matched",
        "random_trained",
        "random_normalized",
        "task_trained",
        "online_random",
        "shift",
    ];
    let mut predictions=String::from("seed,delay,index,input,target,initial,initial_strength_matched,random_trained,random_normalized,task_trained,online_random,shift\n");
    let mut metrics = String::from("seed,delay,model,correct,trials,accuracy\n");
    let mut diagnostics =
        String::from("seed,delay,model,mean_abs_weight,clipped_edges,total_edges,scale_factor\n");
    let mut probes = String::from("seed,delay,model,lag,mean_rms_distance\n");
    let mut weights=String::from("seed,delay,edge_id,from,to,initial,initial_strength_matched,random_trained,random_normalized,task_trained\n");
    let mut totals = vec![[0usize; 7]; c.delays.len()];
    let mut counts = vec![0usize; c.delays.len()];
    for &seed in &c.seeds {
        for (di, &delay) in c.delays.iter().enumerate() {
            let r = run(c, seed, delay);
            let mut correct = [0usize; 7];
            for (i, row) in r.baseline.rows.iter().enumerate() {
                let b = &row.baseline;
                let v = [
                    r.predictions[0][i],
                    r.predictions[1][i],
                    r.predictions[2][i],
                    r.predictions[3][i],
                    r.predictions[4][i],
                    row.random_feedback,
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
                totals[di][k] += correct[k];
                metrics.push_str(&format!(
                    "{seed},{delay},{},{},{n},{}\n",
                    names[k],
                    correct[k],
                    correct[k] as f64 / n as f64
                ));
            }
            for (k, g) in r.graphs.iter().enumerate() {
                let factor = match k {
                    1 => r.scaled_initial_factor,
                    3 => r.normalized_random_factor,
                    _ => 1.0,
                };
                let clipped = g
                    .edges
                    .iter()
                    .filter(|e| e.weight.abs() >= WEIGHT_LIMIT - 1e-12)
                    .count();
                diagnostics.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{factor}\n",
                    names[k],
                    mean_abs(g),
                    clipped,
                    g.edges.len()
                ));
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
                    "{seed},{delay},{i},{},{},{:.17},{:.17},{:.17},{:.17},{:.17}\n",
                    e.from,
                    e.to,
                    e.weight,
                    r.graphs[1].edges[i].weight,
                    r.graphs[2].edges[i].weight,
                    r.graphs[3].edges[i].weight,
                    r.graphs[4].edges[i].weight
                ));
            }
        }
    }
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("diagnostics.csv"), diagnostics)?;
    fs::write(dir.join("impulse.csv"), probes)?;
    fs::write(dir.join("weights.csv"), weights)?;
    let mut summary=String::from("# Trained graph strength intervention\n\n| Delay | Initial | Initial strength-matched | Random trained | Random normalized | Task trained | Online random | Shift |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (i, d) in c.delays.iter().enumerate() {
        summary.push_str(&format!(
            "| {d} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            totals[i][0] as f64 / counts[i] as f64,
            totals[i][1] as f64 / counts[i] as f64,
            totals[i][2] as f64 / counts[i] as f64,
            totals[i][3] as f64 / counts[i] as f64,
            totals[i][4] as f64 / counts[i] as f64,
            totals[i][5] as f64 / counts[i] as f64,
            totals[i][6] as f64 / counts[i] as f64
        ));
    }
    fs::write(dir.join("summary.md"), summary)
}

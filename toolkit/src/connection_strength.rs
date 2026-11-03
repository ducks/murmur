//! Fixed-graph connection-strength intervention and finite-horizon impulse probe.
use crate::{
    temporal_plasticity::{self, WEIGHT_LIMIT},
    temporal_stream::{self, StreamConfig},
};
use murmur::{data::Rng, graph::Graph, learning::Readout};
use std::{collections::HashSet, fs, io, path::Path};
pub const SCALES: [f64; 5] = [0.5, 1.0, 1.5, 2.0, 3.0];
pub const PROBES: usize = 64;
pub const HORIZON: usize = 16;

pub fn scaled_graph(c: &StreamConfig, seed: u64, scale: f64) -> Graph {
    let mut g = Graph::new(c.units, c.degree, seed);
    for e in &mut g.edges {
        e.weight = (e.weight * scale).clamp(-WEIGHT_LIMIT, WEIGHT_LIMIT);
    }
    g
}
/// Starts with states reset: feed 0 to one graph and 1 to the other, then feed
/// both identical pseudorandom symbols. Return mean RMS state distance by lag.
pub fn impulse(g: &Graph, seed: u64, probes: usize, horizon: usize) -> Vec<f64> {
    assert!(probes > 0);
    let mut values = vec![0.0; horizon + 1];
    let mut rng = Rng(seed);
    for _ in 0..probes {
        let mut a = g.clone();
        let mut b = g.clone();
        a.reset();
        b.reset();
        a.step(0);
        b.step(1);
        for (lag, value) in values.iter_mut().enumerate() {
            if lag > 0 {
                let x = (rng.next_u64() & 1) as usize;
                a.step(x);
                b.step(x);
            }
            *value += (a
                .state
                .iter()
                .zip(&b.state)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f64>()
                / a.state.len() as f64)
                .sqrt();
        }
    }
    for x in &mut values {
        *x /= probes as f64;
    }
    values
}
fn predict(c: &StreamConfig, seed: u64, delay: usize, mut graph: Graph) -> Vec<usize> {
    let train = temporal_stream::stream(seed.wrapping_add(31000), c.train_tokens);
    let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
    let mut reader = Readout::new(c.units, c.learning_rate);
    for (t, &symbol) in train.iter().enumerate() {
        graph.step(symbol);
        if t >= delay {
            reader.learn(&graph.state, train[t - delay]);
        }
    }
    graph.reset();
    test.iter()
        .enumerate()
        .filter_map(|(t, &symbol)| {
            graph.step(symbol);
            (t >= delay).then(|| reader.predict(&graph.state))
        })
        .collect()
}
pub struct Run {
    pub baseline: temporal_plasticity::Run,
    pub scaled: Vec<Vec<usize>>,
}
pub fn run(c: &StreamConfig, seed: u64, delay: usize) -> Run {
    let baseline = temporal_plasticity::run_delay(c, seed, delay);
    let scaled: Vec<_> = SCALES
        .iter()
        .map(|&s| predict(c, seed, delay, scaled_graph(c, seed, s)))
        .collect();
    assert!(scaled[1]
        .iter()
        .zip(&baseline.rows)
        .all(|(a, b)| *a == b.baseline.graph));
    Run { baseline, scaled }
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let mut seen = HashSet::new();
    for s in &c.seeds {
        for o in [0, 31000, 32000, 33000, 34000, 35000] {
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
    fs::write(dir.join("protocol.txt"),"task=delayed bit recall; readout trained online, reset state, frozen evaluation\nscales=0.5,1,1.5,2,3 applied to initial directed edge weights with clamp [-0.5,0.5]; topology and input weights unchanged\nimpulse=64 reset paired trajectories: first input differs 0 vs 1, followed by identical random inputs; RMS state distance at lags 0..16\nseed_roles=graph:s train:s+31000 test:s+32000 feedback:s+33000 random_target:s+34000 probe:s+35000 wrapping u64\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let mut predictions=String::from("seed,delay,index,input,target,scale_0_5,scale_1,scale_1_5,scale_2,scale_3,online_task,online_unrelated,shift\n");
    let mut metrics = String::from("seed,delay,model,correct,trials,accuracy\n");
    let mut probes = String::from("seed,scale,lag,mean_rms_distance,clipped_edges,total_edges\n");
    let names = [
        "scale_0_5",
        "scale_1",
        "scale_1_5",
        "scale_2",
        "scale_3",
        "online_task",
        "online_unrelated",
        "shift",
    ];
    let mut totals = vec![[0usize; 8]; c.delays.len()];
    let mut counts = vec![0usize; c.delays.len()];
    for &seed in &c.seeds {
        for &scale in &SCALES {
            let g = scaled_graph(c, seed, scale);
            let clipped = g
                .edges
                .iter()
                .filter(|e| e.weight.abs() >= WEIGHT_LIMIT - 1e-12)
                .count();
            for (lag, x) in impulse(&g, seed.wrapping_add(35000), PROBES, HORIZON)
                .iter()
                .enumerate()
            {
                probes.push_str(&format!(
                    "{seed},{scale},{lag},{x},{clipped},{}\n",
                    g.edges.len()
                ));
            }
        }
        for (di, &delay) in c.delays.iter().enumerate() {
            let r = run(c, seed, delay);
            let mut correct = [0usize; 8];
            for (i, row) in r.baseline.rows.iter().enumerate() {
                let b = &row.baseline;
                let v = [
                    r.scaled[0][i],
                    r.scaled[1][i],
                    r.scaled[2][i],
                    r.scaled[3][i],
                    r.scaled[4][i],
                    row.adaptive,
                    row.random_feedback,
                    b.shift,
                ];
                for k in 0..8 {
                    correct[k] += usize::from(v[k] == b.target);
                }
                predictions.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{},{},{},{}\n",
                    b.index, b.input, b.target, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]
                ));
            }
            let n = r.baseline.rows.len();
            counts[di] += n;
            for k in 0..8 {
                totals[di][k] += correct[k];
                metrics.push_str(&format!(
                    "{seed},{delay},{},{},{n},{}\n",
                    names[k],
                    correct[k],
                    correct[k] as f64 / n as f64
                ));
            }
        }
    }
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("impulse.csv"), probes)?;
    let mut summary=String::from("# Fixed connection-strength intervention\n\n| Delay | 0.5× | 1× | 1.5× | 2× | 3× | Online task | Online unrelated | Shift |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (i, d) in c.delays.iter().enumerate() {
        summary.push_str(&format!(
            "| {d} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            totals[i][0] as f64 / counts[i] as f64,
            totals[i][1] as f64 / counts[i] as f64,
            totals[i][2] as f64 / counts[i] as f64,
            totals[i][3] as f64 / counts[i] as f64,
            totals[i][4] as f64 / counts[i] as f64,
            totals[i][5] as f64 / counts[i] as f64,
            totals[i][6] as f64 / counts[i] as f64,
            totals[i][7] as f64 / counts[i] as f64
        ));
    }
    fs::write(dir.join("summary.md"), summary)
}

//! Isolate the reader-dependent component of pseudo-target feedback.
use crate::{
    centered_feedback,
    connection_strength::{impulse, HORIZON, PROBES},
    temporal_plasticity::{PlasticModel, WEIGHT_LIMIT},
    temporal_stream::{self, StreamConfig},
    trained_strength,
};
use murmur::graph::Graph;
use std::{collections::HashSet, fs, io, path::Path};

pub struct Checkpoint {
    pub step: usize,
    pub scored: usize,
    pub mean_probability: f64,
    pub mean_signal: f64,
    pub mean_abs_signal: f64,
    pub mean_abs_drift: f64,
    pub mean_abs_weight: f64,
    pub clipped: usize,
}
pub struct Run {
    pub comparison: centered_feedback::Run,
    pub reader_graph: Graph,
    pub reader_frozen: Vec<usize>,
    pub reader_online: Vec<usize>,
    pub checkpoints: Vec<Checkpoint>,
}
pub fn run(c: &StreamConfig, seed: u64, delay: usize) -> Run {
    let comparison = centered_feedback::run(c, seed, delay);
    let train = temporal_stream::stream(seed.wrapping_add(31000), c.train_tokens);
    let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
    let mut m = PlasticModel::new(c, seed);
    let initial: Vec<_> = m.graph.edges.iter().map(|e| e.weight).collect();
    let (mut n, mut p_sum, mut signal_sum, mut abs_sum) = (0usize, 0.0, 0.0, 0.0);
    let mut checkpoints = Vec::new();
    for (t, &x) in train.iter().enumerate() {
        m.advance(x, true);
        if t >= delay {
            let p = m.class_one_probability();
            let signal = 0.5 - p;
            n += 1;
            p_sum += p;
            signal_sum += signal;
            abs_sum += signal.abs();
            m.learn_with_error(train[t - delay], signal);
        }
        if (t + 1) % 1000 == 0 || t + 1 == train.len() {
            let denom = n.max(1) as f64;
            checkpoints.push(Checkpoint {
                step: t + 1,
                scored: n,
                mean_probability: p_sum / denom,
                mean_signal: signal_sum / denom,
                mean_abs_signal: abs_sum / denom,
                mean_abs_drift: m
                    .graph
                    .edges
                    .iter()
                    .zip(&initial)
                    .map(|(e, w)| (e.weight - w).abs())
                    .sum::<f64>()
                    / initial.len() as f64,
                mean_abs_weight: trained_strength::mean_abs(&m.graph),
                clipped: m
                    .graph
                    .edges
                    .iter()
                    .filter(|e| e.weight.abs() >= WEIGHT_LIMIT - 1e-12)
                    .count(),
            });
        }
    }
    let reader_graph = m.graph.clone();
    let reader_frozen = trained_strength::predict(c, seed, delay, reader_graph.clone());
    m.reset_state();
    let mut reader_online = Vec::new();
    for (t, &x) in test.iter().enumerate() {
        let p = m.advance(x, false);
        if t >= delay {
            reader_online.push(p);
        }
    }
    Run {
        comparison,
        reader_graph,
        reader_frozen,
        reader_online,
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
    fs::write(dir.join("protocol.txt"),"task=delayed bit recall\nedge_scalars=combined:random_bit-probability; centered:random_bit-0.5; reader_only:0.5-probability; task:true_bit-probability\nmodels use own states and true-label readers; same graph input feedback signs and random bit where used\nfrozen_readers=train from scratch on same training stream, then reset state and freeze weights\nseed_roles=graph:s train:s+31000 test:s+32000 feedback:s+33000 random_bit:s+34000 probe:s+35000 wrapping u64\nprobe=64 zero-state trajectory pairs, first bit differs then same input, RMS distance lags 0..16\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let names = [
        "initial_frozen",
        "combined_frozen",
        "centered_frozen",
        "reader_frozen",
        "task_frozen",
        "combined_online",
        "reader_online",
        "shift",
    ];
    let mut metrics = String::from("seed,delay,model,correct,trials,accuracy\n");
    let mut pred=String::from("seed,delay,index,input,target,initial_frozen,combined_frozen,centered_frozen,reader_frozen,task_frozen,combined_online,reader_online,shift\n");
    let mut diag=String::from("seed,delay,step,scored,mean_probability,mean_signal,mean_abs_signal,mean_abs_drift,mean_abs_weight,clipped\n");
    let mut compare=String::from("seed,delay,step,scored,combined_mean_abs_drift,centered_mean_abs_drift,combined_mean_abs_weight,centered_mean_abs_weight,combined_clipped,centered_clipped\n");
    let mut probes = String::from("seed,delay,model,lag,mean_rms_distance\n");
    let mut weights =
        String::from("seed,delay,edge_id,from,to,initial,combined,centered,reader,task\n");
    let mut totals = vec![[0usize; 8]; c.delays.len()];
    let mut counts = vec![0usize; c.delays.len()];
    for &seed in &c.seeds {
        for (di, &delay) in c.delays.iter().enumerate() {
            let r = run(c, seed, delay);
            let mut correct = [0usize; 8];
            for (i, row) in r.comparison.baseline.rows.iter().enumerate() {
                let b = &row.baseline;
                let v = [
                    r.comparison.frozen[0][i],
                    r.comparison.frozen[1][i],
                    r.comparison.frozen[2][i],
                    r.reader_frozen[i],
                    r.comparison.frozen[3][i],
                    row.random_feedback,
                    r.reader_online[i],
                    b.shift,
                ];
                for k in 0..8 {
                    correct[k] += usize::from(v[k] == b.target);
                }
                pred.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{},{},{},{}\n",
                    b.index, b.input, b.target, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]
                ));
            }
            let n = r.reader_online.len();
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
            for x in &r.checkpoints {
                diag.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{}\n",
                    x.step,
                    x.scored,
                    x.mean_probability,
                    x.mean_signal,
                    x.mean_abs_signal,
                    x.mean_abs_drift,
                    x.mean_abs_weight,
                    x.clipped
                ));
            }
            for x in &r.comparison.checkpoints {
                compare.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{}\n",
                    x.step,
                    x.scored,
                    x.pseudo_drift,
                    x.centered_drift,
                    x.pseudo_mean_abs,
                    x.centered_mean_abs,
                    x.pseudo_clipped,
                    x.centered_clipped
                ));
            }
            let graphs = [
                &r.comparison.graphs[0],
                &r.comparison.graphs[1],
                &r.comparison.graphs[2],
                &r.reader_graph,
                &r.comparison.graphs[3],
            ];
            for (k, g) in graphs.iter().enumerate() {
                for (lag, x) in impulse(g, seed.wrapping_add(35000), PROBES, HORIZON)
                    .iter()
                    .enumerate()
                {
                    probes.push_str(&format!("{seed},{delay},{},{lag},{x}\n", names[k]));
                }
            }
            for i in 0..graphs[0].edges.len() {
                let e = &graphs[0].edges[i];
                weights.push_str(&format!(
                    "{seed},{delay},{i},{},{},{:.17},{:.17},{:.17},{:.17},{:.17}\n",
                    e.from,
                    e.to,
                    e.weight,
                    graphs[1].edges[i].weight,
                    graphs[2].edges[i].weight,
                    graphs[3].edges[i].weight,
                    graphs[4].edges[i].weight
                ));
            }
        }
    }
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("predictions.csv"), pred)?;
    fs::write(dir.join("reader_diagnostics.csv"), diag)?;
    fs::write(dir.join("comparison_diagnostics.csv"), compare)?;
    fs::write(dir.join("impulse.csv"), probes)?;
    fs::write(dir.join("weights.csv"), weights)?;
    let mut summary=String::from("# Reader-dependent feedback component\n\n| Delay | Initial frozen | Combined frozen | Centered frozen | Reader-only frozen | Task frozen | Combined online | Reader-only online | Shift |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (i, d) in c.delays.iter().enumerate() {
        let n = counts[i] as f64;
        summary.push_str(&format!(
            "| {d} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            totals[i][0] as f64 / n,
            totals[i][1] as f64 / n,
            totals[i][2] as f64 / n,
            totals[i][3] as f64 / n,
            totals[i][4] as f64 / n,
            totals[i][5] as f64 / n,
            totals[i][6] as f64 / n,
            totals[i][7] as f64 / n
        ));
    }
    fs::write(dir.join("summary.md"), summary)
}

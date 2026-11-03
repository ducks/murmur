//! Compare learned edges with equally sized frozen random perturbations.
use crate::{
    temporal_plasticity::{self, WEIGHT_LIMIT},
    temporal_stream::{self, StreamConfig},
};
use murmur::{data::Rng, graph::Graph, learning::Readout};
use std::{collections::HashSet, fs, io, path::Path};

/// Preserve the absolute displacement of every individual edge, randomizing its
/// direction when both signs are feasible. Bounds occasionally force a sign.
pub fn matched_random(initial: &Graph, learned: &Graph, seed: u64) -> (Graph, usize) {
    assert_eq!(initial.edges.len(), learned.edges.len());
    let mut result = initial.clone();
    let mut rng = Rng(seed);
    let mut forced = 0;
    for ((out, base), trained) in result
        .edges
        .iter_mut()
        .zip(&initial.edges)
        .zip(&learned.edges)
    {
        assert_eq!((base.from, base.to), (trained.from, trained.to));
        let magnitude = (trained.weight - base.weight).abs();
        let sign = if rng.next_u64() & 1 == 0 { -1.0 } else { 1.0 };
        let mut candidate = base.weight + sign * magnitude;
        if candidate.abs() > WEIGHT_LIMIT + 1e-12 {
            candidate = base.weight - sign * magnitude;
            forced += 1;
        }
        assert!(candidate.abs() <= WEIGHT_LIMIT + 1e-12);
        out.weight = candidate;
    }
    (result, forced)
}

fn frozen_predictions(c: &StreamConfig, seed: u64, delay: usize, mut graph: Graph) -> Vec<usize> {
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
    pub rows: Vec<temporal_plasticity::Row>,
    pub learned_frozen: Vec<usize>,
    pub random_frozen: Vec<usize>,
    pub initial: Graph,
    pub learned: Graph,
    pub random: Graph,
    pub forced: usize,
}
pub fn run(c: &StreamConfig, seed: u64, delay: usize) -> Run {
    let base = temporal_plasticity::run_delay(c, seed, delay);
    let (random, forced) = matched_random(&base.initial, &base.adaptive, seed.wrapping_add(35000));
    let learned_frozen = frozen_predictions(c, seed, delay, base.adaptive.clone());
    let random_frozen = frozen_predictions(c, seed, delay, random.clone());
    Run {
        rows: base.rows,
        learned_frozen,
        random_frozen,
        initial: base.initial,
        learned: base.adaptive,
        random,
        forced,
    }
}

pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let mut seeds = HashSet::new();
    for s in &c.seeds {
        for offset in [0, 31000, 32000, 33000, 34000, 35000] {
            if !seeds.insert(s.wrapping_add(offset)) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "seed roles collide",
                ));
            }
        }
    }
    if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
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
    fs::write(dir.join("protocol.txt"),"task=delayed bit recall; reset state and freeze weights at evaluation\nmodels=fixed, online_task, online_unrelated, frozen_task, frozen_matched_random, shift\nfrozen_task_and_random_readout=trained from scratch on same training inputs after graph is fixed\nmatched_random=each edge uses absolute learned displacement with separately seeded random sign, flipped if necessary to respect bounds\nseed_roles=graph:s train:s+31000 test:s+32000 feedback:s+33000 unrelated:s+34000 signs:s+35000, wrapping u64\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let mut metrics = String::from("seed,delay,model,correct,trials,accuracy\n");
    let mut predictions=String::from("seed,delay,index,input,target,fixed,online_task,online_unrelated,frozen_task,frozen_random,shift\n");
    let mut weights = String::from("seed,delay,edge_id,from,to,initial,learned,matched_random\n");
    let mut diagnostics =
        String::from("seed,delay,mean_absolute_displacement,forced_signs,edges\n");
    let names = [
        "fixed",
        "online_task",
        "online_unrelated",
        "frozen_task",
        "frozen_random",
        "shift",
    ];
    let mut sums = vec![[0usize; 6]; c.delays.len()];
    let mut counts = vec![0usize; c.delays.len()];
    for &seed in &c.seeds {
        for (di, &delay) in c.delays.iter().enumerate() {
            let r = run(c, seed, delay);
            let mut correct = [0usize; 6];
            for (i, row) in r.rows.iter().enumerate() {
                let b = &row.baseline;
                let v = [
                    b.graph,
                    row.adaptive,
                    row.random_feedback,
                    r.learned_frozen[i],
                    r.random_frozen[i],
                    b.shift,
                ];
                for k in 0..6 {
                    correct[k] += usize::from(v[k] == b.target);
                }
                predictions.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{},{}\n",
                    b.index, b.input, b.target, v[0], v[1], v[2], v[3], v[4], v[5]
                ));
            }
            let n = r.rows.len();
            counts[di] += n;
            for k in 0..6 {
                sums[di][k] += correct[k];
                metrics.push_str(&format!(
                    "{seed},{delay},{},{},{n},{}\n",
                    names[k],
                    correct[k],
                    correct[k] as f64 / n as f64
                ));
            }
            let mut drift = 0.0;
            for (i, ((a, b), p)) in r
                .initial
                .edges
                .iter()
                .zip(&r.learned.edges)
                .zip(&r.random.edges)
                .enumerate()
            {
                drift += (a.weight - b.weight).abs();
                weights.push_str(&format!(
                    "{seed},{delay},{i},{},{},{:.17},{:.17},{:.17}\n",
                    a.from, a.to, a.weight, b.weight, p.weight
                ));
            }
            diagnostics.push_str(&format!(
                "{seed},{delay},{},{},{}\n",
                drift / r.initial.edges.len() as f64,
                r.forced,
                r.initial.edges.len()
            ));
        }
    }
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("weights.csv"), weights)?;
    fs::write(dir.join("diagnostics.csv"), diagnostics)?;
    let mut summary=String::from("# Matched perturbation\n\n| Delay | Fixed | Online task | Online unrelated | Frozen task | Frozen matched random | Shift |\n|---:|---:|---:|---:|---:|---:|---:|\n");
    for (i, delay) in c.delays.iter().enumerate() {
        summary.push_str(&format!(
            "| {delay} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            sums[i][0] as f64 / counts[i] as f64,
            sums[i][1] as f64 / counts[i] as f64,
            sums[i][2] as f64 / counts[i] as f64,
            sums[i][3] as f64 / counts[i] as f64,
            sums[i][4] as f64 / counts[i] as f64,
            sums[i][5] as f64 / counts[i] as f64
        ));
    }
    fs::write(dir.join("summary.md"), summary)
}

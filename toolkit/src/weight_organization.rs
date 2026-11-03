//! Interventions preserving trained weight distributions while changing placement.
use crate::{
    connection_strength::{impulse, HORIZON, PROBES},
    temporal_plasticity,
    temporal_stream::StreamConfig,
    trained_strength,
};
use murmur::{data::Rng, graph::Graph};
use std::{
    collections::{BTreeMap, HashSet},
    fs, io,
    path::Path,
};
pub const REPLICATES: usize = 3;
/// Permute existing edge weights either across all edges or within each
/// destination's incoming edges. Wiring and exact weight multiset are retained.
pub fn shuffle(g: &Graph, seed: u64, within_destination: bool) -> Graph {
    let mut out = g.clone();
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, e) in g.edges.iter().enumerate() {
        groups
            .entry(if within_destination { e.to } else { 0 })
            .or_default()
            .push(i);
    }
    let mut rng = Rng(seed);
    for ids in groups.values() {
        let mut values: Vec<f64> = ids.iter().map(|&i| g.edges[i].weight).collect();
        for j in (1..values.len()).rev() {
            let k = (rng.next_u64() % (j as u64 + 1)) as usize;
            values.swap(j, k);
        }
        for (&i, w) in ids.iter().zip(values) {
            out.edges[i].weight = w;
        }
    }
    out
}
pub struct Run {
    pub baseline: temporal_plasticity::Run,
    pub graphs: Vec<Graph>,
    pub predictions: Vec<Vec<usize>>,
}
pub fn run(c: &StreamConfig, seed: u64, delay: usize) -> Run {
    let baseline = temporal_plasticity::run_delay(c, seed, delay);
    let random = &baseline.random_feedback;
    let normalized =
        trained_strength::match_strength(random, trained_strength::mean_abs(&baseline.initial))
            .expect("strength target reachable")
            .0;
    let mut graphs = vec![baseline.initial.clone(), random.clone(), normalized];
    for rep in 0..REPLICATES {
        graphs.push(shuffle(
            random,
            seed.wrapping_add(36000 + 2000 * rep as u64),
            false,
        ));
    }
    for rep in 0..REPLICATES {
        graphs.push(shuffle(
            random,
            seed.wrapping_add(42000 + 2000 * rep as u64),
            true,
        ));
    }
    let predictions = graphs
        .iter()
        .map(|g| trained_strength::predict(c, seed, delay, g.clone()))
        .collect();
    Run {
        baseline,
        graphs,
        predictions,
    }
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    let mut seeds = HashSet::new();
    for s in &c.seeds {
        for o in [
            0, 31000, 32000, 33000, 34000, 35000, 36000, 38000, 40000, 42000, 44000, 46000,
        ] {
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
    fs::write(dir.join("protocol.txt"),"task=delayed bit recall\nmodels=initial, random_feedback_trained, random_normalized_to_initial_mean, three globally shuffled trained weights, three shuffled within each destination\npermutation=Fisher-Yates with modulo mapping; exact signed multiset preserved, wiring fixed\nreadouts=fresh online training on same train stream after graph frozen; independent test stream, state reset\nimpulse=64 reset pairs, first bit differs, common continuation, RMS distance lags 0..16\nseeds=graph:s train:s+31000 test:s+32000 feedback:s+33000 pseudo:s+34000 probe:s+35000 global_shuffle:s+36000,38000,40000 row_shuffle:s+42000,44000,46000; wrapping u64\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let names = [
        "initial",
        "trained",
        "normalized",
        "global_0",
        "global_1",
        "global_2",
        "destination_0",
        "destination_1",
        "destination_2",
    ];
    let mut metrics = String::from("seed,delay,model,correct,trials,accuracy\n");
    let mut predictions=String::from("seed,delay,index,input,target,initial,trained,normalized,global_0,global_1,global_2,destination_0,destination_1,destination_2\n");
    let mut probes = String::from("seed,delay,model,lag,mean_rms_distance\n");
    let mut weights=String::from("seed,delay,edge_id,from,to,initial,trained,normalized,global_0,global_1,global_2,destination_0,destination_1,destination_2\n");
    let mut totals = vec![[0usize; 9]; c.delays.len()];
    let mut counts = vec![0usize; c.delays.len()];
    for &seed in &c.seeds {
        for (di, &delay) in c.delays.iter().enumerate() {
            let r = run(c, seed, delay);
            let mut correct = [0usize; 9];
            for (i, row) in r.baseline.rows.iter().enumerate() {
                let b = &row.baseline;
                let v: Vec<_> = r.predictions.iter().map(|p| p[i]).collect();
                for k in 0..9 {
                    correct[k] += usize::from(v[k] == b.target);
                }
                predictions.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    b.index,
                    b.input,
                    b.target,
                    v[0],
                    v[1],
                    v[2],
                    v[3],
                    v[4],
                    v[5],
                    v[6],
                    v[7],
                    v[8]
                ));
            }
            let n = r.baseline.rows.len();
            counts[di] += n;
            for k in 0..9 {
                totals[di][k] += correct[k];
                metrics.push_str(&format!(
                    "{seed},{delay},{},{},{n},{}\n",
                    names[k],
                    correct[k],
                    correct[k] as f64 / n as f64
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
                weights.push_str(&format!("{seed},{delay},{i},{},{},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17},{:.17}\n",e.from,e.to,e.weight,r.graphs[1].edges[i].weight,r.graphs[2].edges[i].weight,r.graphs[3].edges[i].weight,r.graphs[4].edges[i].weight,r.graphs[5].edges[i].weight,r.graphs[6].edges[i].weight,r.graphs[7].edges[i].weight,r.graphs[8].edges[i].weight));
            }
        }
    }
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("impulse.csv"), probes)?;
    fs::write(dir.join("weights.csv"), weights)?;
    let mut summary=String::from("# Weight placement intervention\n\n| Delay | Initial | Trained | Normalized | Global shuffled (mean) | Destination shuffled (mean) |\n|---:|---:|---:|---:|---:|---:|\n");
    for (i, d) in c.delays.iter().enumerate() {
        let n = counts[i] as f64;
        summary.push_str(&format!(
            "| {d} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            totals[i][0] as f64 / n,
            totals[i][1] as f64 / n,
            totals[i][2] as f64 / n,
            (totals[i][3] + totals[i][4] + totals[i][5]) as f64 / (3.0 * n),
            (totals[i][6] + totals[i][7] + totals[i][8]) as f64 / (3.0 * n)
        ));
    }
    fs::write(dir.join("summary.md"), summary)
}

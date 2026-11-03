//! Transfer reader-shaped graph dynamics across delay tasks and data streams.
use crate::{
    connection_strength::scaled_graph,
    temporal_plasticity::PlasticModel,
    temporal_stream::{self, ShiftRegister, StreamConfig},
    trained_strength,
};
use murmur::{graph::Graph, learning::Readout};
use std::{collections::HashSet, fs, io, path::Path};
pub const SOURCE_DELAYS: [usize; 2] = [4, 8];

/// Train the reader-only feedback graph on a separate pretraining stream. In
/// the control the reader receives separately seeded random labels instead.
pub fn pretrain(c: &StreamConfig, seed: u64, source_delay: usize, random_labels: bool) -> Graph {
    assert!(source_delay < c.train_tokens);
    let inputs = temporal_stream::stream(seed.wrapping_add(31000), c.train_tokens);
    let labels = temporal_stream::stream(seed.wrapping_add(34000), c.train_tokens);
    let mut m = PlasticModel::new(c, seed);
    for (t, &symbol) in inputs.iter().enumerate() {
        m.advance(symbol, true);
        if t >= source_delay {
            let target = if random_labels {
                labels[t]
            } else {
                inputs[t - source_delay]
            };
            let signal = 0.5 - m.class_one_probability();
            m.learn_with_error(target, signal);
        }
    }
    m.graph.reset();
    m.graph
}
/// Reader training and evaluation use separate streams from graph pretraining
/// and one another. Every condition receives an independently trained reader.
pub fn predict(c: &StreamConfig, seed: u64, target_delay: usize, mut graph: Graph) -> Vec<usize> {
    let inputs = temporal_stream::stream(seed.wrapping_add(37000), c.train_tokens);
    let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
    let mut r = Readout::new(c.units, c.learning_rate);
    graph.reset();
    for (t, &x) in inputs.iter().enumerate() {
        graph.step(x);
        if t >= target_delay {
            r.learn(&graph.state, inputs[t - target_delay]);
        }
    }
    graph.reset();
    test.iter()
        .enumerate()
        .filter_map(|(t, &x)| {
            graph.step(x);
            (t >= target_delay).then(|| r.predict(&graph.state))
        })
        .collect()
}
pub struct Graphs {
    pub graphs: Vec<Graph>,
    pub matched_factor: f64,
}
pub fn graphs(c: &StreamConfig, seed: u64, source: usize) -> Graphs {
    let initial = Graph::new(c.units, c.degree, seed);
    let supervised = pretrain(c, seed, source, false);
    let random = pretrain(c, seed, source, true);
    let (matched, factor) =
        trained_strength::match_strength(&initial, trained_strength::mean_abs(&supervised))
            .expect("target mean reachable");
    let fixed2 = scaled_graph(c, seed, 2.0);
    Graphs {
        graphs: vec![initial, supervised, random, matched, fixed2],
        matched_factor: factor,
    }
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    if c.train_tokens <= 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "pretraining stream too short for source delays",
        ));
    }
    let mut seeds = HashSet::new();
    for s in &c.seeds {
        for o in [0, 31000, 32000, 33000, 34000, 37000] {
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
    fs::write(dir.join("protocol.txt"),"task=pretrain graph reader-only feedback on source delay 4 or 8; train fresh reader on target delay using separate stream; reset state and score on separate test stream\nmodels=initial, true_label_pretrained, random_label_pretrained, initial_scaled_to_true_mean_absolute_strength, initial_scaled_2x, causal_shift\nrandom_label_control=reader receives separate pseudorandom bits during graph pretraining; edges use reader-only signal\nseeds=graph:s pretrain_input:s+31000 evaluation:s+32000 feedback:s+33000 random_labels:s+34000 reader_train:s+37000; wrapping u64\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let names = [
        "initial",
        "true_pretrained",
        "random_pretrained",
        "strength_matched",
        "fixed_2x",
        "shift",
    ];
    let mut metrics =
        String::from("seed,source_delay,target_delay,model,correct,trials,accuracy\n");
    let mut predictions=String::from("seed,source_delay,target_delay,index,input,target,initial,true_pretrained,random_pretrained,strength_matched,fixed_2x,shift\n");
    let mut diagnostics=String::from("seed,source_delay,model,mean_abs_weight,mean_abs_drift,clipped_edges,total_edges,matched_scale\n");
    let mut weights=String::from("seed,source_delay,edge_id,from,to,initial,true_pretrained,random_pretrained,strength_matched,fixed_2x\n");
    let mut sum = vec![[[0usize; 6]; 2]; c.delays.len()];
    let mut count = vec![[0usize; 2]; c.delays.len()];
    for &seed in &c.seeds {
        for (si, &source) in SOURCE_DELAYS.iter().enumerate() {
            let g = graphs(c, seed, source);
            for (k, graph) in g.graphs.iter().enumerate() {
                let drift = graph
                    .edges
                    .iter()
                    .zip(&g.graphs[0].edges)
                    .map(|(a, b)| (a.weight - b.weight).abs())
                    .sum::<f64>()
                    / graph.edges.len() as f64;
                let clipped = graph
                    .edges
                    .iter()
                    .filter(|e| e.weight.abs() >= 0.5 - 1e-12)
                    .count();
                diagnostics.push_str(&format!(
                    "{seed},{source},{},{},{},{},{},{}\n",
                    names[k],
                    trained_strength::mean_abs(graph),
                    drift,
                    clipped,
                    graph.edges.len(),
                    if k == 3 { g.matched_factor } else { 1.0 }
                ));
            }
            for i in 0..g.graphs[0].edges.len() {
                let e = &g.graphs[0].edges[i];
                weights.push_str(&format!(
                    "{seed},{source},{i},{},{},{:.17},{:.17},{:.17},{:.17},{:.17}\n",
                    e.from,
                    e.to,
                    e.weight,
                    g.graphs[1].edges[i].weight,
                    g.graphs[2].edges[i].weight,
                    g.graphs[3].edges[i].weight,
                    g.graphs[4].edges[i].weight
                ));
            }
            for (di, &target) in c.delays.iter().enumerate() {
                let all: Vec<_> = g
                    .graphs
                    .iter()
                    .map(|graph| predict(c, seed, target, graph.clone()))
                    .collect();
                let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
                let mut shift = ShiftRegister::new(target);
                let mut correct = [0usize; 6];
                let mut index = 0;
                for (t, &x) in test.iter().enumerate() {
                    let sh = shift.observe(x);
                    if t >= target {
                        let y = test[t - target];
                        let v = [
                            all[0][index],
                            all[1][index],
                            all[2][index],
                            all[3][index],
                            all[4][index],
                            sh.expect("shift warmed"),
                        ];
                        for k in 0..6 {
                            correct[k] += usize::from(v[k] == y);
                        }
                        predictions.push_str(&format!(
                            "{seed},{source},{target},{t},{x},{y},{},{},{},{},{},{}\n",
                            v[0], v[1], v[2], v[3], v[4], v[5]
                        ));
                        index += 1;
                    }
                }
                count[di][si] += index;
                for k in 0..6 {
                    sum[di][si][k] += correct[k];
                    metrics.push_str(&format!(
                        "{seed},{source},{target},{},{},{index},{}\n",
                        names[k],
                        correct[k],
                        correct[k] as f64 / index as f64
                    ));
                }
            }
        }
    }
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("diagnostics.csv"), diagnostics)?;
    fs::write(dir.join("weights.csv"), weights)?;
    let mut summary=String::from("# Reader-trained graph transfer\n\n| Source delay | Target delay | Initial | True-label graph | Random-label graph | Strength-matched | Fixed 2x | Shift |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (si, s) in SOURCE_DELAYS.iter().enumerate() {
        for (di, d) in c.delays.iter().enumerate() {
            let n = count[di][si] as f64;
            let a = sum[di][si];
            summary.push_str(&format!(
                "| {s} | {d} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
                a[0] as f64 / n,
                a[1] as f64 / n,
                a[2] as f64 / n,
                a[3] as f64 / n,
                a[4] as f64 / n,
                a[5] as f64 / n
            ));
        }
    }
    fs::write(dir.join("summary.md"), summary)
}

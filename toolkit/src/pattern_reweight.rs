//! Follow-up: repeat positive-label reader updates and compare direct history.
use crate::{
    pattern_recognition::{self, Confusion},
    reader_transfer,
    temporal_stream::{self, StreamConfig},
};
use murmur::{graph::Graph, learning::Readout};
use std::{collections::HashSet, fs, io, path::Path};
pub const POSITIVE_UPDATES: usize = 7;
pub fn train_and_predict(
    c: &StreamConfig,
    seed: u64,
    graph: Option<Graph>,
    positive_updates: usize,
) -> Vec<usize> {
    assert!(positive_updates > 0);
    let train = temporal_stream::stream(seed.wrapping_add(37000), c.train_tokens);
    let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
    let mut graph = graph;
    let mut reader = Readout::new(if graph.is_some() { c.units } else { 3 }, c.learning_rate);
    if let Some(g) = &mut graph {
        g.reset();
    }
    for (t, &x) in train.iter().enumerate() {
        if let Some(g) = &mut graph {
            g.step(x);
        }
        if t >= 2 {
            let y = pattern_recognition::label(&train, t);
            let raw = [train[t - 2] as f64, train[t - 1] as f64, x as f64];
            let state = graph
                .as_ref()
                .map_or(raw.as_slice(), |g| g.state.as_slice());
            for _ in 0..if y == 1 { positive_updates } else { 1 } {
                reader.learn(state, y);
            }
        }
    }
    if let Some(g) = &mut graph {
        g.reset();
    }
    let mut result = Vec::new();
    for (t, &x) in test.iter().enumerate() {
        if let Some(g) = &mut graph {
            g.step(x);
        }
        if t >= 2 {
            let raw = [test[t - 2] as f64, test[t - 1] as f64, x as f64];
            let state = graph
                .as_ref()
                .map_or(raw.as_slice(), |g| g.state.as_slice());
            result.push(reader.predict(state));
        }
    }
    result
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    if c.train_tokens <= 4 || c.test_tokens <= 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "streams too short",
        ));
    }
    let mut used = HashSet::new();
    for s in &c.seeds {
        for o in [0, 31000, 32000, 33000, 34000, 37000] {
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
fn fmt(x: Option<f64>) -> String {
    x.map(|v| v.to_string()).unwrap_or_default()
}
fn write_outputs(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),"task=detect current 101, first two steps unscored\nmodels=original and true-pretrained graph, trained with one or seven sequential readout updates per positive window; direct last-3-bits logistic reader with same two schedules; always-negative and exact\nreader=separate training stream, frozen test; positive replays use sequential updates, not exact weighted gradients\ngraph pretraining=reader-only feedback on delay4; additional training pass\nseed_roles=graph:s pretrain:s+31000 test:s+32000 feedback:s+33000 random_labels:s+34000 reader_train:s+37000 wrapping u64\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let names = [
        "initial_one",
        "pretrained_one",
        "initial_repeat7",
        "pretrained_repeat7",
        "history_one",
        "history_repeat7",
        "always_negative",
        "exact_pattern",
    ];
    let mut metrics = String::from(
        "seed,model,tp,tn,fp,fn,trials,accuracy,recall,specificity,balanced_accuracy,f1\n",
    );
    let mut predictions=String::from("seed,index,input,target,initial_one,pretrained_one,initial_repeat7,pretrained_repeat7,history_one,history_repeat7,always_negative,exact_pattern\n");
    let mut totals = [Confusion::default(); 8];
    for &seed in &c.seeds {
        let graphs = reader_transfer::graphs(c, seed, 4);
        let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
        let preds = [
            train_and_predict(c, seed, Some(graphs.graphs[0].clone()), 1),
            train_and_predict(c, seed, Some(graphs.graphs[1].clone()), 1),
            train_and_predict(c, seed, Some(graphs.graphs[0].clone()), POSITIVE_UPDATES),
            train_and_predict(c, seed, Some(graphs.graphs[1].clone()), POSITIVE_UPDATES),
            train_and_predict(c, seed, None, 1),
            train_and_predict(c, seed, None, POSITIVE_UPDATES),
        ];
        let mut counts = [Confusion::default(); 8];
        for t in 2..test.len() {
            let y = pattern_recognition::label(&test, t);
            let i = t - 2;
            let v = [
                preds[0][i],
                preds[1][i],
                preds[2][i],
                preds[3][i],
                preds[4][i],
                preds[5][i],
                0,
                y,
            ];
            for k in 0..8 {
                counts[k].add(y, v[k]);
                totals[k].add(y, v[k]);
            }
            predictions.push_str(&format!(
                "{seed},{t},{},{y},{},{},{},{},{},{},{},{}\n",
                test[t], v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]
            ));
        }
        for (k, a) in counts.iter().enumerate() {
            metrics.push_str(&format!(
                "{seed},{},{},{},{},{},{},{},{},{},{},{}\n",
                names[k],
                a.tp,
                a.tn,
                a.fp,
                a.r#fn,
                a.n(),
                a.accuracy(),
                fmt(a.recall()),
                fmt(a.specificity()),
                fmt(a.balanced_accuracy()),
                fmt(a.f1())
            ));
        }
    }
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("predictions.csv"), predictions)?;
    let mut summary=String::from("# Pattern reweighting follow-up\n\n| Model | Accuracy | Recall | Specificity | Balanced accuracy | F1 |\n|---|---:|---:|---:|---:|---:|\n");
    for (k, a) in totals.iter().enumerate() {
        summary.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            names[k],
            a.accuracy(),
            a.recall().unwrap_or(0.0),
            a.specificity().unwrap_or(0.0),
            a.balanced_accuracy().unwrap_or(0.0),
            a.f1().unwrap_or(0.0)
        ));
    }
    fs::write(dir.join("summary.md"), summary)
}

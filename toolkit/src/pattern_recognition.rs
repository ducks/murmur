//! Transfer a reader-shaped graph to online detection of the binary pattern 101.
use crate::{
    reader_transfer,
    temporal_stream::{self, StreamConfig},
};
use murmur::{graph::Graph, learning::Readout};
use std::{collections::HashSet, fs, io, path::Path};
pub const SOURCE_DELAY: usize = 4;
/// At time t, decide whether the three already observed bits are 1,0,1.
pub fn label(stream: &[usize], t: usize) -> usize {
    assert!(t >= 2 && t < stream.len());
    usize::from(stream[t - 2] == 1 && stream[t - 1] == 0 && stream[t] == 1)
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Confusion {
    pub tp: usize,
    pub tn: usize,
    pub fp: usize,
    pub r#fn: usize,
}
impl Confusion {
    pub fn add(&mut self, target: usize, prediction: usize) {
        match (target, prediction) {
            (1, 1) => self.tp += 1,
            (0, 0) => self.tn += 1,
            (0, 1) => self.fp += 1,
            (1, 0) => self.r#fn += 1,
            _ => panic!("binary labels only"),
        }
    }
    pub fn n(&self) -> usize {
        self.tp + self.tn + self.fp + self.r#fn
    }
    pub fn accuracy(&self) -> f64 {
        (self.tp + self.tn) as f64 / self.n() as f64
    }
    pub fn recall(&self) -> Option<f64> {
        let n = self.tp + self.r#fn;
        (n > 0).then(|| self.tp as f64 / n as f64)
    }
    pub fn specificity(&self) -> Option<f64> {
        let n = self.tn + self.fp;
        (n > 0).then(|| self.tn as f64 / n as f64)
    }
    pub fn balanced_accuracy(&self) -> Option<f64> {
        Some((self.recall()? + self.specificity()?) / 2.0)
    }
    pub fn f1(&self) -> Option<f64> {
        let n = 2 * self.tp + self.fp + self.r#fn;
        (n > 0).then(|| 2.0 * self.tp as f64 / n as f64)
    }
}
/// Train the reader with 101 labels on a separate stream, keeping the graph
/// fixed. Evaluation starts with a cleared state and a third independent stream.
pub fn predict(c: &StreamConfig, seed: u64, mut graph: Graph) -> Vec<usize> {
    let train = temporal_stream::stream(seed.wrapping_add(37000), c.train_tokens);
    let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
    let mut readout = Readout::new(c.units, c.learning_rate);
    graph.reset();
    for (t, &symbol) in train.iter().enumerate() {
        graph.step(symbol);
        if t >= 2 {
            readout.learn(&graph.state, label(&train, t));
        }
    }
    graph.reset();
    test.iter()
        .enumerate()
        .filter_map(|(t, &symbol)| {
            graph.step(symbol);
            (t >= 2).then(|| readout.predict(&graph.state))
        })
        .collect()
}
pub struct Run {
    pub graphs: reader_transfer::Graphs,
    pub predictions: Vec<Vec<usize>>,
    pub targets: Vec<usize>,
}
pub fn run(c: &StreamConfig, seed: u64) -> Run {
    let graphs = reader_transfer::graphs(c, seed, SOURCE_DELAY);
    let predictions = graphs
        .graphs
        .iter()
        .map(|g| predict(c, seed, g.clone()))
        .collect();
    let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
    let targets = (2..test.len()).map(|t| label(&test, t)).collect();
    Run {
        graphs,
        predictions,
        targets,
    }
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
    if c.train_tokens <= SOURCE_DELAY || c.test_tokens <= 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "streams too short",
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
fn display(v: Option<f64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_default()
}
fn write_outputs(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),"task=detect current three-bit pattern 101, after seeing current bit; first 2 steps warmup\ngraph_pretraining=reader-only graph feedback on delay-4 recall, 6000 symbols; random-label control uses independent pseudorandom bits\nreader=separately trained online with supervised pattern labels on separate stream for every frozen graph\nmodels=initial, true_pretrained, random_pretrained, strength_matched_initial, fixed_2x_initial, always_negative, exact_pattern\nevaluation=separate stream, reset state, frozen reader and graph; report confusion counts, accuracy, recall, specificity, balanced accuracy, F1\nseeds=graph:s pretrain_input:s+31000 test:s+32000 feedback:s+33000 random_labels:s+34000 reader_train:s+37000, wrapping u64\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n")?;
    let names = [
        "initial",
        "true_pretrained",
        "random_pretrained",
        "strength_matched",
        "fixed_2x",
        "always_negative",
        "exact_pattern",
    ];
    let mut metrics = String::from(
        "seed,model,tp,tn,fp,fn,trials,accuracy,recall,specificity,balanced_accuracy,f1\n",
    );
    let mut predictions=String::from("seed,index,input,target,initial,true_pretrained,random_pretrained,strength_matched,fixed_2x,always_negative,exact_pattern\n");
    let mut diagnostics = String::from(
        "seed,model,mean_abs_weight,mean_abs_drift,clipped_edges,total_edges,matched_scale\n",
    );
    let mut weights=String::from("seed,edge_id,from,to,initial,true_pretrained,random_pretrained,strength_matched,fixed_2x\n");
    let mut total = [Confusion::default(); 7];
    for &seed in &c.seeds {
        let r = run(c, seed);
        let test = temporal_stream::stream(seed.wrapping_add(32000), c.test_tokens);
        let mut counts = [Confusion::default(); 7];
        for (i, &target) in r.targets.iter().enumerate() {
            let v = [
                r.predictions[0][i],
                r.predictions[1][i],
                r.predictions[2][i],
                r.predictions[3][i],
                r.predictions[4][i],
                0,
                target,
            ];
            for k in 0..7 {
                counts[k].add(target, v[k]);
                total[k].add(target, v[k]);
            }
            predictions.push_str(&format!(
                "{seed},{},{},{target},{},{},{},{},{},{},{}\n",
                i + 2,
                test[i + 2],
                v[0],
                v[1],
                v[2],
                v[3],
                v[4],
                v[5],
                v[6]
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
                display(a.recall()),
                display(a.specificity()),
                display(a.balanced_accuracy()),
                display(a.f1())
            ));
        }
        for (k, g) in r.graphs.graphs.iter().enumerate() {
            let drift = g
                .edges
                .iter()
                .zip(&r.graphs.graphs[0].edges)
                .map(|(a, b)| (a.weight - b.weight).abs())
                .sum::<f64>()
                / g.edges.len() as f64;
            let clipped = g
                .edges
                .iter()
                .filter(|e| e.weight.abs() >= 0.5 - 1e-12)
                .count();
            diagnostics.push_str(&format!(
                "{seed},{},{},{},{},{},{}\n",
                names[k],
                g.edges.iter().map(|e| e.weight.abs()).sum::<f64>() / g.edges.len() as f64,
                drift,
                clipped,
                g.edges.len(),
                if k == 3 { r.graphs.matched_factor } else { 1.0 }
            ));
        }
        for i in 0..r.graphs.graphs[0].edges.len() {
            let e = &r.graphs.graphs[0].edges[i];
            weights.push_str(&format!(
                "{seed},{i},{},{},{:.17},{:.17},{:.17},{:.17},{:.17}\n",
                e.from,
                e.to,
                e.weight,
                r.graphs.graphs[1].edges[i].weight,
                r.graphs.graphs[2].edges[i].weight,
                r.graphs.graphs[3].edges[i].weight,
                r.graphs.graphs[4].edges[i].weight
            ));
        }
    }
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("diagnostics.csv"), diagnostics)?;
    fs::write(dir.join("weights.csv"), weights)?;
    let mut summary=String::from("# Pattern detection transfer: 101\n\n| Model | Accuracy | Recall | Specificity | Balanced accuracy | F1 |\n|---|---:|---:|---:|---:|---:|\n");
    for (k, a) in total.iter().enumerate() {
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

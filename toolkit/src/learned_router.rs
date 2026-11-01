//! Supervised linear routing to fixed graph shards; trained from scratch.
//! The external assignment is a label, not self-organized routing.
use crate::{
    sharded::{self, SHARDS},
    Config,
};
use murmur_local_graph::{
    sequence::{graph_predict, Codebook, BITS, POSITIONS},
    Rng,
};
use std::{fs, io, path::Path, time::Instant};
pub const FEATURES: usize = (POSITIONS - 1) * BITS;
pub const EPOCHS: usize = 100;
pub const RATE: f64 = 0.15;
#[derive(Clone, Debug, PartialEq)]
pub struct Router {
    pub weights: [[f64; FEATURES + 1]; SHARDS],
}
impl Router {
    pub fn new() -> Self {
        Self {
            weights: [[0.0; FEATURES + 1]; SHARDS],
        }
    }
    pub fn scores(&self, input: &[i8]) -> [f64; SHARDS] {
        assert_eq!(input.len(), FEATURES);
        std::array::from_fn(|k| {
            self.weights[k][FEATURES]
                + input
                    .iter()
                    .enumerate()
                    .map(|(i, x)| self.weights[k][i] * (*x as f64))
                    .sum::<f64>()
        })
    }
    pub fn predict(&self, input: &[i8]) -> usize {
        let s = self.scores(input);
        let mut best = 0;
        for i in 1..SHARDS {
            if s[i] > s[best] {
                best = i;
            }
        }
        best
    }
    /// Online multiclass logistic update; label is the externally assigned shard.
    pub fn learn(&mut self, input: &[i8], target: usize) {
        assert!(target < SHARDS);
        let scores = self.scores(input);
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut p = scores.map(|x| (x - max).exp());
        let total = p.iter().sum::<f64>();
        for x in &mut p {
            *x /= total;
        }
        for (k, prob) in p.iter().enumerate() {
            let err = (if k == target { 1.0 } else { 0.0 }) - prob;
            for (i, x) in input.iter().enumerate() {
                self.weights[k][i] += RATE * err * (*x as f64) / FEATURES as f64;
            }
            self.weights[k][FEATURES] += RATE * err / FEATURES as f64;
        }
    }
}
impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub source: sharded::Row,
    pub learned_shard: usize,
    pub learned_hebb: usize,
    pub learned_storkey: usize,
}
#[derive(Debug)]
pub struct SeedRun {
    pub rows: Vec<Row>,
    pub stored: Vec<[usize; POSITIONS]>,
    pub router: Router,
    pub train_clean_correct: usize,
    pub train_ns: u128,
    pub route_ns: u128,
    pub query_ns: [u128; 2],
    pub nearest_route_ns: u128,
    pub lookup_ns: u128,
    pub graph_payload: usize,
    pub index_payload: usize,
}
/// Separate augmentation RNG: the labels are assigned during storage and are
/// never obtained from the held-out target symbol or true evaluation labels.
pub fn run_seed(c: &Config, seed: u64) -> Result<SeedRun, &'static str> {
    let base = sharded::run_seed(c, seed)?;
    let book = Codebook::new(seed.wrapping_add(1000));
    let mut router = Router::new();
    let mut rng = Rng(seed.wrapping_add(12000));
    let start = Instant::now();
    for _ in 0..EPOCHS {
        let mut order: Vec<usize> = (0..base.stored.len()).collect();
        for i in (1..order.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            order.swap(i, j);
        }
        for i in order {
            let mut input = book.encode(&base.stored[i])[..FEATURES].to_vec();
            for x in &mut input {
                if rng.uniform() < c.prefix_noise {
                    *x = -*x;
                }
            }
            router.learn(&input, i % SHARDS);
        }
    }
    let train_ns = start.elapsed().as_nanos();
    let train_clean_correct = base
        .stored
        .iter()
        .enumerate()
        .filter(|(i, s)| router.predict(&book.encode(s)[..FEATURES]) == i % SHARDS)
        .count();
    let mut rows = Vec::new();
    let mut route_ns = 0;
    let mut query_ns = [0u128; 2];
    for source in base.rows {
        let start = Instant::now();
        let chosen = router.predict(&source.observed);
        route_ns += start.elapsed().as_nanos();
        let start = Instant::now();
        let learned_hebb = graph_predict(
            &base.shard_hebb[chosen],
            &source.observed,
            &source.missing,
            &book,
            c.sweeps,
        );
        query_ns[0] += start.elapsed().as_nanos();
        let start = Instant::now();
        let learned_storkey = graph_predict(
            &base.shard_storkey[chosen],
            &source.observed,
            &source.missing,
            &book,
            c.sweeps,
        );
        query_ns[1] += start.elapsed().as_nanos();
        rows.push(Row {
            source,
            learned_shard: chosen,
            learned_hebb,
            learned_storkey,
        });
    }
    Ok(SeedRun {
        rows,
        stored: base.stored,
        router,
        train_clean_correct,
        train_ns,
        route_ns,
        query_ns,
        graph_payload: base.resources.four_graph_bytes,
        index_payload: base.resources.router_bytes,
        nearest_route_ns: base.resources.query_ns[6],
        lookup_ns: base.resources.query_ns[7],
    })
}
fn bits(x: &[i8]) -> String {
    x.iter().map(|x| if *x == 1 { '1' } else { '0' }).collect()
}
fn outputs(c: &Config, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),format!("runner_version={}\nrouter=supervised multiclass linear softmax, externally assigned shard labels\nrouter_epochs={}\nrouter_rate={}\nrouter_augmentation_noise=prefix_noise\nrouter_seed=s+12000\nassignment=insertion_index_mod_4\nsplit=stored prefixes reused; fresh evaluation corruptions\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n",env!("CARGO_PKG_VERSION"),EPOCHS,RATE))?;
    let mut predictions=String::from("seed,sequence_index,trial,prefix_bits,initial_missing_bits,target,true_shard,learned_shard,lookup_shard,one_hebb,oracle_hebb,nearest_hebb,learned_hebb,one_storkey,oracle_storkey,nearest_storkey,learned_storkey,lookup\n");
    let mut training =
        String::from("seed,sequence_index,symbol_0,symbol_1,symbol_2,symbol_3,assigned_shard\n");
    let mut metrics = String::from("seed,metric,correct,trials,accuracy\n");
    let mut resources=String::from("seed,router_train_ns,router_predict_ns,nearest_route_ns,lookup_ns,hebb_predict_ns,storkey_predict_ns,trials,router_payload_bytes,graph_payload_bytes,nearest_index_payload_bytes\n");
    let mut weights = String::from("seed,shard,feature,weight\n");
    let names = [
        "one_hebb",
        "oracle_hebb",
        "nearest_hebb",
        "learned_hebb",
        "one_storkey",
        "oracle_storkey",
        "nearest_storkey",
        "learned_storkey",
        "lookup",
        "nearest_route",
        "learned_route",
    ];
    let mut total = 0usize;
    let mut totals = [0usize; 11];
    for &seed in &c.seeds {
        let r = run_seed(c, seed).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        for (i, s) in r.stored.iter().enumerate() {
            training.push_str(&format!(
                "{seed},{i},{},{},{},{},{}\n",
                s[0],
                s[1],
                s[2],
                s[3],
                i % SHARDS
            ));
        }
        for (shard, row) in r.router.weights.iter().enumerate() {
            for (feature, w) in row.iter().enumerate() {
                weights.push_str(&format!("{seed},{shard},{feature},{w:.17}\n"));
            }
        }
        let mut correct = [0usize; 11];
        for row in &r.rows {
            let p = &row.source;
            for (i, v) in [
                p.one_hebb,
                p.oracle_hebb,
                p.routed_hebb,
                row.learned_hebb,
                p.one_storkey,
                p.oracle_storkey,
                p.routed_storkey,
                row.learned_storkey,
                p.lookup,
            ]
            .iter()
            .enumerate()
            {
                correct[i] += usize::from(*v == p.target);
            }
            correct[9] += usize::from(p.routed_shard == p.true_shard);
            correct[10] += usize::from(row.learned_shard == p.true_shard);
            predictions.push_str(&format!(
                "{seed},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                p.sequence_index,
                p.trial,
                bits(&p.observed),
                bits(&p.missing),
                p.target,
                p.true_shard,
                row.learned_shard,
                p.routed_shard,
                p.one_hebb,
                p.oracle_hebb,
                p.routed_hebb,
                row.learned_hebb,
                p.one_storkey,
                p.oracle_storkey,
                p.routed_storkey,
                row.learned_storkey,
                p.lookup
            ));
        }
        let n = r.rows.len();
        total += n;
        for i in 0..11 {
            totals[i] += correct[i];
            metrics.push_str(&format!(
                "{seed},{},{},{},{}\n",
                names[i],
                correct[i],
                n,
                correct[i] as f64 / n as f64
            ));
        }
        metrics.push_str(&format!(
            "{seed},clean_training_route,{},{},{}\n",
            r.train_clean_correct,
            r.stored.len(),
            r.train_clean_correct as f64 / r.stored.len() as f64
        ));
        resources.push_str(&format!(
            "{seed},{},{},{},{},{},{},{n},{},{},{}\n",
            r.train_ns,
            r.route_ns,
            r.nearest_route_ns,
            r.lookup_ns,
            r.query_ns[0],
            r.query_ns[1],
            std::mem::size_of::<Router>(),
            r.graph_payload,
            r.index_payload
        ));
    }
    fs::write(dir.join("training.csv"), training)?;
    fs::write(dir.join("router_weights.csv"), weights)?;
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("resources.csv"), resources)?;
    let mut summary=format!("# Supervised learned router\n\n{} seeds, {} stored sequences, {} epochs with augmented training cues. The router was trained on externally assigned shard labels; evaluation uses fresh corruptions of known prefixes.\n\n| Metric | Correct | Trials | Accuracy |\n|---|---:|---:|---:|\n",c.seeds.len(),c.patterns,EPOCHS);
    for i in 0..11 {
        summary.push_str(&format!(
            "| {} | {} | {} | {:.4} |\n",
            names[i],
            totals[i],
            total,
            totals[i] as f64 / total as f64
        ));
    }
    summary.push_str("\nThe router is trained from scratch, not learned by the graph. True shard labels are supplied during training and not at evaluation. Direct lookup stores the same sequences. No unseen-sequence or self-organized routing claim follows.\n");
    fs::write(dir.join("summary.md"), summary)?;
    Ok(())
}
pub fn execute(c: &Config, dir: &Path) -> io::Result<()> {
    if c.patterns == 0 || c.patterns > 64 || c.patterns % SHARDS != 0 || c.seeds.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "pattern count must be positive, at most 64, and divisible by four; seeds required",
        ));
    }
    if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(dir)?;
    fs::write(dir.join("status.txt"), "running\n")?;
    match outputs(c, dir) {
        Ok(()) => fs::write(dir.join("status.txt"), "complete\n"),
        Err(e) => {
            let _ = fs::write(dir.join("status.txt"), "failed\n");
            Err(e)
        }
    }
}

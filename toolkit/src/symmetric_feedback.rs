//! Local-error graph with symmetric output feedback, for fixed-input retrieval.
use crate::{
    predictive_graph::{self, PredictiveGraph, EPOCHS, TRAIN_SWEEPS},
    Config,
};
use murmur_local_graph::{
    sequence::{Codebook, BITS, N, POSITIONS},
    Graph, Rng,
};
use std::{fs, io, path::Path, time::Instant};
const START: usize = (POSITIONS - 1) * BITS;
#[derive(Clone, Debug)]
pub struct SymmetricModel {
    pub inner: PredictiveGraph,
}
impl SymmetricModel {
    pub fn new(seed: u64, density: f64) -> Self {
        Self {
            inner: PredictiveGraph::new(seed, density, true),
        }
    }
    /// First apply the same local-error update, then average each pair of
    /// feedback weights. Prefix weights remain directed; paired output edges
    /// share the average of the two endpoints' proposed updates.
    pub fn learn_from_state(&mut self, state: &[i8], target: &[i8]) -> f64 {
        let loss = self.inner.learn_from_state(state, target);
        let g = &mut self.inner.graph;
        for i in START..N {
            for j in i + 1..N {
                debug_assert_eq!(g.edges[i * N + j], g.edges[j * N + i]);
                if g.edges[i * N + j] {
                    let average = (g.weights[i * N + j] + g.weights[j * N + i]) / 2.0;
                    g.weights[i * N + j] = average;
                    g.weights[j * N + i] = average;
                }
            }
        }
        loss
    }
}
/// Energy restricted to the output block with externally clamped prefix acting
/// as constant biases. Only meaningful for symmetric zero-diagonal output weights.
pub fn clamped_energy(graph: &Graph, state: &[i8]) -> f64 {
    assert_eq!(state.len(), N);
    let mut e = 0.0;
    for i in START..N {
        for j in 0..START {
            e -= graph.weights[i * N + j] * state[i] as f64 * state[j] as f64;
        }
        for j in i + 1..N {
            e -= graph.weights[i * N + j] * state[i] as f64 * state[j] as f64;
        }
    }
    e
}
fn train(
    model: &mut SymmetricModel,
    stored: &[[usize; POSITIONS]],
    book: &Codebook,
    seed: u64,
) -> f64 {
    let mut rng = Rng(seed.wrapping_add(16000));
    let mut last = 0.0;
    for _ in 0..EPOCHS {
        let mut order: Vec<_> = (0..stored.len()).collect();
        for i in (1..order.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            order.swap(i, j);
        }
        let mut loss = 0.0;
        for i in order {
            let pattern = book.encode(&stored[i]);
            let missing: Vec<i8> = (0..BITS)
                .map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 })
                .collect();
            let state = model
                .inner
                .complete(&pattern[..START], &missing, book, TRAIN_SWEEPS)
                .state;
            loss += model.learn_from_state(&state, &pattern[START..]);
        }
        last = loss / stored.len() as f64;
    }
    last
}
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub source: predictive_graph::Row,
    pub symmetric: usize,
    pub symmetric_converged: bool,
    pub symmetric_sweeps: usize,
    pub symmetric_bit_correct: usize,
}
#[derive(Debug)]
pub struct SeedRun {
    pub rows: Vec<Row>,
    pub symmetric: SymmetricModel,
    pub directed_training_ns: u128,
    pub symmetric_training_ns: u128,
    pub training_loss: f64,
}
pub fn run_seed(c: &Config, seed: u64) -> SeedRun {
    let base = predictive_graph::run_seed(c, seed);
    let book = Codebook::new(seed.wrapping_add(1000));
    let mut model = SymmetricModel::new(seed, c.density);
    let start = Instant::now();
    let loss = train(&mut model, &base.stored, &book, seed);
    let elapsed = start.elapsed().as_nanos();
    let rows = base
        .rows
        .into_iter()
        .map(|source| {
            let p = &source.source;
            let answer = model
                .inner
                .complete(&p.observed, &p.missing, &book, c.sweeps);
            let expected = book.encode(&p.sequence);
            let n = answer.state[START..]
                .iter()
                .zip(&expected[START..])
                .filter(|(a, b)| a == b)
                .count();
            Row {
                symmetric: answer.symbol,
                symmetric_converged: answer.converged,
                symmetric_sweeps: answer.sweeps,
                symmetric_bit_correct: n,
                source,
            }
        })
        .collect();
    SeedRun {
        rows,
        symmetric: model,
        directed_training_ns: base.train_ns[0],
        symmetric_training_ns: elapsed,
        training_loss: loss,
    }
}
fn bits(x: &[i8]) -> String {
    x.iter().map(|x| if *x == 1 { '1' } else { '0' }).collect()
}
fn outputs(c: &Config, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),format!("runner_version={}\nrule=directed local-error update followed by symmetric averaging of each output-output edge pair\ntraining_rng=s+16000\nepochs={}\nsplit=training sequences reused; evaluation uses fresh corruptions\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n",env!("CARGO_PKG_VERSION"),EPOCHS))?;
    let mut pred=String::from("seed,sequence_index,trial,prefix_bits,initial_missing_bits,target,directed,symmetric,no_feedback,storkey,lookup,directed_converged,symmetric_converged,symmetric_sweeps\n");
    let mut metrics = String::from("seed,model,correct,trials,accuracy,converged,training_ns\n");
    let mut weights = String::from("seed,to,from,weight\n");
    let mut training = String::from("seed,sequence_index,symbol_0,symbol_1,symbol_2,symbol_3\n");
    let names = ["directed", "symmetric", "no_feedback", "storkey", "lookup"];
    let mut totals = [0usize; 5];
    let mut total = 0usize;
    for &seed in &c.seeds {
        let r = run_seed(c, seed);
        let mut correct = [0usize; 5];
        let mut conv = [0usize; 3];
        let stored = murmur_local_graph::sequence::sequences(seed.wrapping_add(2000), c.patterns);
        for (i, s) in stored.iter().enumerate() {
            training.push_str(&format!("{seed},{i},{},{},{},{}\n", s[0], s[1], s[2], s[3]));
        }
        for i in START..N {
            for j in 0..N {
                if r.symmetric.inner.graph.edges[i * N + j] {
                    weights.push_str(&format!(
                        "{seed},{i},{j},{:.17}\n",
                        r.symmetric.inner.graph.weights[i * N + j]
                    ));
                }
            }
        }
        for row in &r.rows {
            let a = &row.source;
            let p = &a.source;
            for (i, v) in [a.full, row.symmetric, a.no_feedback, p.storkey, p.lookup]
                .iter()
                .enumerate()
            {
                correct[i] += usize::from(*v == p.target);
            }
            conv[0] += usize::from(a.full_converged);
            conv[1] += usize::from(row.symmetric_converged);
            conv[2] += usize::from(a.no_feedback_converged);
            pred.push_str(&format!(
                "{seed},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                p.sequence_index,
                p.trial,
                bits(&p.observed),
                bits(&p.missing),
                p.target,
                a.full,
                row.symmetric,
                a.no_feedback,
                p.storkey,
                p.lookup,
                a.full_converged,
                row.symmetric_converged,
                row.symmetric_sweeps
            ));
        }
        let n = r.rows.len();
        total += n;
        for i in 0..5 {
            totals[i] += correct[i];
            let convergence = if i < 3 {
                conv[i].to_string()
            } else {
                String::new()
            };
            let time = match i {
                0 => r.directed_training_ns.to_string(),
                1 => r.symmetric_training_ns.to_string(),
                _ => String::new(),
            };
            metrics.push_str(&format!(
                "{seed},{},{},{},{},{},{}\n",
                names[i],
                correct[i],
                n,
                correct[i] as f64 / n as f64,
                convergence,
                time
            ));
        }
    }
    fs::write(dir.join("training.csv"), training)?;
    fs::write(dir.join("predictions.csv"), pred)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("weights.csv"), weights)?;
    let mut summary=format!("# Symmetric feedback experiment\n\n{} seeds, {} stored sequences. Frozen evaluation on corruptions of stored sequences; no unseen-sequence claim.\n\n| Model | Correct | Trials | Accuracy |\n|---|---:|---:|---:|\n",c.seeds.len(),c.patterns);
    for i in 0..5 {
        summary.push_str(&format!(
            "| {} | {} | {} | {:.4} |\n",
            names[i],
            totals[i],
            total,
            totals[i] as f64 / total as f64
        ));
    }
    summary.push_str("\nThe output-feedback weights are symmetrized after each local-error update. Settling is checked within the configured sweep limit, not assumed.\n");
    fs::write(dir.join("summary.md"), summary)?;
    Ok(())
}
pub fn execute(c: &Config, dir: &Path) -> io::Result<()> {
    if c.seeds.is_empty()
        || c.patterns == 0
        || c.patterns > 64
        || c.trials_per_sequence == 0
        || c.sweeps == 0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid configuration",
        ));
    }
    if let Some(p) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(p)?;
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

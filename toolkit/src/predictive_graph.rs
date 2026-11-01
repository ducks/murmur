//! Experimental directed recurrent graph with supervised local prediction errors.
//! This is an established delta-style idea, not a novelty claim.
use crate::{evaluate_seed, Config, Prediction};
use murmur_local_graph::{
    sequence::{Codebook, BITS, N, POSITIONS},
    Graph, Rng,
};
use std::{fs, io, path::Path, time::Instant};
pub const EPOCHS: usize = 30;
pub const RATE: f64 = 0.5;
pub const TRAIN_SWEEPS: usize = 3;
pub const LIMIT: f64 = 1.0;
const START: usize = (POSITIONS - 1) * BITS;

#[derive(Clone, Debug)]
pub struct PredictiveGraph {
    pub graph: Graph,
    pub feedback: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Completion {
    pub symbol: usize,
    pub state: Vec<i8>,
    pub converged: bool,
    pub sweeps: usize,
}
impl PredictiveGraph {
    pub fn new(seed: u64, density: f64, feedback: bool) -> Self {
        let mut graph = Graph::new(N, density, seed);
        if !feedback {
            for i in START..N {
                for j in START..N {
                    graph.edges[i * N + j] = false;
                }
            }
        }
        Self { graph, feedback }
    }
    /// Prefix is clamped; units in the output block update asynchronously.
    pub fn complete(
        &self,
        prefix: &[i8],
        missing: &[i8],
        book: &Codebook,
        max_sweeps: usize,
    ) -> Completion {
        assert_eq!(prefix.len(), START);
        assert_eq!(missing.len(), BITS);
        let mut state = prefix.to_vec();
        state.extend_from_slice(missing);
        let mut clamp = vec![true; N];
        clamp[START..].fill(false);
        let mut converged = false;
        let mut sweeps = 0;
        for _ in 0..max_sweeps {
            sweeps += 1;
            if self.graph.sweep_clamped(&mut state, &clamp) == 0 {
                converged = true;
                break;
            }
        }
        Completion {
            symbol: book.decode(POSITIONS - 1, &state[START..]),
            state,
            converged,
            sweeps,
        }
    }
    /// Each output unit uses its own target minus tanh(local field), and each
    /// incoming edge uses that error and its source activity. Directed updates
    /// need not preserve symmetric Hopfield energy or guarantee convergence.
    pub fn learn_from_state(&mut self, state: &[i8], target: &[i8]) -> f64 {
        assert_eq!(state.len(), N);
        assert_eq!(target.len(), BITS);
        let mut loss = 0.0;
        for (offset, &wanted) in target.iter().enumerate() {
            let i = START + offset;
            let mut field = 0.0;
            let mut degree = 0usize;
            for (j, &activity) in state.iter().enumerate() {
                if self.graph.edges[i * N + j] {
                    field += self.graph.weights[i * N + j] * activity as f64;
                    degree += 1;
                }
            }
            let error = wanted as f64 - field.tanh();
            loss += error * error;
            if degree > 0 {
                for (j, &activity) in state.iter().enumerate() {
                    if self.graph.edges[i * N + j] {
                        self.graph.weights[i * N + j] = (self.graph.weights[i * N + j]
                            + RATE * error * activity as f64 / degree as f64)
                            .clamp(-LIMIT, LIMIT);
                    }
                }
            }
        }
        loss / BITS as f64
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub source: Prediction,
    pub full: usize,
    pub no_feedback: usize,
    pub full_converged: bool,
    pub no_feedback_converged: bool,
    pub full_bit_correct: usize,
    pub no_feedback_bit_correct: usize,
}
#[derive(Debug)]
pub struct SeedRun {
    pub stored: Vec<[usize; POSITIONS]>,
    pub rows: Vec<Row>,
    pub full: PredictiveGraph,
    pub no_feedback: PredictiveGraph,
    pub full_last_epoch_loss: f64,
    pub no_feedback_last_epoch_loss: f64,
    pub train_ns: [u128; 2],
}
fn train(
    model: &mut PredictiveGraph,
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
                .complete(&pattern[..START], &missing, book, TRAIN_SWEEPS)
                .state;
            loss += model.learn_from_state(&state, &pattern[START..]);
        }
        last = loss / stored.len() as f64;
    }
    last
}
pub fn run_seed(c: &Config, seed: u64) -> SeedRun {
    let base = evaluate_seed(c, seed);
    let book = Codebook::new(seed.wrapping_add(1000));
    let mut full = PredictiveGraph::new(seed, c.density, true);
    let mut no_feedback = PredictiveGraph::new(seed, c.density, false);
    let start = Instant::now();
    let full_loss = train(&mut full, &base.stored, &book, seed);
    let full_ns = start.elapsed().as_nanos();
    let start = Instant::now();
    let no_feedback_loss = train(&mut no_feedback, &base.stored, &book, seed);
    let no_feedback_ns = start.elapsed().as_nanos();
    let rows = base
        .rows
        .into_iter()
        .map(|source| {
            let a = full.complete(&source.observed, &source.missing, &book, c.sweeps);
            let b = no_feedback.complete(&source.observed, &source.missing, &book, c.sweeps);
            let pattern = book.encode(&source.sequence);
            let bits = |s: &[i8]| {
                s[START..]
                    .iter()
                    .zip(&pattern[START..])
                    .filter(|(x, y)| x == y)
                    .count()
            };
            Row {
                full: a.symbol,
                no_feedback: b.symbol,
                full_converged: a.converged,
                no_feedback_converged: b.converged,
                full_bit_correct: bits(&a.state),
                no_feedback_bit_correct: bits(&b.state),
                source,
            }
        })
        .collect();
    SeedRun {
        stored: base.stored,
        rows,
        full,
        no_feedback,
        full_last_epoch_loss: full_loss,
        no_feedback_last_epoch_loss: no_feedback_loss,
        train_ns: [full_ns, no_feedback_ns],
    }
}
fn bits(x: &[i8]) -> String {
    x.iter().map(|x| if *x == 1 { '1' } else { '0' }).collect()
}
fn outputs(c: &Config, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),format!("runner_version={}\nlocal_rule=target_minus_tanh_field times source activity / incoming degree\nepochs={}\nrate={}\ntraining_sweeps={}\nweight_limit={}\ntraining_inputs=clean stored prefixes and randomized output state\ntraining_rng=s+16000\nevaluation=novel corruptions of stored sequences; weights frozen\nfeedback=sparse directed output-to-output edges; no symmetry or energy guarantee\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n",env!("CARGO_PKG_VERSION"),EPOCHS,RATE,TRAIN_SWEEPS,LIMIT))?;
    let mut pred=String::from("seed,sequence_index,trial,prefix_bits,initial_missing_bits,target,predictive,predictive_no_feedback,hebb,storkey,lookup,predictive_converged,no_feedback_converged\n");
    let mut metrics=String::from("seed,model,correct,trials,accuracy,converged,output_bits_correct,output_bits_total,last_training_loss,training_ns\n");
    let mut weights = String::from("seed,model,to,from,weight\n");
    let mut training = String::from("seed,sequence_index,symbol_0,symbol_1,symbol_2,symbol_3\n");
    let names = [
        "predictive",
        "predictive_no_feedback",
        "hebb",
        "storkey",
        "lookup",
    ];
    let mut totals = [0usize; 5];
    let mut total = 0;
    for &seed in &c.seeds {
        let r = run_seed(c, seed);
        for (i, s) in r.stored.iter().enumerate() {
            training.push_str(&format!("{seed},{i},{},{},{},{}\n", s[0], s[1], s[2], s[3]));
        }
        for (name, m) in [
            ("predictive", &r.full),
            ("predictive_no_feedback", &r.no_feedback),
        ] {
            for i in START..N {
                for j in 0..N {
                    if m.graph.edges[i * N + j] {
                        weights.push_str(&format!(
                            "{seed},{name},{i},{j},{:.17}\n",
                            m.graph.weights[i * N + j]
                        ));
                    }
                }
            }
        }
        let mut correct = [0usize; 5];
        let mut conv = [0usize; 2];
        let mut bc = [0usize; 2];
        for row in &r.rows {
            let p = &row.source;
            for (i, v) in [row.full, row.no_feedback, p.trained, p.storkey, p.lookup]
                .iter()
                .enumerate()
            {
                correct[i] += usize::from(*v == p.target);
            }
            conv[0] += usize::from(row.full_converged);
            conv[1] += usize::from(row.no_feedback_converged);
            bc[0] += row.full_bit_correct;
            bc[1] += row.no_feedback_bit_correct;
            pred.push_str(&format!(
                "{seed},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                p.sequence_index,
                p.trial,
                bits(&p.observed),
                bits(&p.missing),
                p.target,
                row.full,
                row.no_feedback,
                p.trained,
                p.storkey,
                p.lookup,
                row.full_converged,
                row.no_feedback_converged
            ));
        }
        let n = r.rows.len();
        total += n;
        for i in 0..5 {
            totals[i] += correct[i];
            let extra = if i < 2 {
                format!(
                    "{},{},{},{},{}",
                    conv[i],
                    bc[i],
                    n * BITS,
                    if i == 0 {
                        r.full_last_epoch_loss
                    } else {
                        r.no_feedback_last_epoch_loss
                    },
                    r.train_ns[i]
                )
            } else {
                String::from(",,,,")
            };
            metrics.push_str(&format!(
                "{seed},{},{},{},{},{}\n",
                names[i],
                correct[i],
                n,
                correct[i] as f64 / n as f64,
                extra
            ));
        }
    }
    fs::write(dir.join("training.csv"), training)?;
    fs::write(dir.join("predictions.csv"), pred)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("weights.csv"), weights)?;
    let mut summary=format!("# Local predictive-error graph\n\n{} seeds, {} stored sequences; {} supervised epochs per predictive graph. Evaluation uses fresh corruptions of stored sequences, not unseen sequences.\n\n| Model | Correct | Trials | Accuracy |\n|---|---:|---:|---:|\n",c.seeds.len(),c.patterns,EPOCHS);
    for i in 0..5 {
        summary.push_str(&format!(
            "| {} | {} | {} | {:.4} |\n",
            names[i],
            totals[i],
            total,
            totals[i] as f64 / total as f64
        ));
    }
    summary.push_str("\nThe local error uses a supervised target at each output unit. Directed recurrent weights have no energy guarantee. Hebbian and Storkey baselines see one presentation per sequence, versus 30 epochs here.\n");
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

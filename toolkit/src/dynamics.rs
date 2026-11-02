//! Post-training interventions on directed predictive-graph update dynamics.
use crate::{predictive_graph, Config};
use murmur_local_graph::{
    sequence::{Codebook, BITS, N, POSITIONS},
    Graph, Rng,
};
use std::{collections::HashMap, fs, io, path::Path};
const START: usize = (POSITIONS - 1) * BITS;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Forward20,
    Forward100,
    Reverse20,
    Shuffled20,
    Margin005,
    Margin020,
}
impl Mode {
    pub const ALL: [Mode; 6] = [
        Self::Forward20,
        Self::Forward100,
        Self::Reverse20,
        Self::Shuffled20,
        Self::Margin005,
        Self::Margin020,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Forward20 => "forward20",
            Self::Forward100 => "forward100",
            Self::Reverse20 => "reverse20",
            Self::Shuffled20 => "shuffled20",
            Self::Margin005 => "margin005",
            Self::Margin020 => "margin020",
        }
    }
    fn limit(self) -> usize {
        if self == Self::Forward100 {
            100
        } else {
            20
        }
    }
    fn threshold(self) -> f64 {
        match self {
            Self::Margin005 => 0.05,
            Self::Margin020 => 0.20,
            _ => 0.0,
        }
    }
    fn fixed_order(self) -> bool {
        self != Self::Shuffled20
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub symbol: usize,
    pub converged: bool,
    pub strict_stable: bool,
    pub sweeps: usize,
    pub cycle_length: Option<usize>,
}
fn field(graph: &Graph, state: &[i8], i: usize) -> f64 {
    (0..N)
        .map(|j| graph.weights[i * N + j] * state[j] as f64)
        .sum()
}
/// True iff a standard zero-threshold update would leave every output unit unchanged.
pub fn strict_stable(graph: &Graph, state: &[i8]) -> bool {
    (START..N).all(|i| {
        let h = field(graph, state, i);
        !((h > 0.0 && state[i] < 0) || (h < 0.0 && state[i] > 0))
    })
}
/// Weights and prefix never change. Thresholded convergence is reported separately
/// from stability under the original zero-threshold rule. Cycles are only claimed
/// for deterministic fixed orders, when a full-sweep boundary state repeats.
pub fn simulate(
    graph: &Graph,
    prefix: &[i8],
    missing: &[i8],
    book: &Codebook,
    mode: Mode,
    random_seed: u64,
) -> Outcome {
    assert_eq!(graph.n, N);
    assert_eq!(prefix.len(), START);
    assert_eq!(missing.len(), BITS);
    let mut state = prefix.to_vec();
    state.extend_from_slice(missing);
    let mut seen = HashMap::new();
    if mode.fixed_order() {
        seen.insert(state[START..].to_vec(), 0usize);
    }
    let mut rng = Rng(random_seed);
    let mut cycle = None;
    let mut converged = false;
    let mut sweeps = 0;
    for step in 1..=mode.limit() {
        let mut order: Vec<usize> = (START..N).collect();
        if mode == Mode::Reverse20 {
            order.reverse();
        }
        if mode == Mode::Shuffled20 {
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
        }
        let mut changes = 0;
        for i in order {
            let h = field(graph, &state, i);
            let threshold = mode.threshold();
            let next = if h > threshold {
                1
            } else if h < -threshold {
                -1
            } else {
                state[i]
            };
            changes += usize::from(next != state[i]);
            state[i] = next;
        }
        sweeps = step;
        if changes == 0 {
            converged = true;
            break;
        }
        if mode.fixed_order() {
            let key = state[START..].to_vec();
            if let Some(previous) = seen.get(&key) {
                if cycle.is_none() {
                    cycle = Some(step - *previous);
                }
            } else {
                seen.insert(key, step);
            }
        }
    }
    Outcome {
        symbol: book.decode(POSITIONS - 1, &state[START..]),
        converged,
        strict_stable: strict_stable(graph, &state),
        sweeps,
        cycle_length: cycle,
    }
}
fn bits(x: &[i8]) -> String {
    x.iter().map(|x| if *x == 1 { '1' } else { '0' }).collect()
}
fn outputs(c: &Config, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),format!("runner_version={}\nweights=frozen after predictive_graph training\nmodes=forward20,forward100,reverse20,shuffled20,margin005,margin020\nshuffled_order_rng=(seed+25000 wrapping) XOR (sequence_index*0x9e3779b97f4a7c15 wrapping) XOR trial\ncycles=reported only for deterministic fixed orders\nstrict_stable=stable under original zero-threshold rule\nsplit=training sequences reused; evaluation uses fresh corruptions\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n",env!("CARGO_PKG_VERSION")))?;
    let mut predictions=String::from("seed,sequence_index,trial,mode,prefix_bits,initial_missing_bits,target,prediction,converged,strict_stable,sweeps,cycle_length,no_feedback_prediction,one_pass_storkey_prediction,lookup_prediction\n");
    let mut metrics=String::from("seed,mode,correct,trials,accuracy,converged,strict_stable,cycles,sweeps_total,no_feedback_correct,storkey_correct,lookup_correct\n");
    let mut totals = [(0usize, 0usize, 0usize, 0usize, 0usize); 6];
    let mut trials = 0usize;
    for &seed in &c.seeds {
        let base = predictive_graph::run_seed(c, seed);
        let book = Codebook::new(seed.wrapping_add(1000));
        let mut counts = [(0usize, 0usize, 0usize, 0usize, 0usize); 6];
        let mut controls = [0usize; 3];
        for row in &base.rows {
            let p = &row.source;
            controls[0] += usize::from(row.no_feedback == p.target);
            controls[1] += usize::from(p.storkey == p.target);
            controls[2] += usize::from(p.lookup == p.target);
            for (i, mode) in Mode::ALL.into_iter().enumerate() {
                let random_seed = seed.wrapping_add(25000)
                    ^ (p.sequence_index as u64).wrapping_mul(0x9e3779b97f4a7c15)
                    ^ (p.trial as u64);
                let o = simulate(
                    &base.full.graph,
                    &p.observed,
                    &p.missing,
                    &book,
                    mode,
                    random_seed,
                );
                if mode == Mode::Forward20
                    && (o.symbol != row.full || o.converged != row.full_converged)
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "forward20 differs from original predictive-graph inference",
                    ));
                }
                let a = &mut counts[i];
                a.0 += usize::from(o.symbol == p.target);
                a.1 += usize::from(o.converged);
                a.2 += usize::from(o.strict_stable);
                a.3 += usize::from(o.cycle_length.is_some());
                a.4 += o.sweeps;
                predictions.push_str(&format!(
                    "{seed},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                    p.sequence_index,
                    p.trial,
                    mode.name(),
                    bits(&p.observed),
                    bits(&p.missing),
                    p.target,
                    o.symbol,
                    o.converged,
                    o.strict_stable,
                    o.sweeps,
                    o.cycle_length.map(|v| v.to_string()).unwrap_or_default(),
                    row.no_feedback,
                    p.storkey,
                    p.lookup
                ));
            }
        }
        let n = base.rows.len();
        trials += n;
        for (i, mode) in Mode::ALL.into_iter().enumerate() {
            let x = counts[i];
            let all = &mut totals[i];
            all.0 += x.0;
            all.1 += x.1;
            all.2 += x.2;
            all.3 += x.3;
            all.4 += x.4;
            metrics.push_str(&format!(
                "{seed},{},{},{n},{},{},{},{},{},{},{},{}\n",
                mode.name(),
                x.0,
                x.0 as f64 / n as f64,
                x.1,
                x.2,
                if mode.fixed_order() {
                    x.3.to_string()
                } else {
                    String::new()
                },
                x.4,
                controls[0],
                controls[1],
                controls[2]
            ));
        }
    }
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    let mut summary=format!("# Frozen-weight dynamics interventions\n\n{} seeds, {} stored sequences. Models see the same stored sequences and noisy queries; changing update dynamics does not change weights. Evaluation uses corruptions of stored sequences.\n\n| Mode | Accuracy | Converged by own rule | Stable under original rule | Mean sweeps |\n|---|---:|---:|---:|---:|\n",c.seeds.len(),c.patterns);
    for (i, mode) in Mode::ALL.into_iter().enumerate() {
        let x = totals[i];
        summary.push_str(&format!(
            "| {} | {:.4} | {:.4} | {:.4} | {:.2} |\n",
            mode.name(),
            x.0 as f64 / trials as f64,
            x.1 as f64 / trials as f64,
            x.2 as f64 / trials as f64,
            x.4 as f64 / trials as f64
        ));
    }
    summary.push_str("\nMargin-based convergence is not necessarily stability under the original rule. Cycle detection applies only to fixed-order dynamics. See EXPERIMENTS.md for protocol and caveats.\n");
    fs::write(dir.join("summary.md"), summary)?;
    Ok(())
}
pub fn execute(c: &Config, dir: &Path) -> io::Result<()> {
    if c.seeds.is_empty()
        || c.patterns == 0
        || c.patterns > 64
        || c.trials_per_sequence == 0
        || c.sweeps != 20
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "valid config with sweeps=20 required",
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

//! First research-runner slice: matched sequence-completion inputs and outputs.
pub mod dynamics;
pub mod independent_encoding;
pub mod learned_router;
pub mod matched_budget;
pub mod predictive_graph;
pub mod sharded;
pub mod symmetric_feedback;
pub mod temporal_plasticity;
pub mod temporal_stream;
use murmur_local_graph::{
    diagnostics::{margin_flips, without_output_feedback},
    sequence::{
        graph_complete, graph_predict, lookup, sequences, Codebook, BITS, N, POSITIONS, SYMBOLS,
    },
    Graph, Rng,
};
use std::{collections::BTreeMap, fs, io, path::Path, time::Instant};

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub seeds: Vec<u64>,
    pub patterns: usize,
    pub prefix_noise: f64,
    pub trials_per_sequence: usize,
    pub sweeps: usize,
    pub density: f64,
}
impl Config {
    /// Strict key=value format: unknown, duplicate, or missing keys are errors.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut fields = BTreeMap::new();
        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("line {}: expected key=value", line_no + 1))?;
            let (key, value) = (key.trim(), value.trim());
            if ![
                "schema_version",
                "seeds",
                "patterns",
                "prefix_noise",
                "trials_per_sequence",
                "sweeps",
                "density",
            ]
            .contains(&key)
            {
                return Err(format!("unknown key: {key}"));
            }
            if fields.insert(key, value).is_some() {
                return Err(format!("duplicate key: {key}"));
            }
        }
        let get = |key| {
            fields
                .get(key)
                .copied()
                .ok_or_else(|| format!("missing key: {key}"))
        };
        if get("schema_version")? != "1" {
            return Err("unsupported schema_version (expected 1)".into());
        }
        let seeds: Vec<u64> = get("seeds")?
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<u64>()
                    .map_err(|_| "invalid seed".to_string())
            })
            .collect::<Result<_, _>>()?;
        let patterns = get("patterns")?
            .parse::<usize>()
            .map_err(|_| "invalid patterns")?;
        let prefix_noise = get("prefix_noise")?
            .parse::<f64>()
            .map_err(|_| "invalid prefix_noise")?;
        let trials_per_sequence = get("trials_per_sequence")?
            .parse::<usize>()
            .map_err(|_| "invalid trials_per_sequence")?;
        let sweeps = get("sweeps")?
            .parse::<usize>()
            .map_err(|_| "invalid sweeps")?;
        let density = get("density")?
            .parse::<f64>()
            .map_err(|_| "invalid density")?;
        if seeds.is_empty() {
            return Err("at least one seed required".into());
        }
        let mut unique = seeds.clone();
        unique.sort_unstable();
        unique.dedup();
        if unique.len() != seeds.len() {
            return Err("duplicate seeds".into());
        }
        if !(1..=64).contains(&patterns) {
            return Err("patterns must be between 1 and 64".into());
        }
        if !prefix_noise.is_finite() || !(0.0..=1.0).contains(&prefix_noise) {
            return Err("prefix_noise must be finite and between 0 and 1".into());
        }
        if !density.is_finite() || !(0.0..=1.0).contains(&density) {
            return Err("density must be finite and between 0 and 1".into());
        }
        if trials_per_sequence == 0 || sweeps == 0 {
            return Err("trials_per_sequence and sweeps must be positive".into());
        }
        patterns
            .checked_mul(trials_per_sequence)
            .and_then(|n| n.checked_mul(seeds.len()))
            .ok_or("trial count overflow")?;
        Ok(Self {
            seeds,
            patterns,
            prefix_noise,
            trials_per_sequence,
            sweeps,
            density,
        })
    }
    pub fn resolved(&self) -> String {
        format!("schema_version=1\nseeds={}\npatterns={}\nprefix_noise={}\ntrials_per_sequence={}\nsweeps={}\ndensity={}\n",self.seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(","),self.patterns,self.prefix_noise,self.trials_per_sequence,self.sweeps,self.density)
    }
}
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    pub seed: u64,
    pub sequence_index: usize,
    pub trial: usize,
    pub sequence: [usize; POSITIONS],
    pub observed: Vec<i8>,
    pub missing: Vec<i8>,
    pub target: usize,
    pub trained: usize,
    pub storkey: usize,
    pub no_feedback: usize,
    pub lookup: usize,
    pub untrained: usize,
    pub majority: usize,
}
#[derive(Debug)]
pub struct SeedRun {
    pub seed: u64,
    pub stored: Vec<[usize; POSITIONS]>,
    pub rows: Vec<Prediction>,
    pub resource_rows: Vec<Resource>,
    pub field_flips: [usize; 2],
    pub output_bits_correct: [usize; 2],
}
#[derive(Debug)]
pub struct Resource {
    pub model: &'static str,
    pub training_ns: Option<u128>,
    pub prediction_ns: u128,
    pub owned_payload_bytes: usize,
}
fn graph_payload_bytes(g: &Graph) -> usize {
    g.weights.capacity() * std::mem::size_of::<f64>()
        + g.edges.capacity() * std::mem::size_of::<bool>()
}
/// Every variant receives exactly the same pre-generated prefix and missing-state initialization.
pub fn evaluate_seed(c: &Config, seed: u64) -> SeedRun {
    let book = Codebook::new(seed.wrapping_add(1000));
    let stored = sequences(seed.wrapping_add(2000), c.patterns);
    let mut graph = Graph::new(N, c.density, seed);
    let untrained = graph.clone();
    let mut storkey = untrained.clone();
    let training_start = Instant::now();
    for seq in &stored {
        graph.learn(&book.encode(seq));
    }
    let hebb_train = training_start.elapsed().as_nanos();
    let training_start = Instant::now();
    for seq in &stored {
        storkey.learn_storkey(&book.encode(seq));
    }
    let storkey_train = training_start.elapsed().as_nanos();
    let no_feedback = without_output_feedback(&graph);
    let mut counts = [0usize; SYMBOLS];
    for seq in &stored {
        counts[seq[POSITIONS - 1]] += 1;
    }
    let mut majority = 0;
    for i in 1..SYMBOLS {
        if counts[i] > counts[majority] {
            majority = i;
        }
    }
    let mut rng = Rng(seed.wrapping_add(3000));
    let mut rows = Vec::new();
    let mut elapsed = [0u128; 3];
    let mut output_bits_correct = [0usize; 2];
    let mut field_flips = [0usize; 2];
    for (sequence_index, seq) in stored.iter().enumerate() {
        let pattern = book.encode(seq);
        let mut oracle = untrained.clone();
        oracle.learn(&pattern); // First Storkey update is identical to Hebbian.
        field_flips[0] += margin_flips(&graph, &oracle, &pattern);
        field_flips[1] += margin_flips(&storkey, &oracle, &pattern);
        for trial in 0..c.trials_per_sequence {
            let mut observed = pattern[..(POSITIONS - 1) * BITS].to_vec();
            for bit in &mut observed {
                if rng.uniform() < c.prefix_noise {
                    *bit = -*bit;
                }
            }
            let missing: Vec<i8> = (0..BITS)
                .map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 })
                .collect();
            let target = seq[POSITIONS - 1];
            let start = Instant::now();
            let hebb_state = graph_complete(&graph, &observed, &missing, c.sweeps);
            let trained = book.decode(POSITIONS - 1, &hebb_state[(POSITIONS - 1) * BITS..]);
            elapsed[0] += start.elapsed().as_nanos();
            let start = Instant::now();
            let storkey_state = graph_complete(&storkey, &observed, &missing, c.sweeps);
            let storkey_prediction =
                book.decode(POSITIONS - 1, &storkey_state[(POSITIONS - 1) * BITS..]);
            elapsed[1] += start.elapsed().as_nanos();
            let start = Instant::now();
            let lookup_prediction = lookup(&observed, &stored, &book);
            elapsed[2] += start.elapsed().as_nanos();
            let start_bit = (POSITIONS - 1) * BITS;
            for (i, state) in [&hebb_state, &storkey_state].iter().enumerate() {
                output_bits_correct[i] += state[start_bit..]
                    .iter()
                    .zip(&pattern[start_bit..])
                    .filter(|(a, b)| a == b)
                    .count();
            }
            rows.push(Prediction {
                seed,
                sequence_index,
                trial,
                sequence: *seq,
                trained,
                storkey: storkey_prediction,
                no_feedback: graph_predict(&no_feedback, &observed, &missing, &book, c.sweeps),
                lookup: lookup_prediction,
                untrained: graph_predict(&untrained, &observed, &missing, &book, c.sweeps),
                majority,
                observed,
                missing,
                target,
            });
        }
    }
    let lookup_bytes = stored.capacity() * std::mem::size_of::<[usize; POSITIONS]>();
    let resource_rows = vec![
        Resource {
            model: "trained",
            training_ns: Some(hebb_train),
            prediction_ns: elapsed[0],
            owned_payload_bytes: graph_payload_bytes(&graph),
        },
        Resource {
            model: "storkey",
            training_ns: Some(storkey_train),
            prediction_ns: elapsed[1],
            owned_payload_bytes: graph_payload_bytes(&storkey),
        },
        Resource {
            model: "lookup",
            training_ns: None,
            prediction_ns: elapsed[2],
            owned_payload_bytes: lookup_bytes,
        },
    ];
    SeedRun {
        seed,
        stored,
        rows,
        resource_rows,
        field_flips,
        output_bits_correct,
    }
}
fn bits(x: &[i8]) -> String {
    x.iter().map(|x| if *x == 1 { '1' } else { '0' }).collect()
}
fn make_outputs(c: &Config, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("provenance.txt"),format!("runner_version={}\nos={}\narch={}\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\nseed_scheme=graph:s;codebook:s+1000;train:s+2000;test_cues:s+3000 (wrapping u64)\nsplit=training sequences reused; evaluation uses fresh corrupted cues\n",env!("CARGO_PKG_VERSION"),std::env::consts::OS,std::env::consts::ARCH))?;
    let mut predictions=String::from("seed,sequence_index,trial,observed_prefix_bits,initial_missing_bits,target,trained,storkey,no_feedback,lookup,untrained,majority\n");
    let mut training = String::from("seed,sequence_index,symbol_0,symbol_1,symbol_2,symbol_3\n");
    let mut metrics = String::from("seed,model,correct,trials,accuracy\n");
    let mut resources = String::from(
        "seed,model,training_ns,prediction_total_ns,trials,model_owned_payload_bytes\n",
    );
    let mut diagnostics = String::from(
        "seed,model,field_flips,field_observations,output_bits_correct,output_bit_observations\n",
    );
    let mut totals = [0usize; 6];
    let mut trials = 0usize;
    let names = [
        "trained",
        "storkey",
        "no_feedback",
        "lookup",
        "untrained",
        "majority",
    ];
    for &seed in &c.seeds {
        let run = evaluate_seed(c, seed);
        let mut correct = [0usize; 6];
        for (i, s) in run.stored.iter().enumerate() {
            training.push_str(&format!("{seed},{i},{},{},{},{}\n", s[0], s[1], s[2], s[3]));
        }
        for row in &run.rows {
            for (i, p) in [
                row.trained,
                row.storkey,
                row.no_feedback,
                row.lookup,
                row.untrained,
                row.majority,
            ]
            .iter()
            .enumerate()
            {
                correct[i] += usize::from(*p == row.target);
            }
            predictions.push_str(&format!(
                "{seed},{},{},{},{},{},{},{},{},{},{},{}\n",
                row.sequence_index,
                row.trial,
                bits(&row.observed),
                bits(&row.missing),
                row.target,
                row.trained,
                row.storkey,
                row.no_feedback,
                row.lookup,
                row.untrained,
                row.majority
            ));
        }
        for i in 0..6 {
            totals[i] += correct[i];
            metrics.push_str(&format!(
                "{seed},{},{},{},{}\n",
                names[i],
                correct[i],
                run.rows.len(),
                correct[i] as f64 / run.rows.len() as f64
            ));
        }
        for r in &run.resource_rows {
            let training = r.training_ns.map(|x| x.to_string()).unwrap_or_default();
            resources.push_str(&format!(
                "{seed},{},{training},{},{},{}\n",
                r.model,
                r.prediction_ns,
                run.rows.len(),
                r.owned_payload_bytes
            ));
        }
        for (i, name) in ["trained", "storkey"].iter().enumerate() {
            diagnostics.push_str(&format!(
                "{seed},{name},{},{},{},{}\n",
                run.field_flips[i],
                c.patterns * BITS,
                run.output_bits_correct[i],
                run.rows.len() * BITS
            ));
        }
        trials += run.rows.len();
    }
    fs::write(dir.join("training.csv"), training)?;
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("resources.csv"), resources)?;
    fs::write(dir.join("diagnostics.csv"), diagnostics)?;
    let mut summary=format!("# Sequence completion run\n\n{} seeds, {} stored sequences per seed, {} trials per sequence, prefix bit noise {}.\n\nTraining sequences are reused in evaluation; cues are newly corrupted. This is retrieval, not unseen-sequence generalization. Each model receives identical observations.\n\n| Model | Correct | Trials | Accuracy |\n|---|---:|---:|---:|\n",c.seeds.len(),c.patterns,c.trials_per_sequence,c.prefix_noise);
    for i in 0..6 {
        summary.push_str(&format!(
            "| {} | {} | {} | {:.4} |\n",
            names[i],
            totals[i],
            trials,
            totals[i] as f64 / trials as f64
        ));
    }
    summary.push_str("\nThe no-feedback variant removes edges between missing-block units. The Storkey-style rule uses old local fields on the same sparse topology. Lookup stores training sequences and is not resource-matched to the graph. Resources exclude common data, allocation overhead, and I/O; see resources.csv and documentation. Field flips use an oracle target state, not a prediction input. Source and Nix revisions are unrecorded; see provenance.txt.\n");
    fs::write(dir.join("summary.md"), summary)?;
    Ok(())
}
/// Refuses an existing output path; marks incomplete or failed runs explicitly.
pub fn execute(c: &Config, dir: &Path) -> io::Result<()> {
    if let Some(parent) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(dir)?;
    fs::write(dir.join("status.txt"), "running\n")?;
    match make_outputs(c, dir) {
        Ok(()) => fs::write(dir.join("status.txt"), "complete\n"),
        Err(e) => {
            let _ = fs::write(dir.join("status.txt"), "failed\n");
            Err(e)
        }
    }
}

pub mod matched_perturbation;

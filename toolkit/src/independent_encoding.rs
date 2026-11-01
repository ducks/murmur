//! Sensitivity control: independent extra signal bits on the 256-unit graph.
use crate::{
    matched_budget::{LARGE_BITS, LARGE_N},
    sharded, Config,
};
use murmur_local_graph::{
    sequence::{Codebook, BITS, POSITIONS, SYMBOLS},
    Graph, Rng,
};
use std::{fs, io, path::Path, time::Instant};
/// Interleave corresponding blocks from the original and extra codebooks.
pub fn combine(a: &[i8], b: &[i8]) -> Vec<i8> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % BITS, 0);
    let mut out = Vec::with_capacity(a.len() * 2);
    for (x, y) in a.chunks_exact(BITS).zip(b.chunks_exact(BITS)) {
        out.extend_from_slice(x);
        out.extend_from_slice(y);
    }
    out
}
pub fn decode(a: &Codebook, b: &Codebook, output: &[i8]) -> usize {
    assert_eq!(output.len(), LARGE_BITS);
    let mut best = 0;
    let mut distance = usize::MAX;
    for symbol in 0..SYMBOLS {
        let d = output[..BITS]
            .iter()
            .zip(&a.codes[POSITIONS - 1][symbol])
            .filter(|(x, y)| x != y)
            .count()
            + output[BITS..]
                .iter()
                .zip(&b.codes[POSITIONS - 1][symbol])
                .filter(|(x, y)| x != y)
                .count();
        if d < distance {
            distance = d;
            best = symbol;
        }
    }
    best
}
pub fn predict(
    g: &Graph,
    observed: &[i8],
    extra: &[i8],
    missing: &[i8],
    extra_missing: &[i8],
    a: &Codebook,
    b: &Codebook,
    sweeps: usize,
) -> usize {
    assert_eq!(g.n, LARGE_N);
    assert_eq!(observed.len(), (POSITIONS - 1) * BITS);
    assert_eq!(missing.len(), BITS);
    let mut state = combine(observed, extra);
    state.extend(combine(missing, extra_missing));
    let mut clamped = vec![true; LARGE_N];
    clamped[(POSITIONS - 1) * LARGE_BITS..].fill(false);
    for _ in 0..sweeps {
        if g.sweep_clamped(&mut state, &clamped) == 0 {
            break;
        }
    }
    decode(a, b, &state[(POSITIONS - 1) * LARGE_BITS..])
}
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub source: sharded::Row,
    pub extra_prefix: Vec<i8>,
    pub extra_missing: Vec<i8>,
    pub extra_hebb: usize,
    pub extra_storkey: usize,
}
#[derive(Debug)]
pub struct SeedRun {
    pub rows: Vec<Row>,
    pub train_ns: [u128; 2],
    pub query_ns: [u128; 2],
    pub payload_bytes: usize,
    pub active_edges: usize,
}
pub fn run_seed(c: &Config, seed: u64) -> Result<SeedRun, &'static str> {
    let base = sharded::run_seed(c, seed)?;
    let first = Codebook::new(seed.wrapping_add(1000));
    let extra = Codebook::new(seed.wrapping_add(10000));
    let mut h = Graph::new(LARGE_N, c.density, seed.wrapping_add(9000));
    let mut s = h.clone();
    let start = Instant::now();
    for seq in &base.stored {
        h.learn(&combine(&first.encode(seq), &extra.encode(seq)));
    }
    let ht = start.elapsed().as_nanos();
    let start = Instant::now();
    for seq in &base.stored {
        s.learn_storkey(&combine(&first.encode(seq), &extra.encode(seq)));
    }
    let st = start.elapsed().as_nanos();
    let mut rng = Rng(seed.wrapping_add(11000));
    let mut rows = Vec::new();
    let mut elapsed = [0u128; 2];
    for source in base.rows {
        let seq = &base.stored[source.sequence_index];
        let mut extra_prefix = extra.encode(seq)[..(POSITIONS - 1) * BITS].to_vec();
        for bit in &mut extra_prefix {
            if rng.uniform() < c.prefix_noise {
                *bit = -*bit;
            }
        }
        let extra_missing: Vec<i8> = (0..BITS)
            .map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 })
            .collect();
        let start = Instant::now();
        let extra_hebb = predict(
            &h,
            &source.observed,
            &extra_prefix,
            &source.missing,
            &extra_missing,
            &first,
            &extra,
            c.sweeps,
        );
        elapsed[0] += start.elapsed().as_nanos();
        let start = Instant::now();
        let extra_storkey = predict(
            &s,
            &source.observed,
            &extra_prefix,
            &source.missing,
            &extra_missing,
            &first,
            &extra,
            c.sweeps,
        );
        elapsed[1] += start.elapsed().as_nanos();
        rows.push(Row {
            source,
            extra_prefix,
            extra_missing,
            extra_hebb,
            extra_storkey,
        });
    }
    let payload_bytes = h.weights.capacity() * std::mem::size_of::<f64>()
        + h.edges.capacity() * std::mem::size_of::<bool>();
    let active_edges = h.edges.iter().filter(|e| **e).count() / 2;
    Ok(SeedRun {
        rows,
        train_ns: [ht, st],
        query_ns: elapsed,
        payload_bytes,
        active_edges,
    })
}
fn bits(x: &[i8]) -> String {
    x.iter().map(|x| if *x == 1 { '1' } else { '0' }).collect()
}
fn outputs(c: &Config, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),format!("runner_version={}\nlarge_units=256\nextra_codebook_seed=s+10000\nextra_noise_seed=s+11000\nlarge_topology_seed=s+9000\nextra_signal=independent code and independently corrupted prefix bits; added noisy evidence\nsplit=stored training sequences; new corruptions\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n",env!("CARGO_PKG_VERSION")))?;
    let mut metrics = String::from("seed,metric,correct,trials,accuracy\n");
    let mut predictions=String::from("seed,sequence_index,trial,base_prefix_bits,base_missing_bits,extra_prefix_bits,extra_missing_bits,target,extra_hebb,extra_storkey\n");
    let mut resources = String::from(
        "seed,model,train_ns,prediction_ns,trials,active_edges,model_owned_payload_bytes\n",
    );
    let mut totals = [0usize; 2];
    let mut n_total = 0;
    for &seed in &c.seeds {
        let r = run_seed(c, seed).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        let mut correct = [0usize; 2];
        for row in &r.rows {
            let p = &row.source;
            correct[0] += usize::from(row.extra_hebb == p.target);
            correct[1] += usize::from(row.extra_storkey == p.target);
            predictions.push_str(&format!(
                "{seed},{},{},{},{},{},{},{},{},{}\n",
                p.sequence_index,
                p.trial,
                bits(&p.observed),
                bits(&p.missing),
                bits(&row.extra_prefix),
                bits(&row.extra_missing),
                p.target,
                row.extra_hebb,
                row.extra_storkey
            ));
        }
        let n = r.rows.len();
        n_total += n;
        for (i, name) in ["independent_hebb", "independent_storkey"]
            .iter()
            .enumerate()
        {
            totals[i] += correct[i];
            metrics.push_str(&format!(
                "{seed},{name},{},{n},{}\n",
                correct[i],
                correct[i] as f64 / n as f64
            ));
            resources.push_str(&format!(
                "{seed},{name},{},{},{n},{},{}\n",
                r.train_ns[i], r.query_ns[i], r.active_edges, r.payload_bytes
            ));
        }
    }
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("resources.csv"), resources)?;
    fs::write(dir.join("summary.md"),format!("# Independent-code sensitivity control\n\n256-unit graphs receive original noisy prefix bits plus independent extra encoded bits and independent corruption. This supplies additional noisy evidence relative to the shard router.\n\n| Model | Correct | Trials | Accuracy |\n|---|---:|---:|---:|\n| independent_hebb | {} | {} | {:.4} |\n| independent_storkey | {} | {} | {:.4} |\n",totals[0],n_total,totals[0] as f64/n_total as f64,totals[1],n_total,totals[1] as f64/n_total as f64))?;
    Ok(())
}
pub fn execute(c: &Config, dir: &Path) -> io::Result<()> {
    if c.patterns == 0 || c.patterns > 64 || c.patterns % sharded::SHARDS != 0 || c.seeds.is_empty()
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "pattern count must be positive, at most 64, and divisible by four; seeds required",
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

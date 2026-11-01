//! Compare four 128-unit graphs to one graph with matching dense edge storage.
use crate::{sharded, Config};
use murmur_local_graph::{
    sequence::{Codebook, BITS, POSITIONS, SYMBOLS},
    Graph,
};
use std::{fs, io, path::Path, time::Instant};
pub const LARGE_BITS: usize = BITS * 2;
pub const LARGE_N: usize = POSITIONS * LARGE_BITS;
/// Duplicate each block without adding new input information.
pub fn duplicate_blocks(input: &[i8]) -> Vec<i8> {
    assert_eq!(input.len() % BITS, 0);
    let mut result = Vec::with_capacity(input.len() * 2);
    for block in input.chunks_exact(BITS) {
        result.extend_from_slice(block);
        result.extend_from_slice(block);
    }
    result
}
pub fn decode_large(book: &Codebook, output: &[i8]) -> usize {
    assert_eq!(output.len(), LARGE_BITS);
    let mut best = 0;
    let mut min = usize::MAX;
    for symbol in 0..SYMBOLS {
        let code = &book.codes[POSITIONS - 1][symbol];
        let d = output
            .iter()
            .enumerate()
            .filter(|(i, v)| **v != code[*i % BITS])
            .count();
        if d < min {
            best = symbol;
            min = d;
        }
    }
    best
}
pub fn predict_large(
    g: &Graph,
    observed: &[i8],
    missing: &[i8],
    book: &Codebook,
    sweeps: usize,
) -> usize {
    assert_eq!(g.n, LARGE_N);
    assert_eq!(observed.len(), (POSITIONS - 1) * BITS);
    assert_eq!(missing.len(), BITS);
    let mut state = duplicate_blocks(observed);
    state.extend(duplicate_blocks(missing));
    let mut clamped = vec![true; LARGE_N];
    clamped[(POSITIONS - 1) * LARGE_BITS..].fill(false);
    for _ in 0..sweeps {
        if g.sweep_clamped(&mut state, &clamped) == 0 {
            break;
        }
    }
    decode_large(book, &state[(POSITIONS - 1) * LARGE_BITS..])
}
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub source: sharded::Row,
    pub large_hebb: usize,
    pub large_storkey: usize,
}
#[derive(Debug)]
pub struct SeedRun {
    pub stored: Vec<[usize; POSITIONS]>,
    pub rows: Vec<Row>,
    pub shard_resources: sharded::Resources,
    pub train_ns: [u128; 2],
    pub query_ns: [u128; 2],
    pub large_bytes: usize,
    pub large_edges: usize,
}
pub fn run_seed(c: &Config, seed: u64) -> Result<SeedRun, &'static str> {
    let base = sharded::run_seed(c, seed)?;
    let book = Codebook::new(seed.wrapping_add(1000));
    let mut large_h = Graph::new(LARGE_N, c.density, seed.wrapping_add(9000));
    let mut large_s = large_h.clone();
    let start = Instant::now();
    for seq in &base.stored {
        large_h.learn(&duplicate_blocks(&book.encode(seq)));
    }
    let h_train = start.elapsed().as_nanos();
    let start = Instant::now();
    for seq in &base.stored {
        large_s.learn_storkey(&duplicate_blocks(&book.encode(seq)));
    }
    let s_train = start.elapsed().as_nanos();
    let mut elapsed = [0u128; 2];
    let mut rows = Vec::new();
    for source in base.rows {
        let start = Instant::now();
        let large_hebb =
            predict_large(&large_h, &source.observed, &source.missing, &book, c.sweeps);
        elapsed[0] += start.elapsed().as_nanos();
        let start = Instant::now();
        let large_storkey =
            predict_large(&large_s, &source.observed, &source.missing, &book, c.sweeps);
        elapsed[1] += start.elapsed().as_nanos();
        rows.push(Row {
            source,
            large_hebb,
            large_storkey,
        });
    }
    let large_bytes = large_h.weights.capacity() * std::mem::size_of::<f64>()
        + large_h.edges.capacity() * std::mem::size_of::<bool>();
    let large_edges = large_h.edges.iter().filter(|e| **e).count() / 2;
    Ok(SeedRun {
        stored: base.stored,
        rows,
        shard_resources: base.resources,
        train_ns: [h_train, s_train],
        query_ns: elapsed,
        large_bytes,
        large_edges,
    })
}
fn bits(x: &[i8]) -> String {
    x.iter().map(|x| if *x == 1 { '1' } else { '0' }).collect()
}
fn outputs(c: &Config, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),format!("runner_version={}\nos={}\narch={}\nlarge_units=256\nshards=4\nshard_units=128\nlarge_encoding=duplicate each 32-bit block, including corrupted prefix and initial missing bits\nlarge_topology_seed=s+9000 (wrapping u64)\nshard_assignment=sequence_index_mod_4\nrouter=nearest stored prefix, first on ties\nsplit=training sequences reused; evaluation uses fresh corruptions\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n",env!("CARGO_PKG_VERSION"),std::env::consts::OS,std::env::consts::ARCH))?;
    let mut training =
        String::from("seed,sequence_index,symbol_0,symbol_1,symbol_2,symbol_3,shard\n");
    let mut predictions=String::from("seed,sequence_index,trial,observed_prefix_bits,initial_missing_bits,target,true_shard,routed_shard,one_hebb,large_hebb,oracle_hebb,routed_hebb,one_storkey,large_storkey,oracle_storkey,routed_storkey,lookup\n");
    let mut metrics = String::from("seed,metric,correct,trials,accuracy\n");
    let mut resources=String::from("seed,model,training_ns,prediction_ns,trials,graph_active_undirected_edges,model_owned_payload_bytes\n");
    let names = [
        "one_hebb",
        "large_hebb",
        "oracle_hebb",
        "routed_hebb",
        "one_storkey",
        "large_storkey",
        "oracle_storkey",
        "routed_storkey",
        "lookup",
        "routing_correct",
    ];
    let mut total = 0;
    let mut totals = [0usize; 10];
    for &seed in &c.seeds {
        let r = run_seed(c, seed).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        for (i, s) in r.stored.iter().enumerate() {
            training.push_str(&format!(
                "{seed},{i},{},{},{},{},{}\n",
                s[0],
                s[1],
                s[2],
                s[3],
                i % sharded::SHARDS
            ));
        }
        let mut correct = [0usize; 10];
        for row in &r.rows {
            let p = &row.source;
            for (i, v) in [
                p.one_hebb,
                row.large_hebb,
                p.oracle_hebb,
                p.routed_hebb,
                p.one_storkey,
                row.large_storkey,
                p.oracle_storkey,
                p.routed_storkey,
                p.lookup,
            ]
            .iter()
            .enumerate()
            {
                correct[i] += usize::from(*v == p.target);
            }
            correct[9] += usize::from(p.true_shard == p.routed_shard);
            predictions.push_str(&format!(
                "{seed},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                p.sequence_index,
                p.trial,
                bits(&p.observed),
                bits(&p.missing),
                p.target,
                p.true_shard,
                p.routed_shard,
                p.one_hebb,
                row.large_hebb,
                p.oracle_hebb,
                p.routed_hebb,
                p.one_storkey,
                row.large_storkey,
                p.oracle_storkey,
                p.routed_storkey,
                p.lookup
            ));
        }
        let n = r.rows.len();
        total += n;
        for i in 0..10 {
            totals[i] += correct[i];
            metrics.push_str(&format!(
                "{seed},{},{},{},{}\n",
                names[i],
                correct[i],
                n,
                correct[i] as f64 / n as f64
            ));
        }
        let q = r.shard_resources.query_ns;
        let t = r.shard_resources.train_ns;
        let one = r.shard_resources.one_graph_bytes;
        let four = r.shard_resources.four_graph_bytes;
        let router = r.shard_resources.router_bytes;
        let e1 = r.shard_resources.one_active_edges;
        let e4 = r.shard_resources.four_active_edges;
        for (name, train, query, edges, payload) in [
            ("one_hebb", Some(t[0]), q[0], e1, one),
            (
                "large_hebb",
                Some(r.train_ns[0]),
                r.query_ns[0],
                r.large_edges,
                r.large_bytes,
            ),
            ("oracle_hebb", Some(t[2]), q[1], e4, four),
            ("routed_hebb", Some(t[2]), q[2] + q[6], e4, four + router),
            ("one_storkey", Some(t[1]), q[3], e1, one),
            (
                "large_storkey",
                Some(r.train_ns[1]),
                r.query_ns[1],
                r.large_edges,
                r.large_bytes,
            ),
            ("oracle_storkey", Some(t[3]), q[4], e4, four),
            ("routed_storkey", Some(t[3]), q[5] + q[6], e4, four + router),
            ("lookup", None, q[7], 0, r.shard_resources.lookup_bytes),
        ] {
            resources.push_str(&format!(
                "{seed},{name},{},{query},{n},{edges},{payload}\n",
                train.map(|x| x.to_string()).unwrap_or_default()
            ));
        }
    }
    fs::write(dir.join("training.csv"), training)?;
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("resources.csv"), resources)?;
    let mut summary=format!("# Matched connection-storage comparison\n\n{} seeds; {} sequences; noise {}; same observed data. The 256-unit graph receives duplicated 32-bit input blocks. Four 128-unit graphs have similar edge-array payload; the router adds storage.\n\n| Metric | Correct | Trials | Accuracy |\n|---|---:|---:|---:|\n",c.seeds.len(),c.patterns,c.prefix_noise);
    for i in 0..10 {
        summary.push_str(&format!(
            "| {} | {} | {} | {:.4} |\n",
            names[i],
            totals[i],
            total,
            totals[i] as f64 / total as f64
        ));
    }
    summary.push_str("\nEvaluation corrupts stored training sequences; it does not test unseen-sequence generalization. Oracle uses true shard. Router searches stored prefixes, as does lookup. See protocol and experiment notes for resource scope.\n");
    fs::write(dir.join("summary.md"), summary)?;
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

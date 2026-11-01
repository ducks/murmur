//! Experiment: one graph versus four small, independently seeded graphs.
//! Routing is exact nearest-neighbor search over stored prefixes, not learned.
use crate::Config;
use murmur_local_graph::{
    sequence::{graph_predict, sequences, Codebook, BITS, N, POSITIONS},
    Graph, Rng,
};
use std::{fs, io, path::Path, time::Instant};
pub const SHARDS: usize = 4;
#[derive(Clone, Debug, PartialEq)]
pub struct RouteEntry {
    pub prefix: [usize; POSITIONS - 1],
    pub shard: usize,
}
/// Routing stores only prefixes and shard IDs. It cannot inspect target symbols.
pub fn make_index(stored: &[[usize; POSITIONS]]) -> Vec<RouteEntry> {
    stored
        .iter()
        .enumerate()
        .map(|(i, s)| RouteEntry {
            prefix: [s[0], s[1], s[2]],
            shard: i % SHARDS,
        })
        .collect()
}
/// Finds the nearest prefix; stable first-entry tie break matches lookup.
pub fn route(observed: &[i8], index: &[RouteEntry], book: &Codebook) -> usize {
    assert_eq!(observed.len(), (POSITIONS - 1) * BITS);
    assert!(!index.is_empty());
    let mut best = 0;
    let mut best_distance = usize::MAX;
    for (i, entry) in index.iter().enumerate() {
        let mut distance = 0;
        for pos in 0..POSITIONS - 1 {
            let code = &book.codes[pos][entry.prefix[pos]];
            distance += observed[pos * BITS..(pos + 1) * BITS]
                .iter()
                .zip(code)
                .filter(|(a, b)| a != b)
                .count();
        }
        if distance < best_distance {
            best = i;
            best_distance = distance;
        }
    }
    best
}
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub seed: u64,
    pub sequence_index: usize,
    pub trial: usize,
    pub observed: Vec<i8>,
    pub missing: Vec<i8>,
    pub target: usize,
    pub true_shard: usize,
    pub routed_shard: usize,
    pub one_hebb: usize,
    pub oracle_hebb: usize,
    pub routed_hebb: usize,
    pub one_storkey: usize,
    pub oracle_storkey: usize,
    pub routed_storkey: usize,
    pub lookup: usize,
}
#[derive(Debug)]
pub struct Resources {
    pub train_ns: [u128; 4],
    pub query_ns: [u128; 8],
    pub one_graph_bytes: usize,
    pub four_graph_bytes: usize,
    pub router_bytes: usize,
    pub lookup_bytes: usize,
    pub one_active_edges: usize,
    pub four_active_edges: usize,
}
#[derive(Debug)]
pub struct SeedRun {
    pub stored: Vec<[usize; POSITIONS]>,
    pub rows: Vec<Row>,
    pub resources: Resources,
}
fn bytes(g: &Graph) -> usize {
    g.weights.capacity() * std::mem::size_of::<f64>()
        + g.edges.capacity() * std::mem::size_of::<bool>()
}
fn train(
    graphs: &mut [Graph],
    stored: &[[usize; POSITIONS]],
    book: &Codebook,
    storkey: bool,
    partitioned: bool,
) -> u128 {
    let t = Instant::now();
    for (i, seq) in stored.iter().enumerate() {
        let target = if partitioned { i % SHARDS } else { 0 };
        let p = book.encode(seq);
        if storkey {
            graphs[target].learn_storkey(&p);
        } else {
            graphs[target].learn(&p);
        }
    }
    t.elapsed().as_nanos()
}
fn timed_predict(
    g: &Graph,
    obs: &[i8],
    missing: &[i8],
    book: &Codebook,
    sweeps: usize,
    total: &mut u128,
) -> usize {
    let t = Instant::now();
    let result = graph_predict(g, obs, missing, book, sweeps);
    *total += t.elapsed().as_nanos();
    result
}
pub fn run_seed(c: &Config, seed: u64) -> Result<SeedRun, &'static str> {
    if c.patterns == 0
        || c.patterns % SHARDS != 0
        || c.patterns > 64
        || c.trials_per_sequence == 0
        || c.sweeps == 0
        || !c.density.is_finite()
        || !(0.0..=1.0).contains(&c.density)
        || !c.prefix_noise.is_finite()
        || !(0.0..=1.0).contains(&c.prefix_noise)
    {
        return Err("sharded experiment requires a valid config with a positive pattern count divisible by four");
    }
    let book = Codebook::new(seed.wrapping_add(1000));
    let stored = sequences(seed.wrapping_add(2000), c.patterns);
    let index = make_index(&stored);
    let mut one_h = vec![Graph::new(N, c.density, seed)];
    let mut one_s = one_h.clone();
    let mut shard_h: Vec<_> = (0..SHARDS)
        .map(|i| Graph::new(N, c.density, seed.wrapping_add(4000 + i as u64)))
        .collect();
    let mut shard_s = shard_h.clone();
    let train_ns = [
        train(&mut one_h, &stored, &book, false, false),
        train(&mut one_s, &stored, &book, true, false),
        train(&mut shard_h, &stored, &book, false, true),
        train(&mut shard_s, &stored, &book, true, true),
    ];
    // Query-time order is fixed and timing includes timer/allocation overhead.
    // 0 single H, 1 oracle H, 2 routed H, 3 single S, 4 oracle S,
    // 5 routed S, 6 router search, 7 independent direct lookup search.
    let mut query_ns = [0u128; 8];
    let mut rng = Rng(seed.wrapping_add(3000));
    let mut rows = Vec::new();
    for (sequence_index, seq) in stored.iter().enumerate() {
        for trial in 0..c.trials_per_sequence {
            let mut observed = book.encode(seq)[..(POSITIONS - 1) * BITS].to_vec();
            for bit in &mut observed {
                if rng.uniform() < c.prefix_noise {
                    *bit = -*bit;
                }
            }
            let missing: Vec<i8> = (0..BITS)
                .map(|_| if rng.next_u64() & 1 == 0 { -1 } else { 1 })
                .collect();
            let true_shard = sequence_index % SHARDS;
            let start = Instant::now();
            let matched = route(&observed, &index, &book);
            let routed_shard = index[matched].shard;
            query_ns[6] += start.elapsed().as_nanos();
            let start = Instant::now();
            let lookup = stored[route(&observed, &index, &book)][POSITIONS - 1];
            query_ns[7] += start.elapsed().as_nanos();
            let one_hebb = timed_predict(
                &one_h[0],
                &observed,
                &missing,
                &book,
                c.sweeps,
                &mut query_ns[0],
            );
            let oracle_hebb = timed_predict(
                &shard_h[true_shard],
                &observed,
                &missing,
                &book,
                c.sweeps,
                &mut query_ns[1],
            );
            let routed_hebb = timed_predict(
                &shard_h[routed_shard],
                &observed,
                &missing,
                &book,
                c.sweeps,
                &mut query_ns[2],
            );
            let one_storkey = timed_predict(
                &one_s[0],
                &observed,
                &missing,
                &book,
                c.sweeps,
                &mut query_ns[3],
            );
            let oracle_storkey = timed_predict(
                &shard_s[true_shard],
                &observed,
                &missing,
                &book,
                c.sweeps,
                &mut query_ns[4],
            );
            let routed_storkey = timed_predict(
                &shard_s[routed_shard],
                &observed,
                &missing,
                &book,
                c.sweeps,
                &mut query_ns[5],
            );
            rows.push(Row {
                seed,
                sequence_index,
                trial,
                observed,
                missing,
                target: seq[POSITIONS - 1],
                true_shard,
                routed_shard,
                one_hebb,
                oracle_hebb,
                routed_hebb,
                one_storkey,
                oracle_storkey,
                routed_storkey,
                lookup,
            });
        }
    }
    let resources = Resources {
        train_ns,
        query_ns,
        one_graph_bytes: bytes(&one_h[0]),
        four_graph_bytes: shard_h.iter().map(bytes).sum(),
        router_bytes: index.capacity() * std::mem::size_of::<RouteEntry>(),
        lookup_bytes: stored.capacity() * std::mem::size_of::<[usize; POSITIONS]>(),
        one_active_edges: one_h[0].edges.iter().filter(|e| **e).count() / 2,
        four_active_edges: shard_h
            .iter()
            .map(|g| g.edges.iter().filter(|e| **e).count() / 2)
            .sum(),
    };
    Ok(SeedRun {
        stored,
        rows,
        resources,
    })
}
fn bits(x: &[i8]) -> String {
    x.iter().map(|x| if *x == 1 { '1' } else { '0' }).collect()
}
fn outputs(c: &Config, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),format!("runner_version={}\nos={}\narch={}\nshards=4\nassignment=sequence_index_mod_4\nseed_scheme=single:s;shard:s+4000+shard;codebook:s+1000;train:s+2000;test_cues:s+3000 (wrapping u64)\nrouter=nearest stored prefix by Hamming distance; first index on ties\nsplit=stored sequences reused; evaluation uses fresh corruptions\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n",env!("CARGO_PKG_VERSION"),std::env::consts::OS,std::env::consts::ARCH))?;
    let mut predictions=String::from("seed,sequence_index,trial,observed_prefix_bits,initial_missing_bits,target,true_shard,routed_shard,one_hebb,oracle_hebb,routed_hebb,one_storkey,oracle_storkey,routed_storkey,lookup\n");
    let mut training =
        String::from("seed,sequence_index,symbol_0,symbol_1,symbol_2,symbol_3,shard\n");
    let mut metrics = String::from("seed,metric,correct,trials,accuracy\n");
    let mut resources =
        String::from("seed,model,train_ns,prediction_ns,trials,model_owned_payload_bytes\n");
    let names = [
        "one_hebb",
        "oracle_hebb",
        "routed_hebb",
        "one_storkey",
        "oracle_storkey",
        "routed_storkey",
        "lookup",
        "routing_correct",
    ];
    let mut totals = [0usize; 8];
    let mut total = 0;
    for &seed in &c.seeds {
        let run = run_seed(c, seed).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
        for (i, s) in run.stored.iter().enumerate() {
            training.push_str(&format!(
                "{seed},{i},{},{},{},{},{}\n",
                s[0],
                s[1],
                s[2],
                s[3],
                i % SHARDS
            ));
        }
        let mut correct = [0usize; 8];
        for r in &run.rows {
            for (i, p) in [
                r.one_hebb,
                r.oracle_hebb,
                r.routed_hebb,
                r.one_storkey,
                r.oracle_storkey,
                r.routed_storkey,
                r.lookup,
            ]
            .iter()
            .enumerate()
            {
                correct[i] += usize::from(*p == r.target);
            }
            correct[7] += usize::from(r.true_shard == r.routed_shard);
            predictions.push_str(&format!(
                "{seed},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                r.sequence_index,
                r.trial,
                bits(&r.observed),
                bits(&r.missing),
                r.target,
                r.true_shard,
                r.routed_shard,
                r.one_hebb,
                r.oracle_hebb,
                r.routed_hebb,
                r.one_storkey,
                r.oracle_storkey,
                r.routed_storkey,
                r.lookup
            ));
        }
        let n = run.rows.len();
        total += n;
        for i in 0..8 {
            totals[i] += correct[i];
            metrics.push_str(&format!(
                "{seed},{},{},{},{}\n",
                names[i],
                correct[i],
                n,
                correct[i] as f64 / n as f64
            ));
        }
        let q = run.resources.query_ns;
        let t = run.resources.train_ns;
        let b = run.resources.one_graph_bytes;
        let four = run.resources.four_graph_bytes;
        let route_bytes = run.resources.router_bytes;
        for (name, train, query, payload) in [
            ("one_hebb", Some(t[0]), q[0], b),
            ("oracle_hebb", Some(t[2]), q[1], four),
            ("routed_hebb", Some(t[2]), q[2] + q[6], four + route_bytes),
            ("one_storkey", Some(t[1]), q[3], b),
            ("oracle_storkey", Some(t[3]), q[4], four),
            (
                "routed_storkey",
                Some(t[3]),
                q[5] + q[6],
                four + route_bytes,
            ),
            ("router", None, q[6], route_bytes),
            ("lookup", None, q[7], run.resources.lookup_bytes),
        ] {
            resources.push_str(&format!(
                "{seed},{name},{},{query},{n},{payload}\n",
                train.map(|x| x.to_string()).unwrap_or_default()
            ));
        }
    }
    fs::write(dir.join("training.csv"), training)?;
    fs::write(dir.join("predictions.csv"), predictions)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("resources.csv"), resources)?;
    let mut summary=format!("# Sharded memory experiment\n\n{} seeds; {} sequences; {} shards, {} sequences per shard; prefix noise {}. Evaluation queries corrupt stored training sequences, not unseen sequences.\n\n| Metric | Correct | Trials | Accuracy |\n|---|---:|---:|---:|\n",c.seeds.len(),c.patterns,SHARDS,c.patterns/SHARDS,c.prefix_noise);
    for i in 0..8 {
        summary.push_str(&format!(
            "| {} | {} | {} | {:.4} |\n",
            names[i],
            totals[i],
            total,
            totals[i] as f64 / total as f64
        ));
    }
    summary.push_str("\nOracle routing uses the true training assignment. The actual router searches stored prefixes; direct lookup uses the same search and then reads the stored target. Costs are not budget matched; see protocol.\n");
    fs::write(dir.join("summary.md"), summary)?;
    Ok(())
}
/// Validates before creating the destination; never overwrites an existing run.
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

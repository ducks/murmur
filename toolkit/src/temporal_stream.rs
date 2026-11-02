//! Continuous delayed-bit recall. The graph is a fixed random reservoir; only
//! the supervised readout learns. It is not the symmetric-feedback model.
use murmur::{data::Rng, graph::Graph, learning::Readout};
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    fs, io,
    path::Path,
};
#[derive(Clone, Debug, PartialEq)]
pub struct StreamConfig {
    pub seeds: Vec<u64>,
    pub delays: Vec<usize>,
    pub train_tokens: usize,
    pub test_tokens: usize,
    pub units: usize,
    pub degree: usize,
    pub learning_rate: f64,
}
impl StreamConfig {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut map = BTreeMap::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("line {}: expected key=value", i + 1))?;
            let (key, value) = (key.trim(), value.trim());
            if ![
                "schema_version",
                "seeds",
                "delays",
                "train_tokens",
                "test_tokens",
                "units",
                "degree",
                "learning_rate",
            ]
            .contains(&key)
            {
                return Err(format!("unknown key: {key}"));
            }
            if map.insert(key, value).is_some() {
                return Err(format!("duplicate key: {key}"));
            }
        }
        let get = |key| {
            map.get(key)
                .copied()
                .ok_or_else(|| format!("missing key: {key}"))
        };
        if get("schema_version")? != "1" {
            return Err("unsupported schema_version".into());
        }
        let seeds: Vec<u64> = get("seeds")?
            .split(',')
            .map(|x| x.trim().parse().map_err(|_| "invalid seed".to_string()))
            .collect::<Result<_, _>>()?;
        let delays: Vec<usize> = get("delays")?
            .split(',')
            .map(|x| x.trim().parse().map_err(|_| "invalid delay".to_string()))
            .collect::<Result<_, _>>()?;
        let train_tokens = get("train_tokens")?
            .parse::<usize>()
            .map_err(|_| "invalid train_tokens")?;
        let test_tokens = get("test_tokens")?
            .parse::<usize>()
            .map_err(|_| "invalid test_tokens")?;
        let units = get("units")?
            .parse::<usize>()
            .map_err(|_| "invalid units")?;
        let degree = get("degree")?
            .parse::<usize>()
            .map_err(|_| "invalid degree")?;
        let learning_rate = get("learning_rate")?
            .parse::<f64>()
            .map_err(|_| "invalid learning_rate")?;
        let mut u = seeds.clone();
        u.sort_unstable();
        u.dedup();
        if seeds.is_empty() || u.len() != seeds.len() {
            return Err("seeds must be nonempty and unique".into());
        }
        let train_seeds: HashSet<_> = seeds.iter().map(|s| s.wrapping_add(31000)).collect();
        if seeds
            .iter()
            .any(|s| train_seeds.contains(&s.wrapping_add(32000)))
        {
            return Err("training and test stream seeds overlap across runs".into());
        }
        let mut d = delays.clone();
        d.sort_unstable();
        d.dedup();
        if delays.is_empty()
            || d.len() != delays.len()
            || delays
                .iter()
                .any(|x| *x == 0 || *x >= train_tokens || *x >= test_tokens)
        {
            return Err("delays must be unique, positive and shorter than both streams".into());
        }
        if units == 0 || degree == 0 || !learning_rate.is_finite() || learning_rate <= 0.0 {
            return Err("units, degree and learning rate must be positive".into());
        }
        Ok(Self {
            seeds,
            delays,
            train_tokens,
            test_tokens,
            units,
            degree,
            learning_rate,
        })
    }
    pub fn resolved(&self) -> String {
        format!("schema_version=1\nseeds={}\ndelays={}\ntrain_tokens={}\ntest_tokens={}\nunits={}\ndegree={}\nlearning_rate={}\n",self.seeds.iter().map(u64::to_string).collect::<Vec<_>>().join(","),self.delays.iter().map(usize::to_string).collect::<Vec<_>>().join(","),self.train_tokens,self.test_tokens,self.units,self.degree,self.learning_rate)
    }
}
pub fn stream(seed: u64, n: usize) -> Vec<usize> {
    let mut rng = Rng(seed);
    (0..n).map(|_| (rng.next_u64() & 1) as usize).collect()
}
/// Exact causal reference: retain only the last `delay` input symbols.
pub struct ShiftRegister {
    delay: usize,
    history: VecDeque<usize>,
}
impl ShiftRegister {
    pub fn new(delay: usize) -> Self {
        assert!(delay > 0);
        Self {
            delay,
            history: VecDeque::with_capacity(delay + 1),
        }
    }
    pub fn observe(&mut self, input: usize) -> Option<usize> {
        self.history.push_back(input);
        if self.history.len() > self.delay {
            self.history.pop_front()
        } else {
            None
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub index: usize,
    pub input: usize,
    pub target: usize,
    pub graph: usize,
    pub edge_free: usize,
    pub reset: usize,
    pub shift: usize,
    pub majority: usize,
}
#[derive(Debug)]
pub struct Run {
    pub rows: Vec<Row>,
    pub graph_edges: usize,
    pub readout_parameters: usize,
    pub train_counts: [usize; 2],
}
/// Predict x[t-delay] after consuming x[t]. State resets at the train/test boundary.
pub fn run_delay(c: &StreamConfig, seed: u64, delay: usize) -> Run {
    assert!(delay > 0 && delay < c.train_tokens && delay < c.test_tokens);
    let training = stream(seed.wrapping_add(31000), c.train_tokens);
    let testing = stream(seed.wrapping_add(32000), c.test_tokens);
    let mut graph = Graph::new(c.units, c.degree, seed);
    let mut edge_free = graph.clone();
    edge_free.edges.clear();
    let mut reset = graph.clone();
    let mut graph_reader = Readout::new(c.units, c.learning_rate);
    let mut edge_reader = Readout::new(c.units, c.learning_rate);
    let mut reset_reader = Readout::new(c.units, c.learning_rate);
    let mut counts = [0usize; 2];
    for (t, &input) in training.iter().enumerate() {
        graph.step(input);
        edge_free.step(input);
        reset.reset();
        reset.step(input);
        if t >= delay {
            let target = training[t - delay];
            graph_reader.learn(&graph.state, target);
            edge_reader.learn(&edge_free.state, target);
            reset_reader.learn(&reset.state, target);
            counts[target] += 1;
        }
    }
    let majority = usize::from(counts[1] > counts[0]);
    graph.reset();
    edge_free.reset();
    reset.reset();
    let mut rows = Vec::with_capacity(c.test_tokens - delay);
    let mut shift_register = ShiftRegister::new(delay);
    for (t, &input) in testing.iter().enumerate() {
        graph.step(input);
        edge_free.step(input);
        reset.reset();
        reset.step(input);
        let shift_prediction = shift_register.observe(input);
        if t >= delay {
            rows.push(Row {
                index: t,
                input,
                target: testing[t - delay],
                graph: graph_reader.predict(&graph.state),
                edge_free: edge_reader.predict(&edge_free.state),
                reset: reset_reader.predict(&reset.state),
                shift: shift_prediction.expect("shift register warm-up must match delay"),
                majority,
            });
        }
    }
    Run {
        rows,
        graph_edges: graph.edges.len(),
        readout_parameters: 2 * (c.units + 1),
        train_counts: counts,
    }
}
fn outputs(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("protocol.txt"),format!("runner_version={}\nmodel=fixed leaky random recurrent reservoir with trained logistic readout\nleak={}\ntask=predict x[t-delay] after observing x[t]\ntraining_stream_seed=s+31000\ntest_stream_seed=s+32000\nreset=train/test boundary, and before each token for reset ablation\nsplit=separate deterministic streams; no formal independence guarantee\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n",env!("CARGO_PKG_VERSION"),Graph::new(c.units,c.degree,c.seeds[0]).leak))?;
    let mut streams = String::from("seed,split,index,symbol\n");
    let mut pred =
        String::from("seed,delay,index,input,target,graph,edge_free,reset,shift,majority\n");
    let mut metrics = String::from(
        "seed,delay,model,correct,trials,accuracy,graph_edges,trainable_readout_parameters\n",
    );
    let mut summary=String::from("# Continuous delayed-bit recall\n\nPredict the bit from `delay` steps ago after observing the current bit. Models receive the same streams; evaluation resets state and freezes readouts. Each delay is trained separately.\n\n| Delay | Graph | Edge-free | Reset state | Shift register | Majority |\n|---:|---:|---:|---:|---:|---:|\n");
    let mut totals = vec![[0usize; 5]; c.delays.len()];
    let mut counts = vec![0usize; c.delays.len()];
    for &seed in &c.seeds {
        for (split, offset, n) in [
            ("train", 31000, c.train_tokens),
            ("test", 32000, c.test_tokens),
        ] {
            for (i, symbol) in stream(seed.wrapping_add(offset), n).into_iter().enumerate() {
                streams.push_str(&format!("{seed},{split},{i},{symbol}\n"));
            }
        }
        for (didx, &delay) in c.delays.iter().enumerate() {
            let run = run_delay(c, seed, delay);
            let mut correct = [0usize; 5];
            for row in &run.rows {
                for (k, p) in [row.graph, row.edge_free, row.reset, row.shift, row.majority]
                    .iter()
                    .enumerate()
                {
                    correct[k] += usize::from(*p == row.target);
                }
                pred.push_str(&format!(
                    "{seed},{delay},{},{},{},{},{},{},{},{}\n",
                    row.index,
                    row.input,
                    row.target,
                    row.graph,
                    row.edge_free,
                    row.reset,
                    row.shift,
                    row.majority
                ));
            }
            let n = run.rows.len();
            counts[didx] += n;
            for (k, name) in ["graph", "edge_free", "reset", "shift", "majority"]
                .iter()
                .enumerate()
            {
                totals[didx][k] += correct[k];
                let edges = match *name {
                    "graph" | "reset" => run.graph_edges,
                    _ => 0,
                };
                let params = if k < 3 { run.readout_parameters } else { 0 };
                metrics.push_str(&format!(
                    "{seed},{delay},{name},{},{n},{},{edges},{params}\n",
                    correct[k],
                    correct[k] as f64 / n as f64
                ));
            }
        }
    }
    for (i, delay) in c.delays.iter().enumerate() {
        let n = counts[i] as f64;
        summary.push_str(&format!(
            "| {delay} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            totals[i][0] as f64 / n,
            totals[i][1] as f64 / n,
            totals[i][2] as f64 / n,
            totals[i][3] as f64 / n,
            totals[i][4] as f64 / n
        ));
    }
    summary.push_str("\nThe shift register stores explicit input history and is not parameter- or compute-matched. The recurrent graph is fixed; only its readout learns. Separate seeded random streams are not a proof of independence.\n");
    fs::write(dir.join("streams.csv"), streams)?;
    fs::write(dir.join("predictions.csv"), pred)?;
    fs::write(dir.join("metrics.csv"), metrics)?;
    fs::write(dir.join("summary.md"), summary)?;
    Ok(())
}
pub fn execute(c: &StreamConfig, dir: &Path) -> io::Result<()> {
    // Revalidate public structs before creating the destination.
    StreamConfig::parse(&c.resolved())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
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

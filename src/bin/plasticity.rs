use murmur::{
    experiment::Config,
    graph::Graph,
    plasticity::{run_plasticity, EDGE_RATE, TRACE_DECAY, WEIGHT_LIMIT},
};
use std::{env, fs, path::Path};

fn save_graph(path: &Path, graph: &Graph) -> std::io::Result<()> {
    let mut data = String::from("from,to,weight\n");
    for e in &graph.edges {
        data.push_str(&format!("{},{},{:.17}\n", e.from, e.to, e.weight));
    }
    fs::write(path, data)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 3 {
        return Err("usage: plasticity [seed] [output-directory] [delay]".into());
    }
    let mut c = Config::default();
    if let Some(seed) = args.first() {
        c.seed = seed.parse()?;
    }
    if let Some(delay) = args.get(2) {
        c.delay = delay.parse()?;
    }
    if c.delay == 0 {
        return Err("delay must be positive".into());
    }
    let dir = Path::new(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("results/plasticity"),
    );
    fs::create_dir_all(dir)?;
    let (m, learned, shuffled, log) = run_plasticity(&c);
    let n = m.test_trials as f64;
    let json = format!("{{\n  \"seed\": {}, \"units\": {}, \"degree\": {}, \"delay\": {}, \"train_trials\": {}, \"test_trials\": {}, \"readout_rate\": {}, \"edge_rate\": {}, \"trace_decay\": {}, \"weight_limit\": {},\n  \"frozen_accuracy\": {}, \"learned_accuracy\": {}, \"shuffled_accuracy\": {}, \"learned_mean_abs_edge_change\": {}, \"shuffled_mean_abs_edge_change\": {}\n}}\n", c.seed,c.units,c.degree,c.delay,c.train_trials,c.test_trials,c.learning_rate,EDGE_RATE,TRACE_DECAY,WEIGHT_LIMIT,m.frozen_correct as f64/n,m.learned_correct as f64/n,m.shuffled_correct as f64/n,m.learned_mean_abs_change,m.shuffled_mean_abs_change);
    fs::write(dir.join("metrics.json"), &json)?;
    save_graph(
        &dir.join("initial_topology.csv"),
        &Graph::new(c.units, c.degree, c.seed),
    )?;
    save_graph(&dir.join("learned_topology.csv"), &learned)?;
    save_graph(&dir.join("shuffled_topology.csv"), &shuffled)?;
    let mut records =
        String::from("training_trial,learned_mean_abs_change,shuffled_mean_abs_change\n");
    for (trial, learned, shuffled) in log {
        records.push_str(&format!("{trial},{learned:.17},{shuffled:.17}\n"));
    }
    fs::write(dir.join("edge_changes.csv"), records)?;
    let mut diagnostics = String::from("model,cue_state_distance,cue0_margin,cue1_margin\n");
    for (name, d) in ["frozen", "learned", "shuffled"].iter().zip(m.diagnostics) {
        diagnostics.push_str(&format!(
            "{name},{:.17},{:.17},{:.17}\n",
            d.cue_state_distance, d.cue0_margin, d.cue1_margin
        ));
    }
    fs::write(dir.join("diagnostics.csv"), diagnostics)?;
    print!("{json}");
    Ok(())
}

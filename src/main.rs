use murmur::experiment::{run, Config};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut c = Config::default();
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() > 3 {
        return Err("usage: murmur [seed] [output-directory] [delay]".into());
    }
    if let Some(s) = args.first() {
        c.seed = s.parse()?;
    }
    let dir = args.get(1).map(String::as_str).unwrap_or("results");
    if let Some(delay) = args.get(2) {
        c.delay = delay.parse()?;
        if c.delay == 0 {
            return Err("delay must be positive".into());
        }
    }
    fs::create_dir_all(dir)?;
    let (m, graph, activations) = run(&c);
    let n = m.test_trials as f64;
    let json = format!("{{\n  \"seed\": {}, \"units\": {}, \"degree\": {}, \"delay\": {}, \"train_trials\": {}, \"test_trials\": {}, \"learning_rate\": {},\n  \"graph_accuracy\": {}, \"edge_free_accuracy\": {}, \"reset_accuracy\": {}, \"markov_accuracy\": {},\n  \"edges\": {}, \"trainable_parameters_per_readout\": {}\n}}\n", c.seed,c.units,c.degree,c.delay,c.train_trials,c.test_trials,c.learning_rate,m.graph_correct as f64/n,m.edge_free_correct as f64/n,m.reset_correct as f64/n,m.markov_correct as f64/n,graph.edges.len(),2*(c.units+1));
    fs::write(Path::new(dir).join("metrics.json"), &json)?;
    let mut topology = String::from("from,to,weight\n");
    for e in &graph.edges {
        topology.push_str(&format!("{},{},{:.17}\n", e.from, e.to, e.weight));
    }
    fs::write(Path::new(dir).join("topology.csv"), topology)?;
    let mut log = String::from("trial,cue,unit_0_after_marker\n");
    for a in activations {
        log.push_str(&format!("{},{},{:.17}\n", a[0], a[1], a[2]));
    }
    fs::write(Path::new(dir).join("activations.csv"), log)?;
    print!("{json}");
    Ok(())
}

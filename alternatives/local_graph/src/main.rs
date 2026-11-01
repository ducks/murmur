use murmur_local_graph::{evaluate, patterns, Graph};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: murmur-local-graph [output-directory]".into());
    }
    let dir = Path::new(args.first().map(String::as_str).unwrap_or("results"));
    fs::create_dir_all(dir)?;
    let mut csv=String::from("seed,n,patterns,density,noise,sweeps,trials,trained_exact,untrained_exact,trained_bit_accuracy,untrained_bit_accuracy,baseline_bit_accuracy,undirected_edges\n");
    for seed in 101..111 {
        for &k in &[2, 4, 8, 16] {
            let mut graph = Graph::new(128, 0.25, seed);
            let untrained = graph.clone();
            let stored = patterns(128, k, seed + 1000);
            for p in &stored {
                graph.learn(p);
            }
            let trained = evaluate(&graph, &stored, 0.2, 25, seed + 2000, 20);
            let control = evaluate(&untrained, &stored, 0.2, 25, seed + 2000, 20);
            let denom = (trained.trials * trained.n) as f64;
            let edge_count = graph.edges.iter().filter(|e| **e).count() / 2;
            csv.push_str(&format!(
                "{seed},128,{k},0.25,0.2,20,{},{},{},{},{},{},{}\n",
                trained.trials,
                trained.graph_exact as f64 / trained.trials as f64,
                control.graph_exact as f64 / control.trials as f64,
                trained.graph_correct_bits as f64 / denom,
                control.graph_correct_bits as f64 / denom,
                trained.baseline_correct_bits as f64 / denom,
                edge_count
            ));
        }
    }
    fs::write(dir.join("metrics.csv"), &csv)?;
    print!("{csv}");
    Ok(())
}

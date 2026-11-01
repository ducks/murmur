use murmur_local_graph::sequence::evaluate;
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: sequence [output-directory]".into());
    }
    let dir = Path::new(
        args.first()
            .map(String::as_str)
            .unwrap_or("results/sequence"),
    );
    fs::create_dir_all(dir)?;
    let mut csv=String::from("seed,units,positions,symbols,patterns,density,prefix_noise,trials_per_pattern,sweeps,trials,graph_accuracy,lookup_accuracy,untrained_accuracy,majority_accuracy\n");
    for seed in 201..211 {
        for k in [2, 4, 8, 16, 32] {
            for noise in [0.0, 0.1, 0.2] {
                let m = evaluate(seed, k, noise, 20, 20);
                let n = m.trials as f64;
                csv.push_str(&format!(
                    "{seed},128,4,4,{k},0.25,{noise},20,20,{},{},{},{},{}\n",
                    m.trials,
                    m.graph_correct as f64 / n,
                    m.lookup_correct as f64 / n,
                    m.untrained_correct as f64 / n,
                    m.majority_correct as f64 / n
                ));
            }
        }
    }
    fs::write(dir.join("metrics.csv"), &csv)?;
    print!("{csv}");
    Ok(())
}

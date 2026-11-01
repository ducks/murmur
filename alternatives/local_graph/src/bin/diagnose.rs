use murmur_local_graph::diagnostics::run;
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 1 {
        return Err("usage: diagnose [output-directory]".into());
    }
    let dir = Path::new(
        args.first()
            .map(String::as_str)
            .unwrap_or("results/diagnostics"),
    );
    fs::create_dir_all(dir)?;
    let mut csv=String::from("seed,units,patterns,density,prefix_noise,codebook,trials_per_sequence,trials,graph_accuracy,target_only_accuracy,no_feedback_accuracy,one_sweep_accuracy,lookup_accuracy,output_bit_accuracy,target_field_flip_fraction\n");
    for seed in 301..311 {
        for k in [2, 4, 8, 16, 32] {
            for noise in [0.0, 0.2] {
                for orthogonal in [false, true] {
                    let m = run(seed, k, noise, orthogonal, 10);
                    let n = m.trials as f64;
                    csv.push_str(&format!(
                        "{seed},128,{k},0.25,{noise},{},10,{},{},{},{},{},{},{},{}\n",
                        if orthogonal { "orthogonal" } else { "random" },
                        m.trials,
                        m.original as f64 / n,
                        m.target_only as f64 / n,
                        m.no_feedback as f64 / n,
                        m.one_sweep as f64 / n,
                        m.lookup as f64 / n,
                        m.target_bit_correct as f64 / m.bit_observations as f64,
                        m.flip_bits as f64 / m.margin_bits as f64
                    ));
                }
            }
        }
    }
    fs::write(dir.join("metrics.csv"), &csv)?;
    print!("{csv}");
    Ok(())
}

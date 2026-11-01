use murmur::{
    experiment::Config,
    probe::{run_probe, NOISE_LEVELS},
};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: probe [seed] [output-directory]".into());
    }
    let mut config = Config::default();
    if let Some(s) = args.first() {
        config.seed = s.parse()?;
    }
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("results/probe"));
    fs::create_dir_all(dir)?;
    let mut csv = String::from("seed,units,degree,train_trials,test_trials,learning_rate,delay,noise_half_width,state_distance,online_correct,probe_correct,online_accuracy,probe_accuracy\n");
    for delay in [4, 8, 12, 24] {
        config.delay = delay;
        for r in run_probe(&config, &NOISE_LEVELS)? {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{:.17},{},{},{},{}\n",
                config.seed,
                config.units,
                config.degree,
                config.train_trials,
                config.test_trials,
                config.learning_rate,
                r.delay,
                r.noise,
                r.state_distance,
                r.online_correct,
                r.probe_correct,
                r.online_correct as f64 / r.trials as f64,
                r.probe_correct as f64 / r.trials as f64
            ));
        }
    }
    fs::write(dir.join("metrics.csv"), &csv)?;
    print!("{csv}");
    Ok(())
}

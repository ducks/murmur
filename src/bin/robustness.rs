use murmur::{
    experiment::Config,
    robustness::{run, Task},
};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: robustness [seed] [output-directory]".into());
    }
    let mut c = Config::default();
    if let Some(s) = args.first() {
        c.seed = s.parse()?;
    }
    let dir = Path::new(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("results/robustness"),
    );
    fs::create_dir_all(dir)?;
    let mut csv = String::from("seed,units,degree,train_trials,test_trials,learning_rate,delay,varied,internal_noise_half_width,online_correct,probe_correct,online_accuracy,probe_accuracy\n");
    for delay in [4, 8, 12, 24] {
        c.delay = delay;
        for task in [
            Task {
                varied: false,
                noise: 0.0,
            },
            Task {
                varied: false,
                noise: 0.001,
            },
            Task {
                varied: false,
                noise: 0.01,
            },
            Task {
                varied: true,
                noise: 0.0,
            },
            Task {
                varied: true,
                noise: 0.001,
            },
            Task {
                varied: true,
                noise: 0.01,
            },
        ] {
            let m = run(&c, task)?;
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
                c.seed,
                c.units,
                c.degree,
                c.train_trials,
                c.test_trials,
                c.learning_rate,
                delay,
                task.varied,
                task.noise,
                m.online_correct,
                m.probe_correct,
                m.online_correct as f64 / m.test_trials as f64,
                m.probe_correct as f64 / m.test_trials as f64
            ));
        }
    }
    fs::write(dir.join("metrics.csv"), &csv)?;
    print!("{csv}");
    Ok(())
}

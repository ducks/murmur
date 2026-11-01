use murmur::trial_growth::run_scheduled;
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: sequential [seed] [output-directory]".into());
    }
    let seed = match args.first() {
        Some(s) => s.parse()?,
        None => 501u64,
    };
    let dir = Path::new(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("results/sequential"),
    );
    fs::create_dir_all(dir)?;
    let mut metrics=String::from("seed,case,switching,noise,block,train_steps,test_steps,base_penalty,scale,trials,completed,accepted,first_commit,test_accuracy,fixed_accuracy,single_accuracy\n");
    let mut trials =
        String::from("seed,case,scale,attempt,start,end,completed,gain,penalty,accepted\n");
    for scale in [0.0, 20.0, 40.0, 80.0] {
        for (name, switching, noise, block, train) in [
            ("switch100", true, 0.1, 100, 1200),
            ("switch200", true, 0.1, 200, 1200),
            ("stable_long", false, 0.1, 100, 5000),
            ("noisy_long", false, 0.25, 100, 5000),
        ] {
            let r = run_scheduled(seed, switching, noise, 40.0, train, 400, block, scale)?;
            let score = |k: usize| {
                r.test
                    .iter()
                    .filter(|e| [e.1, e.2, e.3][k] == e.0.target)
                    .count() as f64
                    / r.test.len() as f64
            };
            let first = r
                .trials
                .iter()
                .find(|t| t.accepted)
                .map(|t| t.end.to_string())
                .unwrap_or_default();
            metrics.push_str(&format!("{seed},{name},{switching},{noise},{block},{train},400,40,{scale},{},{},{},{first},{},{},{}\n",r.trials.len(),r.trials.iter().filter(|t|t.completed).count(),r.trials.iter().filter(|t|t.accepted).count(),score(0),score(1),score(2)));
            for (i, t) in r.trials.iter().enumerate() {
                trials.push_str(&format!(
                    "{seed},{name},{scale},{},{},{},{},{},{},{}\n",
                    i + 1,
                    t.start,
                    t.end,
                    t.completed,
                    t.gain,
                    t.penalty,
                    t.accepted
                ));
            }
        }
    }
    fs::write(dir.join("metrics.csv"), &metrics)?;
    fs::write(dir.join("trials.csv"), trials)?;
    print!("{metrics}");
    Ok(())
}

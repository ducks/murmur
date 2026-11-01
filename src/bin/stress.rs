use murmur::{stress::Cusum, trial_growth::run_custom};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: stress [seed] [output-directory]".into());
    }
    let seed = match args.first() {
        Some(s) => s.parse()?,
        None => 301u64,
    };
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("results/stress"));
    fs::create_dir_all(dir)?;
    let mut csv=String::from("seed,case,switching,noise,block,train_steps,test_steps,penalty,trials,accepted,first_commit,test_accuracy,fixed_accuracy,single_accuracy,cusum5,cusum10,cusum20,cusum40,cusum80\n");
    for (name, switching, noise, block, train) in [
        ("switch50", true, 0.1, 50, 1200),
        ("switch100", true, 0.1, 100, 1200),
        ("switch200", true, 0.1, 200, 1200),
        ("stable_short", false, 0.1, 100, 1200),
        ("stable_long", false, 0.1, 100, 5000),
        ("noisy_short", false, 0.25, 100, 1200),
        ("noisy_long", false, 0.25, 100, 5000),
    ] {
        let r = run_custom(seed, switching, noise, 40.0, train, 400, block)?;
        let mut alarms = [
            Cusum::new(5.0),
            Cusum::new(10.0),
            Cusum::new(20.0),
            Cusum::new(40.0),
            Cusum::new(80.0),
        ];
        for e in &r.trace {
            for a in &mut alarms {
                a.observe(e.source, e.target);
            }
        }
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
        let alarm = |i: usize| {
            alarms[i]
                .first_alarm
                .map(|x| x.to_string())
                .unwrap_or_default()
        };
        csv.push_str(&format!("{seed},{name},{switching},{noise},{block},{train},400,40,{},{},{},{},{},{},{},{},{},{},{}\n",r.trials.len(),r.trials.iter().filter(|t|t.accepted).count(),first,score(0),score(1),score(2),alarm(0),alarm(1),alarm(2),alarm(3),alarm(4)));
    }
    fs::write(dir.join("metrics.csv"), &csv)?;
    print!("{csv}");
    Ok(())
}

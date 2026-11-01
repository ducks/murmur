use murmur::sequence_engine::{run, EngineConfig};
use std::{env, fs, path::Path};
fn topology(weights: &[Vec<f64>]) -> String {
    let mut csv = String::from("from,to,weight\n");
    for (from, row) in weights.iter().enumerate() {
        for (to, w) in row.iter().enumerate() {
            csv.push_str(&format!("{from},{to},{w:.17}\n"));
        }
    }
    csv
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: engine [seed] [output-directory]".into());
    }
    let mut c = EngineConfig::default();
    if let Some(seed) = args.first() {
        c.seed = seed.parse()?;
    }
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("results/engine"));
    fs::create_dir_all(dir)?;
    let result = run(&c)?;
    let mut log = String::from(
        "phase,index,from,target,engine_prediction,frozen_prediction,markov_prediction\n",
    );
    let mut correct = [[0usize; 3]; 2];
    for s in &result.steps {
        let phase = usize::from(s.phase == "changed");
        for (k, p) in [s.engine, s.frozen, s.markov].iter().enumerate() {
            correct[phase][k] += usize::from(*p == s.target);
        }
        log.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            s.phase, s.index, s.from, s.target, s.engine, s.frozen, s.markov
        ));
    }
    fs::write(dir.join("predictions.csv"), log)?;
    fs::write(
        dir.join("before_change.csv"),
        topology(&result.train_weights),
    )?;
    fs::write(
        dir.join("after_change.csv"),
        topology(&result.final_weights),
    )?;
    let opt = |x: Option<usize>| x.map(|v| v.to_string()).unwrap_or_else(|| "null".into());
    let mut metrics=format!("{{\n  \"seed\": {}, \"symbols\": {}, \"train_steps\": {}, \"change_steps\": {}, \"rate\": {}, \"noise_probability\": {}, \"recovery_window\": {}, \"recovery_threshold\": {},\n  \"models\": {{\n",c.seed,c.symbols,c.train_steps,c.change_steps,c.rate,c.noise,c.window,c.recovery_threshold);
    for (k, name) in ["engine", "frozen", "markov"].iter().enumerate() {
        metrics.push_str(&format!("    \"{}\": {{\"train_accuracy\": {}, \"change_accuracy\": {}, \"old_rule_recall\": {}, \"new_rule_recall\": {}, \"recovery_step\": {}}}{}\n",name,correct[0][k] as f64/c.train_steps as f64,correct[1][k] as f64/c.change_steps as f64,result.old_recall[k],result.new_recall[k],opt(result.recovery[k]),if k<2 {","} else {""}));
    }
    metrics.push_str("  }\n}\n");
    fs::write(dir.join("metrics.json"), &metrics)?;
    print!("{metrics}");
    Ok(())
}

use murmur::{context_engine::run, sequence_engine::EngineConfig};
use std::{env, fs, path::Path};
fn snapshot(edges: &[Vec<Vec<f64>>]) -> String {
    let mut s = String::from("context,from,to,weight\n");
    for (context, graph) in edges.iter().enumerate() {
        for (from, row) in graph.iter().enumerate() {
            for (to, w) in row.iter().enumerate() {
                s.push_str(&format!("{context},{from},{to},{w:.17}\n"));
            }
        }
    }
    s
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: context [seed] [output-directory]".into());
    }
    let mut c = EngineConfig::default();
    if let Some(seed) = args.first() {
        c.seed = seed.parse()?;
    }
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("results/context"));
    fs::create_dir_all(dir)?;
    let r = run(&c)?;
    let mut log = String::from(
        "phase,index,context,source,target,context_prediction,plain_prediction,counts_prediction\n",
    );
    let mut counts = [[0usize; 3]; 3];
    let mut totals = [0usize; 3];
    for s in &r.steps {
        let phase = match s.phase {
            "A" => 0,
            "B" => 1,
            _ => 2,
        };
        totals[phase] += 1;
        for (k, p) in [
            s.context_prediction,
            s.plain_prediction,
            s.counts_prediction,
        ]
        .iter()
        .enumerate()
        {
            counts[phase][k] += usize::from(*p == s.target);
        }
        log.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            s.phase,
            s.index,
            s.context,
            s.source,
            s.target,
            s.context_prediction,
            s.plain_prediction,
            s.counts_prediction
        ));
    }
    fs::write(dir.join("predictions.csv"), log)?;
    fs::write(dir.join("after_a.csv"), snapshot(&r.edges_after_a))?;
    fs::write(dir.join("after_b.csv"), snapshot(&r.edges_after_b))?;
    let mut json=format!("{{\n  \"seed\": {}, \"symbols\": {}, \"contexts\": 2, \"phase_a_steps\": {}, \"phase_b_steps\": {}, \"alternating_steps\": 200, \"block_steps\": 50, \"rate\": {}, \"noise_probability\": {},\n  \"models\": {{\n",c.seed,c.symbols,c.train_steps,c.change_steps,c.rate,c.noise);
    for (k, name) in ["context_graph", "plain_graph", "context_counts"]
        .iter()
        .enumerate()
    {
        json.push_str(&format!("    \"{}\": {{\"phase_a_accuracy\": {}, \"phase_b_accuracy\": {}, \"alternating_accuracy\": {}, \"after_b_old_recall\": {}, \"after_b_new_recall\": {}, \"final_old_recall\": {}, \"final_new_recall\": {}}}{}\n",name,counts[0][k] as f64/totals[0] as f64,counts[1][k] as f64/totals[1] as f64,counts[2][k] as f64/totals[2] as f64,r.after_b[k][0],r.after_b[k][1],r.final_recall[k][0],r.final_recall[k][1],if k<2 {","}else{""}));
    }
    json.push_str("  }\n}\n");
    fs::write(dir.join("metrics.json"), &json)?;
    print!("{json}");
    Ok(())
}

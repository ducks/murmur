use murmur::{hidden_context::run, sequence_engine::EngineConfig};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: hidden [seed] [output-directory]".into());
    }
    let mut c = EngineConfig::default();
    if let Some(seed) = args.first() {
        c.seed = seed.parse()?;
    }
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("results/hidden"));
    fs::create_dir_all(dir)?;
    let (hazard, block, blocks) = (0.02, 50, 8);
    let result = run(&c, hazard, block, blocks)?;
    let mut csv=String::from("step,true_context,source,target,graph_prediction,counts_prediction,oracle_prediction,plain_prediction,prior_context_a,posterior_context_a,counts_posterior_context_a\n");
    let mut correct = [0usize; 4];
    let mut context_correct = [0usize; 2];
    let mut brier = [0.0; 2];
    for s in &result.steps {
        for (k, p) in [
            s.graph_prediction,
            s.count_prediction,
            s.oracle_prediction,
            s.plain_prediction,
        ]
        .iter()
        .enumerate()
        {
            correct[k] += usize::from(*p == s.target);
        }
        for (k, p) in [s.posterior_a, s.counts_posterior_a].iter().enumerate() {
            context_correct[k] += usize::from(usize::from(*p < 0.5) == s.context);
            let label = if s.context == 0 { 1.0 } else { 0.0 };
            brier[k] += (p - label) * (p - label);
        }
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{:.17},{:.17},{:.17}\n",
            s.index,
            s.context,
            s.source,
            s.target,
            s.graph_prediction,
            s.count_prediction,
            s.oracle_prediction,
            s.plain_prediction,
            s.prior_a,
            s.posterior_a,
            s.counts_posterior_a
        ));
    }
    fs::write(dir.join("predictions.csv"), csv)?;
    let mut weights = String::from("context,from,to,weight\n");
    for (context, g) in result.training_weights.iter().enumerate() {
        for (from, row) in g.iter().enumerate() {
            for (to, w) in row.iter().enumerate() {
                weights.push_str(&format!("{context},{from},{to},{w:.17}\n"));
            }
        }
    }
    fs::write(dir.join("trained_edges.csv"), weights)?;
    let n = result.steps.len() as f64;
    let json=format!("{{\n  \"seed\": {}, \"symbols\": {}, \"training_a_steps\": {}, \"training_b_steps\": {}, \"training_rate\": {}, \"noise_probability\": {}, \"switch_probability_assumed\": {}, \"block_steps\": {}, \"blocks\": {}, \"evaluation_steps\": {},\n  \"graph_accuracy\": {}, \"count_filter_accuracy\": {}, \"oracle_accuracy\": {}, \"context_free_accuracy\": {},\n  \"graph_context_accuracy_after_observation\": {}, \"count_context_accuracy_after_observation\": {}, \"graph_context_brier\": {}, \"count_context_brier\": {}\n}}\n",c.seed,c.symbols,c.train_steps,c.change_steps,c.rate,c.noise,hazard,block,blocks,result.steps.len(),correct[0] as f64/n,correct[1] as f64/n,correct[2] as f64/n,correct[3] as f64/n,context_correct[0] as f64/n,context_correct[1] as f64/n,brier[0]/n,brier[1]/n);
    fs::write(dir.join("metrics.json"), &json)?;
    print!("{json}");
    Ok(())
}

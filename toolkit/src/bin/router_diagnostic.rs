//! Exploratory development-only check: can the linear router fit clean labels?
use murmur_local_graph::{sequence::Codebook, Rng};
use murmur_research_runner::{
    learned_router::{Router, FEATURES},
    sharded::{self, SHARDS},
    Config,
};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: router_diagnostic <config.cfg> <new-output-directory>".into());
    }
    let c = Config::parse(&fs::read_to_string(&args[0])?)
        .map_err(|e| format!("invalid configuration: {e}"))?;
    let dir = Path::new(&args[1]);
    if let Some(p) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(p)?;
    }
    fs::create_dir(dir)?;
    let mut csv=String::from("seed,epochs,training_noise,clean_training_correct,clean_training_total,noisy_route_correct,noisy_route_total\n");
    for &seed in &c.seeds {
        let base = sharded::run_seed(&c, seed)?;
        let book = Codebook::new(seed.wrapping_add(1000));
        let mut model = Router::new();
        let mut rng = Rng(seed.wrapping_add(13000));
        for epoch in 1..=5000 {
            let mut order: Vec<_> = (0..base.stored.len()).collect();
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            for i in order {
                let p = book.encode(&base.stored[i]);
                model.learn(&p[..FEATURES], i % SHARDS);
            }
            if [100, 1000, 5000].contains(&epoch) {
                let train = base
                    .stored
                    .iter()
                    .enumerate()
                    .filter(|(i, s)| model.predict(&book.encode(s)[..FEATURES]) == i % SHARDS)
                    .count();
                let test = base
                    .rows
                    .iter()
                    .filter(|r| model.predict(&r.observed) == r.true_shard)
                    .count();
                csv.push_str(&format!(
                    "{seed},{epoch},0,{train},{},{test},{}\n",
                    base.stored.len(),
                    base.rows.len()
                ));
            }
        }
    }
    fs::write(dir.join("metrics.csv"), &csv)?;
    println!("{csv}");
    Ok(())
}

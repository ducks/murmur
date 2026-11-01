use murmur_research_runner::{predictive_graph::execute, Config};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = env::args().skip(1).collect();
    if a.len() != 2 {
        return Err("usage: predictive_graph <config.cfg> <new-output-directory>".into());
    }
    let c = Config::parse(&fs::read_to_string(&a[0])?)
        .map_err(|e| format!("invalid configuration: {e}"))?;
    execute(&c, Path::new(&a[1]))?;
    println!("Run complete: {}/summary.md", a[1]);
    Ok(())
}

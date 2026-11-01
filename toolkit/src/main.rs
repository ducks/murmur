use murmur_research_runner::{execute, Config};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: murmur-research-runner <config.cfg> <new-output-directory>".into());
    }
    let text = fs::read_to_string(&args[0])?;
    let config = Config::parse(&text).map_err(|e| format!("invalid configuration: {e}"))?;
    execute(&config, Path::new(&args[1]))?;
    println!("Run complete: {}/summary.md", args[1]);
    Ok(())
}

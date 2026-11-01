use murmur_research_runner::{learned_router::execute, Config};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: learned_router <config.cfg> <new-output-directory>".into());
    }
    let c = Config::parse(&fs::read_to_string(&args[0])?)
        .map_err(|e| format!("invalid configuration: {e}"))?;
    execute(&c, Path::new(&args[1]))?;
    println!("Run complete: {}/summary.md", args[1]);
    Ok(())
}

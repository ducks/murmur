use murmur_research_runner::{independent_encoding::execute, Config};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: independent_encoding <config.cfg> <new-output-directory>".into());
    }
    let c = Config::parse(&fs::read_to_string(&args[0])?)
        .map_err(|e| format!("invalid configuration: {e}"))?;
    execute(&c, Path::new(&args[1]))?;
    println!("Run complete: {}/summary.md", args[1]);
    Ok(())
}

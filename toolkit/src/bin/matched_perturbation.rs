use murmur_research_runner::{matched_perturbation, temporal_stream::StreamConfig};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: matched_perturbation <config> <new-output-directory>".into());
    }
    let config = StreamConfig::parse(&fs::read_to_string(&args[0])?)?;
    matched_perturbation::execute(&config, Path::new(&args[1]))?;
    println!("Run complete: {}/summary.md", args[1]);
    Ok(())
}

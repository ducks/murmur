use murmur_research_runner::{temporal_stream::StreamConfig, trained_strength};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = env::args().skip(1).collect();
    if a.len() != 2 {
        return Err("usage: trained_strength <config> <new-output-directory>".into());
    }
    let c = StreamConfig::parse(&fs::read_to_string(&a[0])?)?;
    trained_strength::execute(&c, Path::new(&a[1]))?;
    println!("Run complete: {}/summary.md", a[1]);
    Ok(())
}

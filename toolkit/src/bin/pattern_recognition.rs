use murmur_research_runner::{pattern_recognition, temporal_stream::StreamConfig};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<_> = env::args().skip(1).collect();
    if a.len() != 2 {
        return Err("usage: pattern_recognition <config> <new-output-directory>".into());
    }
    let c = StreamConfig::parse(&fs::read_to_string(&a[0])?)?;
    pattern_recognition::execute(&c, Path::new(&a[1]))?;
    println!("Run complete: {}/summary.md", a[1]);
    Ok(())
}

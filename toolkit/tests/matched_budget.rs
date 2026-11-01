use murmur_local_graph::sequence::{Codebook, BITS};
use murmur_research_runner::{
    matched_budget::{decode_large, duplicate_blocks, execute, run_seed, LARGE_N},
    sharded, Config,
};
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};
fn config() -> Config {
    Config::parse("schema_version=1\nseeds=42001\npatterns=8\nprefix_noise=0.2\ntrials_per_sequence=2\nsweeps=5\ndensity=0.25\n").unwrap()
}
#[test]
fn expanded_code_contains_only_duplicates_and_decodes() {
    let book = Codebook::new(1);
    let seq = [0, 1, 2, 3];
    let base = book.encode(&seq);
    let expanded = duplicate_blocks(&base);
    assert_eq!(expanded.len(), LARGE_N);
    for (small, big) in base.chunks_exact(BITS).zip(expanded.chunks_exact(2 * BITS)) {
        assert_eq!(&big[..BITS], small);
        assert_eq!(&big[BITS..], small);
    }
    assert_eq!(decode_large(&book, &expanded[3 * 2 * BITS..]), 3);
}
#[test]
fn matched_storage_and_shared_examples_are_reproducible() {
    let c = config();
    let a = run_seed(&c, 42001).unwrap();
    let b = run_seed(&c, 42001).unwrap();
    let prior = sharded::run_seed(&c, 42001).unwrap();
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.stored, prior.stored);
    assert_eq!(a.large_bytes, a.shard_resources.four_graph_bytes);
    for (row, prior) in a.rows.iter().zip(prior.rows) {
        assert_eq!(row.source, prior);
    }
}
#[test]
fn output_refuses_existing_directories() {
    let c = config();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "murmur-matched-test-{}-{unique}",
        std::process::id()
    ));
    execute(&c, &dir).unwrap();
    assert_eq!(
        fs::read_to_string(dir.join("status.txt")).unwrap(),
        "complete\n"
    );
    assert!(execute(&c, &dir).is_err());
    assert!(dir.join("resources.csv").exists());
    fs::remove_dir_all(dir).unwrap();
}

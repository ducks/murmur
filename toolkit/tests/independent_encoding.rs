use murmur_local_graph::sequence::{Codebook, BITS};
use murmur_research_runner::{
    independent_encoding::{combine, decode, run_seed},
    matched_budget, sharded, Config,
};
#[test]
fn combines_independent_codes_and_decodes() {
    let a = Codebook::new(1);
    let b = Codebook::new(2);
    let seq = [0, 1, 2, 3];
    let first = a.encode(&seq);
    let extra = b.encode(&seq);
    let combined = combine(&first, &extra);
    for (i, block) in combined.chunks_exact(2 * BITS).enumerate() {
        assert_eq!(&block[..BITS], &first[i * BITS..(i + 1) * BITS]);
        assert_eq!(&block[BITS..], &extra[i * BITS..(i + 1) * BITS]);
    }
    assert_eq!(decode(&a, &b, &combined[6 * BITS..]), 3);
}
#[test]
fn both_controls_share_the_original_cues_and_storage() {
    let c=Config::parse("schema_version=1\nseeds=42001\npatterns=8\nprefix_noise=0.2\ntrials_per_sequence=2\nsweeps=5\ndensity=0.25\n").unwrap();
    let a = run_seed(&c, 42001).unwrap();
    let b = run_seed(&c, 42001).unwrap();
    let base = sharded::run_seed(&c, 42001).unwrap();
    let duplicate = matched_budget::run_seed(&c, 42001).unwrap();
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.payload_bytes, duplicate.large_bytes);
    assert_eq!(a.active_edges, duplicate.large_edges);
    for (row, original) in a.rows.iter().zip(base.rows) {
        assert_eq!(row.source, original);
        assert_eq!(row.extra_prefix.len(), 3 * BITS);
    }
}

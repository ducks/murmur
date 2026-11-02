use murmur_research_runner::temporal_stream::{
    execute, run_delay, stream, ShiftRegister, StreamConfig,
};
use std::{
    fs,
    time::{SystemTime, UNIX_EPOCH},
};
fn config() -> StreamConfig {
    StreamConfig::parse("schema_version=1\nseeds=92001\ndelays=1,4\ntrain_tokens=100\ntest_tokens=80\nunits=16\ndegree=3\nlearning_rate=0.05\n").unwrap()
}
#[test]
fn targets_are_prior_inputs_not_future_or_training_data() {
    let c = config();
    let test = stream(92001 + 32000, c.test_tokens);
    let r = run_delay(&c, 92001, 4);
    assert_eq!(r.rows.len(), 76);
    for x in &r.rows {
        assert_eq!(x.input, test[x.index]);
        assert_eq!(x.target, test[x.index - 4]);
        assert_eq!(x.shift, x.target);
    }
}
#[test]
fn shift_register_uses_only_seen_symbols() {
    let mut memory = ShiftRegister::new(2);
    assert_eq!(memory.observe(1), None);
    assert_eq!(memory.observe(0), None);
    assert_eq!(memory.observe(0), Some(1));
    assert_eq!(memory.observe(1), Some(0));
}
#[test]
fn deterministic_and_configuration_checked() {
    let c = config();
    let a = run_delay(&c, 92001, 4);
    let b = run_delay(&c, 92001, 4);
    assert_eq!(a.rows, b.rows);
    assert_eq!(StreamConfig::parse(&c.resolved()).unwrap(), c);
    assert!(StreamConfig::parse(&c.resolved().replace("delays=1,4", "delays=0,4")).is_err());
    assert!(
        StreamConfig::parse(&c.resolved().replace("seeds=92001", "seeds=92001,92001")).is_err()
    );
    assert!(
        StreamConfig::parse(&c.resolved().replace("seeds=92001", "seeds=92001,93001")).is_err()
    );
}
#[test]
fn provided_development_and_test_stream_seeds_do_not_overlap() {
    use std::collections::HashSet;
    let dev = StreamConfig::parse(include_str!("../configs/temporal/dev.cfg")).unwrap();
    let test = StreamConfig::parse(include_str!("../configs/temporal/test.cfg")).unwrap();
    let seeds = |c: &StreamConfig| -> HashSet<u64> {
        c.seeds
            .iter()
            .flat_map(|s| [s.wrapping_add(31000), s.wrapping_add(32000)])
            .collect()
    };
    assert!(seeds(&dev).is_disjoint(&seeds(&test)));
}
#[test]
fn output_is_complete_and_does_not_overwrite() {
    let c = config();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("murmur-stream-{}-{unique}", std::process::id()));
    execute(&c, &dir).unwrap();
    assert_eq!(
        fs::read_to_string(dir.join("status.txt")).unwrap(),
        "complete\n"
    );
    assert!(execute(&c, &dir).is_err());
    assert!(dir.join("streams.csv").exists());
    fs::remove_dir_all(dir).unwrap();
}

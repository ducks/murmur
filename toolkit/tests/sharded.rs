use murmur_local_graph::sequence::{lookup, sequences, Codebook, BITS};
use murmur_research_runner::{
    evaluate_seed,
    sharded::{execute, make_index, route, run_seed, SHARDS},
    Config,
};
use std::{
    collections::HashSet,
    fs,
    time::{SystemTime, UNIX_EPOCH},
};
fn config() -> Config {
    Config::parse("schema_version=1\nseeds=31001\npatterns=8\nprefix_noise=0.2\ntrials_per_sequence=3\nsweeps=5\ndensity=0.25\n").unwrap()
}
#[test]
fn router_uses_only_prefix_and_balances_assignments() {
    let stored = sequences(3, 32);
    let index = make_index(&stored);
    let book = Codebook::new(2);
    assert_eq!(
        (0..SHARDS)
            .map(|shard| index.iter().filter(|x| x.shard == shard).count())
            .collect::<Vec<_>>(),
        vec![8; 4]
    );
    for seq in &stored {
        let encoded = book.encode(seq);
        let prefix = &encoded[..3 * BITS];
        let found = route(prefix, &index, &book);
        assert_eq!(stored[found][3], lookup(prefix, &stored, &book));
        let mut changed = stored.clone();
        for s in &mut changed {
            s[3] = (s[3] + 1) % 4;
        }
        assert_eq!(make_index(&changed), index);
        assert_eq!(route(prefix, &make_index(&changed), &book), found);
    }
}
#[test]
fn single_graph_matches_existing_runner_and_oracle_is_conditional() {
    let c = config();
    let a = run_seed(&c, 31001).unwrap();
    let b = run_seed(&c, 31001).unwrap();
    let old = evaluate_seed(&c, 31001);
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.stored, old.stored);
    assert_eq!(a.rows.len(), c.patterns * c.trials_per_sequence);
    assert_eq!(
        a.stored
            .iter()
            .map(|x| [x[0], x[1], x[2]])
            .collect::<HashSet<_>>()
            .len(),
        c.patterns
    );
    for (r, o) in a.rows.iter().zip(old.rows) {
        assert_eq!(r.observed, o.observed);
        assert_eq!(r.missing, o.missing);
        assert_eq!(r.one_hebb, o.trained);
        assert_eq!(r.one_storkey, o.storkey);
        assert_eq!(r.lookup, o.lookup);
        if r.true_shard == r.routed_shard {
            assert_eq!(r.oracle_hebb, r.routed_hebb);
            assert_eq!(r.oracle_storkey, r.routed_storkey);
        }
    }
    assert_eq!(
        a.resources.four_graph_bytes,
        4 * a.resources.one_graph_bytes
    );
}
#[test]
fn output_refuses_overwrite_and_rejects_bad_partition() {
    let mut c = config();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("murmur-shard-test-{}-{unique}", std::process::id()));
    c.patterns = 7;
    assert!(execute(&c, &dir).is_err());
    assert!(!dir.exists());
    c.patterns = 8;
    execute(&c, &dir).unwrap();
    assert_eq!(
        fs::read_to_string(dir.join("status.txt")).unwrap(),
        "complete\n"
    );
    assert!(execute(&c, &dir).is_err());
    assert!(dir.join("predictions.csv").exists());
    fs::remove_dir_all(dir).unwrap();
}

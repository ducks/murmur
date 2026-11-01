use murmur_research_runner::{
    learned_router::{run_seed, Router, FEATURES},
    sharded, Config,
};
#[test]
fn linear_router_learns_and_is_deterministic() {
    let mut a = Router::new();
    let mut b = Router::new();
    let x = vec![1; FEATURES];
    let y = vec![-1; FEATURES];
    for _ in 0..100 {
        a.learn(&x, 1);
        a.learn(&y, 2);
        b.learn(&x, 1);
        b.learn(&y, 2);
    }
    assert_eq!(a, b);
    assert_eq!(a.predict(&x), 1);
    assert_eq!(a.predict(&y), 2);
}
#[test]
fn evaluated_router_uses_shared_cues_and_correct_shard_graphs() {
    let c=Config::parse("schema_version=1\nseeds=52001\npatterns=8\nprefix_noise=0.2\ntrials_per_sequence=2\nsweeps=5\ndensity=0.25\n").unwrap();
    let a = run_seed(&c, 52001).unwrap();
    let b = run_seed(&c, 52001).unwrap();
    let base = sharded::run_seed(&c, 52001).unwrap();
    assert_eq!(a.router, b.router);
    assert_eq!(a.rows, b.rows);
    for (row, old) in a.rows.iter().zip(base.rows) {
        assert_eq!(row.source, old);
        assert_eq!(row.learned_shard, a.router.predict(&row.source.observed));
        if row.learned_shard == row.source.true_shard {
            assert_eq!(row.learned_hebb, row.source.oracle_hebb);
            assert_eq!(row.learned_storkey, row.source.oracle_storkey);
        }
    }
}

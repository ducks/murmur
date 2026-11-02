use murmur_local_graph::{
    sequence::{Codebook, BITS, N},
    Graph,
};
use murmur_research_runner::{
    dynamics::{simulate, Mode},
    predictive_graph, Config,
};
#[test]
fn fixed_order_detects_a_real_two_unit_cycle() {
    let mut g = Graph::new(N, 0.0, 1);
    let a = 96;
    let b = 97;
    g.weights[a * N + b] = 1.0;
    g.weights[b * N + a] = -1.0;
    let book = Codebook::new(1);
    let outcome = simulate(&g, &vec![1; 96], &vec![1; BITS], &book, Mode::Forward20, 1);
    assert!(!outcome.converged);
    assert!(!outcome.strict_stable);
    assert_eq!(outcome.cycle_length, Some(2));
}
#[test]
fn margin_settling_is_not_claimed_as_original_stability() {
    let mut g = Graph::new(N, 0.0, 1);
    g.weights[96 * N] = 0.1;
    let book = Codebook::new(1);
    let o = simulate(&g, &vec![1; 96], &vec![-1; BITS], &book, Mode::Margin020, 1);
    assert!(o.converged);
    assert!(!o.strict_stable);
}
#[test]
fn forward_reproduces_existing_model_and_random_order_is_seeded() {
    let c=Config::parse("schema_version=1\nseeds=72001\npatterns=4\nprefix_noise=0.2\ntrials_per_sequence=2\nsweeps=20\ndensity=0.25\n").unwrap();
    let run = predictive_graph::run_seed(&c, 72001);
    let book = Codebook::new(73001);
    for row in &run.rows {
        let p = &row.source;
        let a = simulate(
            &run.full.graph,
            &p.observed,
            &p.missing,
            &book,
            Mode::Forward20,
            2,
        );
        assert_eq!(a.symbol, row.full);
        assert_eq!(a.converged, row.full_converged);
        let x = simulate(
            &run.full.graph,
            &p.observed,
            &p.missing,
            &book,
            Mode::Shuffled20,
            3,
        );
        assert_eq!(
            x,
            simulate(
                &run.full.graph,
                &p.observed,
                &p.missing,
                &book,
                Mode::Shuffled20,
                3
            )
        );
        assert_eq!(x.cycle_length, None);
    }
}

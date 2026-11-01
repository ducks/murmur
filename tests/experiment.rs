use murmur::{
    data::trials,
    experiment::{run, Config},
    graph::Graph,
};
#[test]
fn propagation_is_deterministic_and_bounded() {
    let mut a = Graph::new(128, 4, 9);
    let mut b = a.clone();
    for i in 0..1000 {
        a.step(i % 3);
        b.step(i % 3);
    }
    assert_eq!(a.state, b.state);
    assert!(a.state.iter().all(|x| x.is_finite() && x.abs() <= 1.0));
}
#[test]
fn cue_survives_marker_and_reset_removes_it() {
    let mut a = Graph::new(128, 4, 42);
    let mut b = a.clone();
    a.step(0);
    b.step(1);
    a.step(2);
    b.step(2);
    assert_ne!(a.state, b.state);
    a.reset();
    b.reset();
    a.step(2);
    b.step(2);
    assert_eq!(a.state, b.state);
}
#[test]
fn run_is_reproducible_and_learns_held_out_trials() {
    let c = Config::default();
    let (a, _, _) = run(&c);
    let (b, _, _) = run(&c);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
    assert!(a.graph_correct > 900);
    assert_eq!(trials(7, 20), trials(7, 20));
}

#[test]
fn removing_edges_preserves_inputs_and_changes_dynamics() {
    let mut graph = Graph::new(128, 4, 42);
    let mut no_edges = graph.clone();
    no_edges.edges.clear();
    assert_eq!(graph.input, no_edges.input);
    graph.step(0);
    no_edges.step(0);
    assert_eq!(graph.state, no_edges.state);
    graph.step(2);
    no_edges.step(2);
    assert_ne!(graph.state, no_edges.state);
}

#[test]
fn longer_delay_is_reproducible() {
    let c = Config {
        delay: 10,
        train_trials: 50,
        test_trials: 50,
        ..Config::default()
    };
    assert_eq!(format!("{:?}", run(&c).0), format!("{:?}", run(&c).0));
}

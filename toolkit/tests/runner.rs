use murmur_local_graph::{
    diagnostics::without_output_feedback,
    sequence::{evaluate, graph_predict, Codebook, BITS, POSITIONS},
    Graph,
};
use murmur_research_runner::{evaluate_seed, execute, Config};
use std::{
    collections::HashSet,
    fs,
    time::{SystemTime, UNIX_EPOCH},
};
fn config() -> Config {
    Config::parse("schema_version=1\nseeds=201,202\npatterns=8\nprefix_noise=0.2\ntrials_per_sequence=5\nsweeps=5\ndensity=0.25\n").unwrap()
}
#[test]
fn configuration_round_trips_and_invalid_fields_are_rejected() {
    let c = config();
    assert_eq!(Config::parse(&c.resolved()).unwrap(), c);
    assert!(Config::parse(&c.resolved().replace("schema_version=1", "schema_version=2")).is_err());
    assert!(Config::parse(&format!("{}patterns=2\n", c.resolved())).is_err());
    assert!(Config::parse(&format!("{}bogus=1\n", c.resolved())).is_err());
    assert!(Config::parse(&c.resolved().replace("prefix_noise=0.2", "prefix_noise=NaN")).is_err());
    assert!(Config::parse(&c.resolved().replace("seeds=201,202", "seeds=201,201")).is_err());
}
#[test]
fn reproduces_original_and_models_use_identical_inputs() {
    let c = config();
    let a = evaluate_seed(&c, 201);
    let b = evaluate_seed(&c, 201);
    assert_eq!(a.rows, b.rows);
    let expected_graph_bytes =
        128 * 128 * (std::mem::size_of::<f64>() + std::mem::size_of::<bool>());
    for model in ["trained", "storkey"] {
        assert_eq!(
            a.resource_rows
                .iter()
                .find(|r| r.model == model)
                .unwrap()
                .owned_payload_bytes,
            expected_graph_bytes
        );
    }
    let original = evaluate(
        201,
        c.patterns,
        c.prefix_noise,
        c.trials_per_sequence,
        c.sweeps,
    );
    assert_eq!(
        a.rows.iter().filter(|r| r.trained == r.target).count(),
        original.graph_correct
    );
    assert_eq!(
        a.rows.iter().filter(|r| r.lookup == r.target).count(),
        original.lookup_correct
    );
    assert_eq!(
        a.rows.iter().filter(|r| r.untrained == r.target).count(),
        original.untrained_correct
    );
    assert_eq!(
        a.rows.iter().filter(|r| r.majority == r.target).count(),
        original.majority_correct
    );
    let prefixes: HashSet<_> = a.stored.iter().map(|s| [s[0], s[1], s[2]]).collect();
    assert_eq!(prefixes.len(), c.patterns);
    let book = Codebook::new(1201);
    let mut graph = Graph::new(128, c.density, 201);
    let mut storkey = graph.clone();
    for seq in &a.stored {
        graph.learn(&book.encode(seq));
        storkey.learn_storkey(&book.encode(seq));
    }
    let no_feedback = without_output_feedback(&graph);
    for row in &a.rows {
        assert_eq!(row.sequence, a.stored[row.sequence_index]);
        assert_eq!(row.observed.len(), (POSITIONS - 1) * BITS);
        assert_eq!(
            row.no_feedback,
            graph_predict(&no_feedback, &row.observed, &row.missing, &book, c.sweeps)
        );
        assert_eq!(
            row.trained,
            graph_predict(&graph, &row.observed, &row.missing, &book, c.sweeps)
        );
        assert_eq!(
            row.storkey,
            graph_predict(&storkey, &row.observed, &row.missing, &book, c.sweeps)
        );
    }
}
#[test]
fn creates_evidence_and_never_overwrites_a_run() {
    let c = config();
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "murmur-runner-test-{}-{unique}",
        std::process::id()
    ));
    execute(&c, &dir).unwrap();
    assert_eq!(
        fs::read_to_string(dir.join("status.txt")).unwrap(),
        "complete\n"
    );
    assert_eq!(
        Config::parse(&fs::read_to_string(dir.join("resolved.cfg")).unwrap()).unwrap(),
        c
    );
    let predictions = fs::read_to_string(dir.join("predictions.csv")).unwrap();
    assert_eq!(
        predictions.lines().count(),
        1 + c.seeds.len() * c.patterns * c.trials_per_sequence
    );
    assert_eq!(
        fs::read_to_string(dir.join("resources.csv"))
            .unwrap()
            .lines()
            .count(),
        1 + 3 * c.seeds.len()
    );
    assert_eq!(
        fs::read_to_string(dir.join("diagnostics.csv"))
            .unwrap()
            .lines()
            .count(),
        1 + 2 * c.seeds.len()
    );
    assert!(fs::read_to_string(dir.join("summary.md"))
        .unwrap()
        .contains("not unseen-sequence generalization"));
    assert!(execute(&c, &dir).is_err());
    assert_eq!(
        predictions,
        fs::read_to_string(dir.join("predictions.csv")).unwrap()
    );
    fs::remove_dir_all(dir).unwrap();
}

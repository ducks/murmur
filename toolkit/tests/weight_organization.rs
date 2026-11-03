use murmur::graph::Graph;
use murmur_research_runner::{
    temporal_stream::StreamConfig,
    weight_organization::{self, shuffle},
};
use std::collections::HashSet;
fn config() -> StreamConfig {
    StreamConfig {
        seeds: vec![23],
        delays: vec![4],
        train_tokens: 120,
        test_tokens: 100,
        units: 16,
        degree: 3,
        learning_rate: 0.05,
    }
}
fn sorted(g: &Graph) -> Vec<u64> {
    let mut v: Vec<_> = g.edges.iter().map(|e| e.weight.to_bits()).collect();
    v.sort();
    v
}
#[test]
fn permutations_preserve_weights_and_destination_multisets() {
    let g = Graph::new(16, 3, 23);
    for within in [false, true] {
        let a = shuffle(&g, 99, within);
        let b = shuffle(&g, 99, within);
        assert_eq!(sorted(&g), sorted(&a));
        assert_eq!(
            a.edges
                .iter()
                .map(|e| e.weight.to_bits())
                .collect::<Vec<_>>(),
            b.edges
                .iter()
                .map(|e| e.weight.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(a.input, g.input);
        for (x, y) in g.edges.iter().zip(&a.edges) {
            assert_eq!((x.from, x.to), (y.from, y.to));
        }
        if within {
            for dest in 0..16 {
                let mut x: Vec<_> = g
                    .edges
                    .iter()
                    .filter(|e| e.to == dest)
                    .map(|e| e.weight.to_bits())
                    .collect();
                let mut y: Vec<_> = a
                    .edges
                    .iter()
                    .filter(|e| e.to == dest)
                    .map(|e| e.weight.to_bits())
                    .collect();
                x.sort();
                y.sort();
                assert_eq!(x, y);
            }
        }
    }
}
#[test]
fn paired_runs_are_deterministic() {
    let c = config();
    let a = weight_organization::run(&c, 23, 4);
    let b = weight_organization::run(&c, 23, 4);
    assert_eq!(a.predictions, b.predictions);
    assert_eq!(a.baseline.rows, b.baseline.rows);
    assert_eq!(a.predictions.len(), 9);
    for g in &a.graphs[3..] {
        assert_eq!(sorted(g), sorted(&a.graphs[1]));
    }
}
#[test]
fn seed_roles_do_not_overlap() {
    let a = StreamConfig::parse(include_str!("../configs/weight-organization/dev.cfg")).unwrap();
    let b = StreamConfig::parse(include_str!("../configs/weight-organization/test.cfg")).unwrap();
    let offsets = [
        0u64, 31000, 32000, 33000, 34000, 35000, 36000, 38000, 40000, 42000, 44000, 46000,
    ];
    let roles = |c: &StreamConfig| {
        c.seeds
            .iter()
            .flat_map(|s| offsets.map(|o| s.wrapping_add(o)))
            .collect::<HashSet<_>>()
    };
    assert_eq!(roles(&a).len(), a.seeds.len() * offsets.len());
    assert_eq!(roles(&b).len(), b.seeds.len() * offsets.len());
    assert!(roles(&a).is_disjoint(&roles(&b)));
}

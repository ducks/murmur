use murmur::{
    experiment::Config,
    robustness::{generate, run, test_sequence, Task},
};
use std::collections::HashSet;
#[test]
fn varied_sequences_have_disjoint_train_and_test_splits() {
    for delay in [4, 8] {
        let train = generate(1, 500, delay, true, false);
        let test = generate(2, 500, delay, true, true);
        assert!(train.iter().all(|x| !test_sequence(&x.distractors)));
        assert!(test.iter().all(|x| test_sequence(&x.distractors)));
        let a: HashSet<_> = train.iter().map(|x| x.distractors.clone()).collect();
        assert!(test.iter().all(|x| !a.contains(&x.distractors)));
    }
}
#[test]
fn noisy_experiment_is_reproducible() {
    let c = Config {
        delay: 4,
        train_trials: 50,
        test_trials: 50,
        ..Config::default()
    };
    let t = Task {
        varied: true,
        noise: 0.01,
    };
    assert_eq!(
        format!("{:?}", run(&c, t).unwrap()),
        format!("{:?}", run(&c, t).unwrap())
    );
    assert!(run(
        &c,
        Task {
            varied: true,
            noise: -0.1
        }
    )
    .is_err());
}

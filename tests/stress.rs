use murmur::{
    discovery::{stream, stream_with_block},
    stress::Cusum,
    trial_growth::{run, run_custom},
};
#[test]
fn default_stream_and_experiment_remain_unchanged() {
    let a = stream(1, 300, true, 0.1);
    let b = stream_with_block(1, 300, true, 0.1, 100);
    for (x, y) in a.iter().zip(b) {
        assert_eq!(
            (x.source, x.target, x.context),
            (y.source, y.target, y.context)
        );
    }
    assert_eq!(
        format!("{:?}", run(12, true, 0.1, 40.0).unwrap()),
        format!(
            "{:?}",
            run_custom(12, true, 0.1, 40.0, 1200, 400, 100).unwrap()
        )
    );
}
#[test]
fn cusum_does_not_alarm_on_predictable_sequence() {
    let mut c = Cusum::new(10.0);
    for i in 0..1000 {
        c.observe(i % 4, (i + 1) % 4);
    }
    assert_eq!(c.first_alarm, None);
}

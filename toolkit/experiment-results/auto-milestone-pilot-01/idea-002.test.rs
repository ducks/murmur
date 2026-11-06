use murmur::data::Rng;
use murmur_research_runner::{agent_candidate::Candidate, unlabeled_context::Mixture};

#[test]
fn switch_guard_and_frozen_learning() {
    let p = [0.99, 0.01];
    let stable = Candidate::persistence_probability(p, [0.9, 0.1], 0.05);
    let changed = Candidate::persistence_probability(p, [0.01, 0.99], 0.05);
    assert!(stable > changed && changed > 0.0 && stable < 1.0);
    let mut model = Candidate::new(71, 0.05).unguarded_smoothing().guarded_smoothing();
    let mut control = model.unguarded_smoothing();
    for i in 0..300 {
        let x = i % 2;
        let y = if i % 9 < 6 { x } else { 1-x };
        model.observe(x, y, true);
        control.observe(x, y, true);
    }
    model.reset(); control.reset();
    assert!((0..2).any(|x| (model.predict(x)-control.predict(x)).abs() > 1e-10));
    let mut frozen = model.clone();
    for i in 0..100 { frozen.observe(i % 2, (i / 3) % 2, false); }
    frozen.reset(); model.reset();
    for x in 0..2 { assert_eq!(frozen.predict(x), model.predict(x)); }
}

#[test]
fn consistency_filter_changes_learning_but_not_frozen_parameters() {
    let mut filtered = Candidate::new(871, 0.05);
    let mut unguarded = filtered.unguarded_smoothing();
    for i in 0..600 {
        let x = (i / 3) % 2;
        let y = usize::from(i % 11 < 7) ^ x;
        filtered.observe(x, y, true);
        unguarded.observe(x, y, true);
    }
    filtered.reset(); unguarded.reset();
    assert!((0..2).any(|x| (filtered.predict(x)-unguarded.predict(x)).abs() > 1e-10));
    let mut frozen = filtered.clone();
    for i in 0..100 { frozen.observe(i % 2, (i / 3) % 2, false); }
    frozen.reset();
    for x in 0..2 { assert_eq!(frozen.predict(x), filtered.predict(x)); }
}

// Prespecified replication: 200 paired independent seeds, 200 events per
// phase, and 1,000 frozen B transitions. Percentile bootstrap resamples seeds.
#[test]
fn high_noise_boundary_suppression_replication() {
    const N: usize = 200;
    const LENGTH: usize = 200;
    const PROBE: usize = 1000;
    let rules = [[0.3, 0.7], [0.7, 0.3]];
    let mut totals = [[0usize; 4]; 3];
    let mut differences = [[Vec::<f64>::new(), Vec::<f64>::new()],
                           [Vec::<f64>::new(), Vec::<f64>::new()],
                           [Vec::<f64>::new(), Vec::<f64>::new()]];
    for seed in 0..N as u64 {
        let mut rng = Rng(94_100_000 + seed);
        let model = Candidate::new(94_200_000 + seed, 0.05).unguarded_smoothing();
        let mut models = [model.clone(), model, Candidate::new(94_200_000 + seed, 0.05)];
        let mut mixture = Mixture::new(94_200_000 + seed, 0.05);
        let mut counts = [[0usize; 4]; 3];
        let mut x = (rng.next_u64() & 1) as usize;
        for t in 0..3 * LENGTH {
            let phase = t / LENGTH;
            let context = usize::from(phase == 1);
            let y = usize::from(rng.uniform() < rules[context][x]);
            let predictions = [models[0].predict(x), models[1].predict(x), mixture.step(x, y, true).0, models[2].predict(x)];
            if phase > 0 && t % LENGTH < 100 {
                for k in 0..4 { counts[phase - 1][k] += usize::from(usize::from(predictions[k] > 0.5) == y); }
            }
            models[0].observe_with_boundary_suppression(x, y, true, t == LENGTH || t == 2 * LENGTH);
            models[1].observe(x, y, true);
            models[2].observe(x, y, true);
            x = y;
        }
        for model in &mut models { model.reset(); }
        mixture.reset();
        let mut probe = Rng(94_300_000 + seed);
        let mut x = (probe.next_u64() & 1) as usize;
        for _ in 0..PROBE {
            let y = usize::from(probe.uniform() < rules[1][x]);
            for k in 0..2 {
                counts[2][k] += usize::from(usize::from(models[k].predict(x) > 0.5) == y);
                models[k].observe(x, y, false);
            }
            counts[2][3] += usize::from(usize::from(models[2].predict(x) > 0.5) == y);
            models[2].observe(x, y, false);
            counts[2][2] += usize::from(usize::from(mixture.step(x, y, false).0 > 0.5) == y);
            x = y;
        }
        for w in 0..3 {
            let trials = if w == 2 { PROBE } else { 100 };
            for k in 0..4 { totals[w][k] += counts[w][k]; }
            for k in 1..3 { differences[w][k-1].push((counts[w][0] as f64 - counts[w][k] as f64) / trials as f64); }
        }
    }
    for (w, label) in ["switch_to_b", "switch_to_a", "retention_b"].iter().enumerate() {
        let trials = N * if w == 2 { PROBE } else { 100 };
        println!("MURMUR_FOLLOWUP {{\"measure\":\"high_noise_200_{label}\",\"candidate\":\"consistency_filter\",\"control\":\"mixture\",\"candidate_correct\":{},\"control_correct\":{},\"trials\":{trials}}}", totals[w][3], totals[w][2]);
        for (k, control) in [(1, "unguarded"), (2, "mixture")] {
            let d = &differences[w][k-1];
            let mean = d.iter().sum::<f64>() / N as f64;
            let mut rng = Rng(94_400_000 + (w * 2 + k) as u64);
            let mut bootstrap = Vec::with_capacity(10_000);
            for _ in 0..10_000 {
                let sum: f64 = (0..N).map(|_| d[(rng.next_u64() % N as u64) as usize]).sum();
                bootstrap.push(sum / N as f64);
            }
            bootstrap.sort_by(|a,b| a.total_cmp(b));
            let (lo, hi) = (bootstrap[250], bootstrap[9749]);
            println!("paired high_noise_200 {label} boundary-suppression-minus-{control}: mean={mean:.6} bootstrap95=[{lo:.6},{hi:.6}]");
            if w == 2 && k == 1 {
                let decision = if lo > 0.0 && mean >= 0.01 { "supports benefit" } else if hi < 0.0 && mean <= -0.01 { "weighs against boundary suppression" } else { "inconclusive" };
                println!("primary decision: {decision}");
            }
            if k == 2 { println!("mixture criterion {label}: {}", if mean < -0.02 { "fail" } else { "pass" }); }
            println!("MURMUR_FOLLOWUP {{\"measure\":\"high_noise_200_{label}\",\"candidate\":\"boundary_suppression\",\"control\":\"{control}\",\"candidate_correct\":{},\"control_correct\":{},\"trials\":{trials}}}", totals[w][0], totals[w][k]);
        }
    }
}

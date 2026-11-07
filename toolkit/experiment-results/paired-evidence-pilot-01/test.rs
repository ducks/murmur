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

// Prespecified seed ranges (inclusive): streams 97_100_000..=97_100_999,
// initialization 97_200_000..=97_200_999, probes 97_300_000..=97_300_999.
// These do not overlap the preceding 96-million-range replication. Each seed is a resampling unit.
#[test]
fn high_noise_consistency_filter_replication() {
    const N: usize = 1000;
    const LENGTH: usize = 200;
    const PROBE: usize = 1000;
    const BOOTSTRAPS: usize = 10_000;
    let rules = [[0.3, 0.7], [0.7, 0.3]];
    // Model indices: filter, unguarded, fixed mixture. Outcomes: B, A, recall.
    let mut totals = [[0usize; 3]; 3];
    let mut per_seed = [[[0usize; 3]; 3]; N];
    for seed in 0..N {
        let mut rng = Rng(97_100_000 + seed as u64);
        let mut filtered = Candidate::new(97_200_000 + seed as u64, 0.05);
        let mut unguarded = filtered.unguarded_smoothing();
        let mut mixture = Mixture::new(97_200_000 + seed as u64, 0.05);
        let mut x = (rng.next_u64() & 1) as usize;
        for t in 0..3 * LENGTH {
            let phase = t / LENGTH;
            let y = usize::from(rng.uniform() < rules[usize::from(phase == 1)][x]);
            let predictions = [filtered.predict(x), unguarded.predict(x), mixture.step(x, y, true).0];
            if phase > 0 && t % LENGTH < 100 {
                for k in 0..3 {
                    per_seed[seed][phase - 1][k] += usize::from(usize::from(predictions[k] > 0.5) == y);
                }
            }
            filtered.observe(x, y, true);
            unguarded.observe(x, y, true);
            x = y;
        }
        filtered.reset(); unguarded.reset(); mixture.reset();
        let mut probe = Rng(97_300_000 + seed as u64);
        let mut x = (probe.next_u64() & 1) as usize;
        for _ in 0..PROBE {
            let y = usize::from(probe.uniform() < rules[1][x]);
            let predictions = [filtered.predict(x), unguarded.predict(x), mixture.step(x, y, false).0];
            for k in 0..3 {
                per_seed[seed][2][k] += usize::from(usize::from(predictions[k] > 0.5) == y);
            }
            filtered.observe(x, y, false);
            unguarded.observe(x, y, false);
            x = y;
        }
        println!("seed={seed} stream={} init={} probe={} switch_b={:?} switch_a={:?} recall_b={:?}", 97_100_000 + seed, 97_200_000 + seed, 97_300_000 + seed, per_seed[seed][0], per_seed[seed][1], per_seed[seed][2]);
        for w in 0..3 { for k in 0..3 { totals[w][k] += per_seed[seed][w][k]; } }
    }
    for (w, label) in ["switch_to_b", "switch_to_a", "retention_b"].iter().enumerate() {
        let per_trial = if w == 2 { PROBE } else { 100 };
        let trials = N * per_trial;
        for (candidate, control, a, b) in [("consistency_filter", "unguarded", 0, 1),
                                           ("consistency_filter", "mixture", 0, 2),
                                           ("unguarded", "mixture", 1, 2)] {
            let differences: Vec<f64> = per_seed.iter().map(|c|
                (c[w][a] as f64 - c[w][b] as f64) / per_trial as f64).collect();
            let mean = differences.iter().sum::<f64>() / N as f64;
            let mut rng = Rng(97_400_000 + (w * 9 + a * 3 + b) as u64);
            let mut bootstrap = Vec::with_capacity(BOOTSTRAPS);
            for _ in 0..BOOTSTRAPS {
                let sum: f64 = (0..N).map(|_| differences[(rng.next_u64() % N as u64) as usize]).sum();
                bootstrap.push(sum / N as f64);
            }
            bootstrap.sort_by(|a, b| a.total_cmp(b));
            let (lo, hi) = (bootstrap[250], bootstrap[9749]);
            println!("paired {label} {candidate}-minus-{control}: mean={mean:.6} bootstrap95=[{lo:.6},{hi:.6}]");
            if w == 2 && a == 0 && b == 1 {
                let decision = if mean >= 0.01 && lo > 0.0 { "supports benefit" }
                    else if mean <= -0.01 && hi < 0.0 { "weighs against filter" }
                    else { "inconclusive" };
                println!("primary decision: {decision}");
            }
            if b == 2 { println!("mixture criterion {label} {candidate}: {}", if mean < -0.02 { "flag" } else { "pass" }); }
            println!("MURMUR_FOLLOWUP {{\"measure\":\"high_noise_200_{label}\",\"candidate\":\"{candidate}\",\"control\":\"{control}\",\"candidate_correct\":{},\"control_correct\":{},\"trials\":{trials}}}", totals[w][a], totals[w][b]);
        }
    }
}

// Prespecified before execution: 500 base seeds 98_000_000..=98_000_499.
// Evaluator mapping: training=base+40_000, probe B=base+48_000,
// initialization=base+51_000; bootstrap seed=98_100_000+outcome index.
// Public milestone-comparison settings: 1000 events per A/B/A phase,
// 1000 B probes, learning rate 0.05. Source revision: ad1c8baaf20b21049b359052ea3a907bdcb13f56 (candidate hash ac85654ec9b0c5f498a05f9ed1caabb7c6824c06d6fa9262722c8f1eeeb6ba2c).
// Available prior test records include 96-million and 97.1/97.2/97.3-million
// ranges; a global registry is unavailable, so freshness is not guaranteed.
#[test]
fn similar_rules_paired_followup() {
    const N: usize = 500;
    const LENGTH: usize = 1000;
    const BASE: u64 = 98_000_000;
    let rules = [[0.10, 0.70], [0.30, 0.90]];
    let mut counts = [[[0usize; 2]; 3]; N];
    for i in 0..N {
        let seed = BASE + i as u64;
        let mut candidate = Candidate::new(seed + 51_000, 0.05);
        let mut mixture = Mixture::new(seed + 51_000, 0.05);
        let mut rng = Rng(seed + 40_000);
        let mut x = (rng.next_u64() & 1) as usize;
        for t in 0..3 * LENGTH {
            let phase = t / LENGTH;
            let y = usize::from(rng.uniform() < rules[usize::from(phase == 1)][x]);
            let predictions = [candidate.predict(x), mixture.step(x, y, true).0];
            if phase > 0 && t % LENGTH < 100 {
                for k in 0..2 { counts[i][phase-1][k] += usize::from(usize::from(predictions[k] > 0.5) == y); }
            }
            candidate.observe(x, y, true);
            x = y;
        }
        candidate.reset(); mixture.reset();
        let mut rng = Rng(seed + 48_000);
        let mut x = (rng.next_u64() & 1) as usize;
        for _ in 0..LENGTH {
            let y = usize::from(rng.uniform() < rules[1][x]);
            let predictions = [candidate.predict(x), mixture.step(x, y, false).0];
            for k in 0..2 { counts[i][2][k] += usize::from(usize::from(predictions[k] > 0.5) == y); }
            candidate.observe(x, y, false);
            x = y;
        }
        println!("mapping seed={seed} train={} init={} probe_b={} switch_b={:?} switch_a={:?} recall_b={:?}", seed+40_000, seed+51_000, seed+48_000, counts[i][0], counts[i][1], counts[i][2]);
        println!("MURMUR_PAIRED {{\"measure\":\"similar_rules_retention_b\",\"candidate\":\"consistency_filter\",\"control\":\"mixture\",\"seed\":{seed},\"candidate_correct\":{},\"control_correct\":{},\"trials\":1000}}", counts[i][2][0], counts[i][2][1]);
    }
    for (w, name) in ["switch_to_b", "switch_to_a", "retention_b"].iter().enumerate() {
        let trials = if w == 2 { 1000 } else { 100 };
        let a: usize = counts.iter().map(|c| c[w][0]).sum();
        let b: usize = counts.iter().map(|c| c[w][1]).sum();
        let diffs: Vec<f64> = counts.iter().map(|c| (c[w][0] as f64-c[w][1] as f64)/trials as f64).collect();
        let mean = diffs.iter().sum::<f64>() / N as f64;
        let mut rng = Rng(98_100_000 + w as u64);
        let mut boot = Vec::with_capacity(10_000);
        for _ in 0..10_000 { boot.push((0..N).map(|_| diffs[(rng.next_u64() % N as u64) as usize]).sum::<f64>() / N as f64); }
        boot.sort_by(f64::total_cmp);
        let (lo, hi) = (boot[250], boot[9749]);
        println!("paired {name} mean={mean:.6} bootstrap95=[{lo:.6},{hi:.6}]");
        if w == 2 { println!("primary decision: {}", if lo > -0.02 { "non-inferiority supported" } else if hi < -0.02 { "loss beyond margin" } else { "inconclusive" }); }
        println!("MURMUR_FOLLOWUP {{\"measure\":\"similar_rules_{name}\",\"candidate\":\"consistency_filter\",\"control\":\"mixture\",\"candidate_correct\":{a},\"control_correct\":{b},\"trials\":{}}}", N*trials);
    }
}

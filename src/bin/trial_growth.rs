use murmur::{
    discovery::{HAZARD, RATE, SUSTAINED, THRESHOLD, WARMUP, WINDOW},
    trial_growth::{run, ResultSet, CANDIDATE_EVAL, CANDIDATE_TRAIN, COOLDOWN},
};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 3 {
        return Err("usage: trial_growth [seed] [output-directory] [penalty]".into());
    }
    let seed = match args.first() {
        Some(s) => s.parse()?,
        None => 101u64,
    };
    let penalty = match args.get(2) {
        Some(s) => s.parse()?,
        None => 10.0,
    };
    let dir = Path::new(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("results/trial-growth"),
    );
    fs::create_dir_all(dir)?;
    let mut metrics=String::from("seed,switching,noise,train_steps,test_steps,hazard,rate,warmup,window,threshold,sustained,candidate_train,candidate_eval,cooldown,penalty,trials_started,trials_completed,accepted,final_models,auto_test_accuracy,fixed_test_accuracy,single_test_accuracy,auto_rule_match,fixed_rule_match\n");
    for (name, switching, noise) in [
        ("switching", true, 0.1),
        ("stationary", false, 0.1),
        ("stationary_noisy", false, 0.25),
    ] {
        let r = run(seed, switching, noise, penalty)?;
        let n = r.test.len() as f64;
        let score = |k: usize| {
            r.test
                .iter()
                .filter(|row| [row.1, row.2, row.3][k] == row.0.target)
                .count() as f64
                / n
        };
        let match_rules = |models: &[murmur::sequence_engine::SequenceEngine]| -> f64 {
            let score = |swap: usize| {
                let mut n = 0;
                for context in 0..2 {
                    for source in 0..4 {
                        let target = if context == 0 {
                            (source + 1) % 4
                        } else {
                            (source + 3) % 4
                        };
                        let idx = if models.len() == 1 { 0 } else { context ^ swap };
                        n += usize::from(models[idx].predict_from(source) == target);
                    }
                }
                n as f64 / 8.0
            };
            score(0).max(score(1))
        };
        metrics.push_str(&format!("{seed},{switching},{noise},{},{},{HAZARD},{RATE},{WARMUP},{WINDOW},{THRESHOLD},{SUSTAINED},{CANDIDATE_TRAIN},{CANDIDATE_EVAL},{COOLDOWN},{penalty},{},{},{},{},{},{},{},{},{}\n",r.trace.len(),r.test.len(),r.trials.len(),r.trials.iter().filter(|t|t.completed).count(),r.trials.iter().filter(|t|t.accepted).count(),r.model.models.len(),score(0),score(1),score(2),match_rules(&r.model.models),match_rules(&r.fixed.models)));
        save(dir, name, &r)?;
    }
    fs::write(dir.join("metrics.csv"), &metrics)?;
    print!("{metrics}");
    Ok(())
}
fn save(dir: &Path, name: &str, r: &ResultSet) -> std::io::Result<()> {
    let mut trials =
        String::from("start,end,completed,control_minus_candidate_log_loss,penalty,accepted\n");
    for t in &r.trials {
        trials.push_str(&format!(
            "{},{},{},{},{},{}\n",
            t.start, t.end, t.completed, t.gain, t.penalty, t.accepted
        ));
    }
    fs::write(dir.join(format!("{name}-trials.csv")), trials)?;
    let mut trace = String::from(
        "step,true_context,source,target,control_prediction,surprise,mode,committed_models\n",
    );
    for t in &r.trace {
        trace.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            t.step, t.context, t.source, t.target, t.prediction, t.surprise, t.mode, t.model_count
        ));
    }
    fs::write(dir.join(format!("{name}-trace.csv")), trace)?;
    let mut log = String::from(
        "step,true_context,source,target,auto_prediction,fixed_prediction,single_prediction\n",
    );
    for (i, (e, a, b, c)) in r.test.iter().enumerate() {
        log.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            i + 1,
            e.context,
            e.source,
            e.target,
            a,
            b,
            c
        ));
    }
    fs::write(dir.join(format!("{name}-test.csv")), log)?;
    let mut weights = String::from("model,index,source,destination,weight\n");
    for (kind, models) in [
        ("auto", &r.model.models),
        ("fixed", &r.fixed.models),
        ("single", &r.single.models),
    ] {
        for (i, m) in models.iter().enumerate() {
            for (source, row) in m.weights.iter().enumerate() {
                for (dest, w) in row.iter().enumerate() {
                    weights.push_str(&format!("{kind},{i},{source},{dest},{w:.17}\n"));
                }
            }
        }
    }
    fs::write(dir.join(format!("{name}-weights.csv")), weights)?;
    Ok(())
}

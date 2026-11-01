use murmur::discovery::{
    run, Experiment, HAZARD, MAX_MODELS, RATE, SUSTAINED, THRESHOLD, WARMUP, WINDOW,
};
use std::{env, fs, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: discovery [seed] [output-directory]".into());
    }
    let seed = match args.first() {
        Some(s) => s.parse()?,
        None => 81u64,
    };
    let dir = Path::new(
        args.get(1)
            .map(String::as_str)
            .unwrap_or("results/discovery"),
    );
    fs::create_dir_all(dir)?;
    let mut metrics=String::from("seed,switching,noise,train_steps,eval_steps,hazard,rate,warmup,window,threshold,sustained,max_models,auto_models,auto_growth_step,auto_eval_accuracy,fixed_eval_accuracy,single_eval_accuracy,auto_context_alignment,fixed_context_alignment,auto_rule_match,fixed_rule_match\n");
    for (name, switching, noise) in [
        ("switching", true, 0.1),
        ("stationary", false, 0.1),
        ("stationary_noisy", false, 0.25),
    ] {
        let r = run(seed, switching, noise);
        let n = r.evaluation.len() as f64;
        let scores = |field: usize| {
            r.evaluation
                .iter()
                .filter(|row| [row.1, row.2, row.3][field] == row.0.target)
                .count() as f64
                / n
        };
        // Context IDs have no intrinsic meaning. Best permutation uses true labels
        // for scoring only, and cannot be interpreted as an online classifier.
        let alignment = |field: usize| {
            let same = r
                .evaluation
                .iter()
                .filter(|row| [row.4, row.5][field] == row.0.context)
                .count() as f64
                / n;
            same.max(1.0 - same)
        };
        let rule_match = |models: &[murmur::sequence_engine::SequenceEngine]| -> f64 {
            if models.len() < 2 {
                let mut correct = 0;
                for context in 0..2 {
                    for source in 0..4 {
                        let target = if context == 0 {
                            (source + 1) % 4
                        } else {
                            (source + 3) % 4
                        };
                        correct += usize::from(models[0].predict_from(source) == target);
                    }
                }
                return correct as f64 / 8.0;
            }
            let score = |swap: usize| {
                let mut correct = 0;
                for context in 0..2 {
                    for source in 0..4 {
                        let target = if context == 0 {
                            (source + 1) % 4
                        } else {
                            (source + 3) % 4
                        };
                        correct +=
                            usize::from(models[context ^ swap].predict_from(source) == target);
                    }
                }
                correct as f64 / 8.0
            };
            score(0).max(score(1))
        };
        let growth = r
            .auto
            .events
            .first()
            .map(|e| e.step.to_string())
            .unwrap_or_default();
        metrics.push_str(&format!("{seed},{switching},{noise},{},{},{HAZARD},{RATE},{WARMUP},{WINDOW},{THRESHOLD},{SUSTAINED},{MAX_MODELS},{},{},{},{},{},{},{},{},{}\n",r.rows.len(),r.evaluation.len(),r.auto.models.len(),growth,scores(0),scores(1),scores(2),alignment(0),alignment(1),rule_match(&r.auto.models),rule_match(&r.fixed.models)));
        save(dir, name, &r)?;
    }
    fs::write(dir.join("metrics.csv"), &metrics)?;
    print!("{metrics}");
    Ok(())
}
fn save(dir: &Path, name: &str, r: &Experiment) -> std::io::Result<()> {
    let mut rows=String::from("phase,step,true_context,source,target,auto_prediction,fixed_prediction,single_prediction,auto_models,auto_context_after_observation,fixed_context_after_observation,auto_surprise\n");
    for (i, (e, a, f, s, n, surprise)) in r.rows.iter().enumerate() {
        rows.push_str(&format!(
            "train,{},{},{},{},{},{},{},{},,,{}\n",
            i + 1,
            e.context,
            e.source,
            e.target,
            a,
            f,
            s,
            n,
            surprise
        ));
    }
    for (i, (e, a, f, s, ai, fi)) in r.evaluation.iter().enumerate() {
        rows.push_str(&format!(
            "eval,{},{},{},{},{},{},{},{},{},{},\n",
            i + 1,
            e.context,
            e.source,
            e.target,
            a,
            f,
            s,
            r.auto.models.len(),
            ai,
            fi
        ));
    }
    fs::write(dir.join(format!("{name}-predictions.csv")), rows)?;
    let mut events = String::from("step,mean_surprise,new_model_count\n");
    for e in &r.auto.events {
        events.push_str(&format!("{},{},{}\n", e.step, e.mean_surprise, e.new_count));
    }
    fs::write(dir.join(format!("{name}-events.csv")), events)?;
    let mut weights = String::from("model,index,source,destination,weight\n");
    for (kind, models) in [
        ("auto", &r.auto.models),
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

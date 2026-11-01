//! Compare repeated Storkey presentations with the predictive graph's budget.
use murmur_local_graph::{
    sequence::{graph_predict, Codebook, BITS, N},
    Graph, Rng,
};
use murmur_research_runner::{
    predictive_graph::{run_seed, EPOCHS},
    Config,
};
use std::{env, fs, path::Path, time::Instant};
const LIMIT: f64 = 1_000_000.0;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: training_budget <config.cfg> <new-output-directory>".into());
    }
    let c = Config::parse(&fs::read_to_string(&args[0])?)
        .map_err(|e| format!("invalid configuration: {e}"))?;
    let dir = Path::new(&args[1]);
    if let Some(p) = dir.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(p)?;
    }
    fs::create_dir(dir)?;
    let mut csv=String::from("seed,patterns,epochs,repeated_status,epochs_completed,max_abs_weight,repeated_correct,trials,repeated_accuracy,predictive_correct,no_feedback_correct,storkey_once_correct,lookup_correct,training_ns\n");
    let mut pred=String::from("seed,sequence_index,trial,target,repeated_prediction,predictive_prediction,no_feedback_prediction,storkey_once_prediction,lookup_prediction\n");
    for &seed in &c.seeds {
        let base = run_seed(&c, seed);
        let book = Codebook::new(seed.wrapping_add(1000));
        let mut g = Graph::new(N, c.density, seed);
        let mut rng = Rng(seed.wrapping_add(16000));
        let mut completed = 0;
        let mut max_weight = 0.0f64;
        let start = Instant::now();
        for epoch in 0..EPOCHS {
            let mut order: Vec<_> = (0..base.stored.len()).collect();
            for i in (1..order.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                order.swap(i, j);
            }
            let mut valid = true;
            for i in order {
                // Consume the same random initialization draws as predictive training.
                for _ in 0..BITS {
                    rng.next_u64();
                }
                g.learn_storkey(&book.encode(&base.stored[i]));
                if g.weights.iter().any(|x| !x.is_finite() || x.abs() > LIMIT) {
                    valid = false;
                    break;
                }
            }
            if !valid {
                break;
            }
            completed = epoch + 1;
            max_weight = g.weights.iter().fold(0.0f64, |a, x| a.max(x.abs()));
        }
        let training_ns = start.elapsed().as_nanos();
        let ok = completed == EPOCHS;
        let mut counts = [0usize; 5];
        for row in &base.rows {
            let p = &row.source;
            let repeated = if ok {
                Some(graph_predict(&g, &p.observed, &p.missing, &book, c.sweeps))
            } else {
                None
            };
            if let Some(v) = repeated {
                counts[0] += usize::from(v == p.target);
            }
            for (j, v) in [row.full, row.no_feedback, p.storkey, p.lookup]
                .iter()
                .enumerate()
            {
                counts[j + 1] += usize::from(*v == p.target);
            }
            pred.push_str(&format!(
                "{seed},{},{},{},{},{},{},{},{}\n",
                p.sequence_index,
                p.trial,
                p.target,
                repeated.map(|x| x.to_string()).unwrap_or_default(),
                row.full,
                row.no_feedback,
                p.storkey,
                p.lookup
            ));
        }
        let n = base.rows.len();
        let correct = if ok {
            counts[0].to_string()
        } else {
            String::new()
        };
        let accuracy = if ok {
            (counts[0] as f64 / n as f64).to_string()
        } else {
            String::new()
        };
        csv.push_str(&format!("{seed},{},{},{},{completed},{max_weight},{correct},{n},{accuracy},{},{},{},{},{training_ns}\n",c.patterns,EPOCHS,if ok {"complete"}else{"nonfinite_or_limit"},counts[1],counts[2],counts[3],counts[4]));
    }
    fs::write(dir.join("resolved.cfg"), c.resolved())?;
    fs::write(dir.join("metrics.csv"), &csv)?;
    fs::write(dir.join("predictions.csv"), pred)?;
    fs::write(dir.join("protocol.txt"),format!("epochs={}\nstorkey_repeat_guard={}\ntraining_order=shuffled with same seed and random draws as predictive model\nsplit=training sequences reused; evaluation uses fresh corruptions\nsource_revision=unrecorded\nnixpkgs_revision=unrecorded\n",EPOCHS,LIMIT))?;
    print!("{csv}");
    Ok(())
}

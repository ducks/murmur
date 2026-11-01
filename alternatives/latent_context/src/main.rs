//! Bounded Bayesian online change-point filter with Dirichlet transition rows.
use murmur::discovery::{stream_with_block, Discovery};
use std::{env, fs, path::Path};
const N: usize = 4;
// Above all benchmark stream lengths so reported runs do not truncate hypotheses.
const CAP: usize = 8192;
const HAZARD: f64 = 0.01;
#[derive(Clone)]
struct Hypothesis {
    probability: f64,
    counts: [[u32; N]; N],
}
struct Bocpd {
    hypotheses: Vec<Hypothesis>,
    last_change: f64,
    last_discarded: f64,
}
impl Bocpd {
    fn new() -> Self {
        Self {
            hypotheses: vec![Hypothesis {
                probability: 1.0,
                counts: [[1; N]; N],
            }],
            last_change: 0.0,
            last_discarded: 0.0,
        }
    }
    fn distribution(&self, source: usize) -> [f64; N] {
        let mut out = [HAZARD / N as f64; N];
        for h in &self.hypotheses {
            let sum = h.counts[source].iter().sum::<u32>() as f64;
            for (target, x) in out.iter_mut().enumerate() {
                *x += (1.0 - HAZARD) * h.probability * h.counts[source][target] as f64 / sum;
            }
        }
        out
    }
    fn observe(&mut self, source: usize, target: usize) {
        let mut changed = Hypothesis {
            probability: HAZARD / N as f64,
            counts: [[1; N]; N],
        };
        changed.counts[source][target] += 1;
        let mut next = Vec::with_capacity((self.hypotheses.len() + 1).min(CAP));
        next.push(changed);
        // Truncate oldest hypotheses, then renormalize; approximation is explicit.
        for h in self.hypotheses.iter().take(CAP - 1) {
            let mut g = h.clone();
            let row = &h.counts[source];
            let sum = row.iter().sum::<u32>() as f64;
            g.probability *= (1.0 - HAZARD) * row[target] as f64 / sum;
            g.counts[source][target] += 1;
            next.push(g);
        }
        let total: f64 = next.iter().map(|h| h.probability).sum();
        let omitted: f64 = self
            .hypotheses
            .iter()
            .skip(CAP - 1)
            .map(|h| {
                let sum = h.counts[source].iter().sum::<u32>() as f64;
                h.probability * (1.0 - HAZARD) * h.counts[source][target] as f64 / sum
            })
            .sum();
        self.last_discarded = omitted / (total + omitted);
        for h in &mut next {
            h.probability /= total;
        }
        self.last_change = next[0].probability;
        self.hypotheses = next;
    }
}
fn argmax(v: &[f64; N]) -> usize {
    let mut b = 0;
    for i in 1..N {
        if v[i] > v[b] {
            b = i;
        }
    }
    b
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: murmur-latent-context-prototype [seed] [output-dir]".into());
    }
    let seed = match args.first() {
        Some(s) => s.parse()?,
        None => 701,
    };
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("results"));
    fs::create_dir_all(dir)?;
    let mut csv=String::from("seed,case,block,noise,steps,bocpd_accuracy,bocpd_log_loss,fixed_two_accuracy,single_accuracy,mean_change_probability,change_boundary_probability,mean_discarded_mass,max_discarded_mass\n");
    for (case, switch, block, noise, steps) in [
        ("switch50", true, 50, 0.1, 1200),
        ("switch100", true, 100, 0.1, 1200),
        ("switch200", true, 200, 0.1, 1200),
        ("stationary10", false, 100, 0.1, 5000),
        ("stationary25", false, 100, 0.25, 5000),
    ] {
        let mut b = Bocpd::new();
        let mut fixed = Discovery::new(2, false, seed + 8001);
        let mut single = Discovery::new(1, false, seed + 8001);
        let (mut bc, mut fc, mut sc) = (0, 0, 0);
        let (mut nll, mut cp, mut edge_cp, mut edges) = (0.0, 0.0, 0.0, 0usize);
        let (mut discarded, mut max_discarded) = (0.0_f64, 0.0_f64);
        for (i, e) in stream_with_block(seed + 8002, steps, switch, noise, block)
            .into_iter()
            .enumerate()
        {
            let p = b.distribution(e.source);
            bc += usize::from(argmax(&p) == e.target);
            nll -= p[e.target].ln();
            let f = fixed.step(e.source, e.target, true);
            let s = single.step(e.source, e.target, true);
            fc += usize::from(f.prediction == e.target);
            sc += usize::from(s.prediction == e.target);
            b.observe(e.source, e.target);
            cp += b.last_change;
            discarded += b.last_discarded;
            max_discarded = max_discarded.max(b.last_discarded);
            if switch && i > 0 && i % block == 0 {
                edge_cp += b.last_change;
                edges += 1;
            }
        }
        csv.push_str(&format!(
            "{seed},{case},{block},{noise},{steps},{},{},{},{},{},{},{},{}\n",
            bc as f64 / steps as f64,
            nll / steps as f64,
            fc as f64 / steps as f64,
            sc as f64 / steps as f64,
            cp / steps as f64,
            if edges == 0 {
                0.0
            } else {
                edge_cp / edges as f64
            },
            discarded / steps as f64,
            max_discarded
        ));
    }
    fs::write(dir.join("metrics.csv"), &csv)?;
    print!("{csv}");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn normalized_and_deterministic() {
        let mut a = Bocpd::new();
        for i in 0..500 {
            let p = a.distribution(i % 4);
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            a.observe(i % 4, (i + 1) % 4);
            assert!((a.hypotheses.iter().map(|h| h.probability).sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(a.hypotheses.len() <= CAP);
        }
    }
    #[test]
    fn learning_raises_probability_of_predictable_transition() {
        let mut b = Bocpd::new();
        for _ in 0..100 {
            b.observe(0, 1);
        }
        assert!(b.distribution(0)[1] > 0.9);
    }
}

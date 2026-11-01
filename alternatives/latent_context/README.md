# Alternative: Bayesian run-length model

This prototype tests a different way to handle unlabelled changes: instead of assuming two recurring rule memories, maintain probabilities over the number of transitions since the latest change. Each hypothesis has a separate Dirichlet-smoothed transition table. Before each transition, a 1% change hazard mixes in a fresh uniform table; after observing the target, Bayesian weights update and each hypothesis records the transition. The mechanism never receives true context labels. This adapts the established approach of [Adams and MacKay, Bayesian Online Changepoint Detection (2007)](https://arxiv.org/abs/0710.3742) to four-symbol Markov transition rows. It is not an architectural novelty claim or a local self-organizing graph.

Hypotheses are capped at 8,192; when the cap is reached, the oldest hypothesis is discarded and the remaining mass is renormalized. This is an approximation beyond the cap. The program logs discarded posterior mass. The reported runs contain at most 5,000 transitions, so **no truncation occurs**. Exact per-step computation grows with run length, so these results are not compute-matched to the baselines. This filter resets statistical memory after an inferred change rather than learning to recognize a recurring old rule.

For each stream, compare it with the main project's online two-memory `Discovery` model (no growth) and a single-memory model. All predict each next symbol before observing it and then learn on that same transition. Runs use the same generator/stream, 1,200 transitions for switching rules and 5,000 for stationary rules; there is no separate frozen test set. Fixed-two and single models are online as well. Thus this evaluates online predictive adaptation, **not** the parent project's frozen-evaluation accuracy, nor are the methods resource matched. Hazard 0.01, symmetric count-one priors, and the 8,192 cap were not tuned on an independent development set; observed results are exploratory.

```sh
cd /home/urho/dev/murmur
nix-shell --run 'cargo test --manifest-path alternatives/latent_context/Cargo.toml && cargo build --release --manifest-path alternatives/latent_context/Cargo.toml'
nix-shell --run 'for s in 701 702 703 704 705 706 707 708 709 710; do alternatives/latent_context/target/release/murmur-latent-context-prototype "$s" "alternatives/latent_context/results/$s"; done'
```

Mean online accuracy across 10 seeds:

| Stream | Run-length model | Fixed two-memory mixture | Single-memory model |
|---|---:|---:|---:|
| Rules switch every 50 steps | 0.796 | 0.836 | 0.620 |
| Rules switch every 100 steps | 0.856 | 0.846 | 0.745 |
| Rules switch every 200 steps | 0.879 | 0.848 | 0.827 |
| Stationary, 10% noise, 5,000 steps | 0.896 | 0.896 | 0.897 |
| Stationary, 25% noise, 5,000 steps | 0.743 | 0.745 | 0.744 |

Switching streams have 10% noise. No significance estimates, compute timings, memory budgets, change-detection precision/recall, or generalization to unknown grammars were measured. The new model performed worse on 50-step switches but better in the 200-step case in this small, non-preregistered study; this is not proof of superiority. Prediction log loss is also saved for this model; equivalent baseline losses are not available in the current `Discovery` API. `mean_change_probability` and `change_boundary_probability` are calculated **after** seeing the transition; boundary values use hidden labels only for analysis and do not give a false-alarm rate or pre-observation prediction. A complete detection study would set thresholds using independent development data, measure alarms on long stationary streams and detection delays, and compare against adequately budgeted and independently implemented baselines.

Two tests check normalization, bounded hypothesis count, and learning on a predictable transition. CSV files are written to the chosen output directory; generated results and build artifacts are ignored by the local `.gitignore`.

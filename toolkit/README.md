# Research runner: first slice

This small offline Rust runner compares the existing Hebbian graph, a Storkey-style graph, the Hebbian output-feedback ablation, nearest-prefix lookup, an untrained graph, and a majority-symbol baseline. All receive identical generated cues. It is purpose-built for one experiment; it is not yet a general model plugin framework. It has no network or third-party Rust dependencies.

From the repository root:

```sh
nix-shell
cargo test --manifest-path toolkit/Cargo.toml
cargo run --release --manifest-path toolkit/Cargo.toml -- toolkit/configs/sequence.cfg toolkit/runs/my-first-run
```

Choose a **new** output directory for each run. The runner refuses any existing destination and does not silently replace evidence. If interrupted, `status.txt` can remain `running`; a caught output-write error is marked `failed` where possible. Configuration is validated before the destination is created. A partially created directory may need inspection or manual removal before rerunning.

## Configuration

`configs/sequence.cfg` is the example. Version 1 uses strict UTF-8 `key=value` lines, comma-separated seeds, and optional whole-line `#` comments. All seven keys are required; unknown or repeated keys are errors. `patterns` must be 1–64; `prefix_noise` and `density` must be finite values in [0,1]; `trials_per_sequence` and `sweeps` must be positive. The alphabet and representation are fixed by the existing task: four positions, four symbols, 32 bits per position. Change a copy of the config to run different settings.

The seed `s` initializes the graph; `s+1000`, `s+2000`, and `s+3000` (wrapping u64 arithmetic) seed the codebook, stored sequences, and evaluation-cue generator. These are separate deterministic streams, not a mathematical independence guarantee. All underlying sequences in evaluation were stored during training; evaluation uses new corruptions and random missing-block initializations. This tests retrieval, not unseen-sequence or compositional generalization.

## Artifacts

- `resolved.cfg`: the parsed configuration, serialized with explicit values.
- `provenance.txt`: runner version, platform, seed scheme, and split description. Source and Nix revisions are explicitly unrecorded; the Nix channel is not pinned.
- `training.csv`: the generated stored symbol sequences per seed.
- `predictions.csv`: the exact noisy prefix bits and missing-block initialization, target, and all models' predictions per trial. `1` and `0` encode unit values +1 and −1. These are evaluation artifacts and include the answer for later analysis.
- `metrics.csv`: correct counts and accuracy by seed and model.
- `diagnostics.csv`: target-state oracle field reversals and output-bit counts for the two trained graphs. The true target is used for diagnosis only.
- `resources.csv`: per-seed training and total prediction wall time, and model-owned heap-payload estimates for the two graphs and lookup. These exclude shared data, temporary memory, allocation overhead, and other costs; lookup has no separately timed fitting step. Measurements are not resource-matched benchmarks.
- `summary.md`: pooled accuracy and a reminder of the experiment's limitations.
- `status.txt`: `running`, `complete`, or `failed` when the status write succeeds.

`toolkit/runs` is ignored by Git. The files are evidence for analysis, not restartable model checkpoints. The graph uses a dense weight array with sparse nonzero connections; lookup stores sequences and has a different resource profile. Peak memory measurement, model serialization, model selection, truly held-out sequence splits, generic plugins, and Hugging Face integration are not implemented. See [EXPERIMENTS.md](../EXPERIMENTS.md) for timing and payload-estimate scopes and the correction to the first 0.2.0 measurement.

The existing legacy evaluator remains in `alternatives/local_graph`. Tests check that its trained graph, lookup, untrained graph and majority counts match the runner on identical seeds and settings, and that the ablation receives the same observed inputs. For the example configuration the first converted run also matched all 40 per-seed legacy accuracy values across those four models.

## Sharded graph experiment

The separate `sharded` binary compares one graph with four smaller-load graphs, with either an oracle assignment or a nearest-stored-prefix router, against direct lookup. It runs Hebbian and Storkey-style learning on paired cues. This is externally assigned, fixed partitioning—not learned routing or growth.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin sharded -- toolkit/configs/sharded/test.cfg toolkit/runs/my-sharded-run'
```

The destination must not exist. `protocol.txt`, `training.csv`, `predictions.csv`, `metrics.csv`, `resources.csv`, `summary.md` and `status.txt` explain and preserve the run. The resource estimates are model-owned heap payloads, not peak process memory; routed timings include prefix search plus one graph query, but exclude index construction, graph construction and shared data. The router itself stores prefixes, while direct lookup stores full training sequences. Each query is a new corruption of a **stored** sequence. See [EXPERIMENTS.md](../EXPERIMENTS.md) for scope and results.

## Matched connection-storage experiment

`matched_budget` compares the four-shard system with a single 256-unit graph using duplicated input blocks. `independent_encoding` runs a separate sensitivity control with independent extra codes and noisy bits; it supplies more noisy evidence to the large graph. The dense edge-array payload matches four 128-unit graphs; the router adds storage, and unit counts and computation are not matched.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin matched_budget -- toolkit/configs/matched/test-noisy.cfg toolkit/runs/my-matched-run'
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin independent_encoding -- toolkit/configs/matched/test-noisy.cfg toolkit/runs/my-independent-run'
```

Each destination must be new. Run-level protocol, predictions, metrics, approximate resource measurements and status are saved; see [EXPERIMENTS.md](../EXPERIMENTS.md) for limitations and the fact that the independent-code control was added after inspecting initial results.

## Supervised learned-router experiment

`learned_router` trains a small linear classifier on externally assigned shard labels and compares its routing and downstream graph predictions with the oracle shard, nearest-prefix search and lookup. It does not learn the assignment or generalize to unseen stored prefixes. `router_diagnostic` is an exploratory development-only check with longer training on clean prefixes.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin learned_router -- toolkit/configs/learned-router/test.cfg toolkit/runs/my-learned-router'
```

The destination must be new. The runner saves labels, router coefficients, predictions, per-seed metrics, timings and parameter-size estimates; see [EXPERIMENTS.md](../EXPERIMENTS.md) for protocol and limitations.

## Local prediction-error graph

`predictive_graph` compares an experimental directed recurrent graph and a no-feedback ablation against the existing one-pass graph rules and lookup. It trains each output unit with an explicitly supplied target bit and saves convergence and learned-weight diagnostics. `training_budget` is a separate 30-presentation Storkey control; its numerical guard and timing scope are documented in [EXPERIMENTS.md](../EXPERIMENTS.md).

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin predictive_graph -- toolkit/configs/predictive/test-32.cfg toolkit/runs/my-predictive-run'
```

The destination must be new. This tests retrieval of stored sequences, not generalization to unseen sequences. The full graph has directed weights and no Hopfield-energy convergence guarantee.

`settling_analysis` summarizes an existing predictive-graph predictions file by whether the full graph settled, alongside the feedback-free model's accuracy on the same queries. This is a descriptive, post-hoc grouping, not a causal test:

```sh
nix-shell --run 'cargo run --quiet --release --manifest-path toolkit/Cargo.toml --bin settling_analysis -- toolkit/runs/predictive-test-32/predictions.csv'
```

The input run must exist locally; outputs are printed as CSV. See [EXPERIMENTS.md](../EXPERIMENTS.md).

## Frozen-weight dynamics interventions

`dynamics` retrains the existing predictive graph deterministically, then compares fixed-order, reversed, seeded-shuffled, longer-running, and thresholded updates on the same frozen weights and cues. It records fixed-order cycles and checks stability under the original rule separately from threshold-based stopping. It reuses the earlier exploratory seed groups; see [EXPERIMENTS.md](../EXPERIMENTS.md).

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin dynamics -- toolkit/configs/predictive/test-32.cfg toolkit/runs/my-dynamics-run'
```

The destination must be new. Per-seed metrics, per-query outcomes and protocol are saved locally; the outputs are not restartable checkpoints.

## Symmetric-feedback graph

`symmetric_feedback` compares local-error graphs trained with directed, symmetric, or absent output feedback. It records settling and accuracy on shared stored-sequence cues; the symmetric energy property assumes frozen weights and clamped inputs. See [EXPERIMENTS.md](../EXPERIMENTS.md).

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin symmetric_feedback -- toolkit/configs/symmetric/test-32.cfg toolkit/runs/my-symmetric-run'
```

## Continuous-stream delayed recall

`temporal_stream` uses the earlier fixed recurrent reservoir and separately trained readout to recall prior bits from continuous streams. It compares matched edge-free and reset-state graph controls with an explicit causal shift register and majority baseline. The task-specific configuration is versioned separately in `configs/temporal/`; the dev/test seed sets are checked for cross-group stream-seed collisions by tests.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin temporal_stream -- toolkit/configs/temporal/test.cfg toolkit/runs/my-temporal-run'
```

The output directory must be new. The runner logs streams, per-query predictions, per-seed metrics, resolved configuration and protocol. It does not save a restartable model or measure actual buffer memory; see [EXPERIMENTS.md](../EXPERIMENTS.md) for the corrected seed split, protocol, results and limitations.

## Temporal connection plasticity

`temporal_plasticity` compares frozen recurrent edges against eligibility-trace learning with task feedback and unrelated-feedback control, on paired continuous streams. It logs edge weights and training drift alongside held-out predictions; see [EXPERIMENTS.md](../EXPERIMENTS.md) for the limitations and results.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin temporal_plasticity -- toolkit/configs/temporal-plasticity/test.cfg toolkit/runs/my-plasticity-run'
```

The output directory must be new.

## Matched frozen perturbations

`matched_perturbation` compares task-trained connection directions against random directions with the same per-edge displacement, with new readouts trained on both frozen graphs. See [EXPERIMENTS.md](../EXPERIMENTS.md) for its two-stage training protocol and limitations.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin matched_perturbation -- toolkit/configs/matched-perturbation/test.cfg toolkit/runs/my-matched-run'
```

## Matched online feedback

`matched_online` compares task feedback with unrelated feedback whose global edge-update norm is matched to the task model at each step. This cross-model normalization is a diagnostic intervention; see [EXPERIMENTS.md](../EXPERIMENTS.md).

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin matched_online -- toolkit/configs/matched-online/test.cfg toolkit/runs/my-online-match'
```

## Connection strength and impulse persistence

`connection_strength` compares fixed graphs with scaled initial recurrent weights, online feedback learners, and a shift register. It also records finite-horizon responses to a one-bit difference. See [EXPERIMENTS.md](../EXPERIMENTS.md) for clipping, the probe protocol, and limits on interpretation.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin connection_strength -- toolkit/configs/connection-strength/test.cfg toolkit/runs/my-strength-run'
```

## Trained-graph strength intervention

`trained_strength` scales trained random-feedback graphs back to their initial mean absolute edge weight and scales initial graphs up to match trained strength, then trains fresh readers. It logs weights, impulse curves and scaling diagnostics. See [EXPERIMENTS.md](../EXPERIMENTS.md) for the two-pass protocol and limitations.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin trained_strength -- toolkit/configs/trained-strength/test.cfg toolkit/runs/my-trained-strength-run'
```

## Weight placement interventions

`weight_organization` compares a trained frozen graph with three global weight shuffles and three per-destination shuffles, preserving the exact signed-weight multiset and endpoints. `scripts/weight_organization_summary.py` reports paired seed-level bootstrap intervals. See [EXPERIMENTS.md](../EXPERIMENTS.md) for protocol and caveats.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin weight_organization -- toolkit/configs/weight-organization/test.cfg toolkit/runs/my-placement-run'
nix-shell --run 'python3 toolkit/scripts/weight_organization_summary.py toolkit/runs/my-placement-run/metrics.csv'
```

## Centered random feedback

`centered_feedback` compares the original random-target-minus-reader-prediction edge signal against random-target-minus-0.5, with paired streams and true-label readers. It records training signals, edge drift, impulse curves and frozen/online recall; see [EXPERIMENTS.md](../EXPERIMENTS.md).

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin centered_feedback -- toolkit/configs/centered-feedback/test.cfg toolkit/runs/my-centered-run'
nix-shell --run 'python3 toolkit/scripts/centered_feedback_summary.py toolkit/runs/my-centered-run/metrics.csv'
```

## Reader-only feedback

`reader_signal` compares the reader-dependent term `0.5 - probability` with centered random feedback, their original combined rule, and task-feedback baselines. `scripts/reader_signal_summary.py` reports seed-level paired bootstrap intervals and edge-change alignment; see [EXPERIMENTS.md](../EXPERIMENTS.md) for the full two-pass protocol and caveats.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin reader_signal -- toolkit/configs/reader-signal/test.cfg toolkit/runs/my-reader-run'
nix-shell --run 'python3 toolkit/scripts/reader_signal_summary.py toolkit/runs/my-reader-run/metrics.csv toolkit/runs/my-reader-run/weights.csv'
```

## Reader-trained graph transfer

`reader_transfer` pretrains a graph with reader-only feedback on one delay, then trains a fresh reader on a different, separately generated stream and target delay. It includes random-label, strength-matched and fixed-strength controls; see [EXPERIMENTS.md](../EXPERIMENTS.md).

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin reader_transfer -- toolkit/configs/reader-transfer/test.cfg toolkit/runs/my-transfer-run'
nix-shell --run 'python3 toolkit/scripts/reader_transfer_summary.py toolkit/runs/my-transfer-run/metrics.csv'
```

## Pattern detection transfer

`pattern_recognition` tests graphs pretrained on delayed recall with a new `101` detection reader, recording confusion metrics and exact/majority controls. `pattern_reweight` follows up with seven sequential positive reader updates and a direct three-bit reader; this follow-up was selected after the original protocol failed. See [EXPERIMENTS.md](../EXPERIMENTS.md).

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin pattern_recognition -- toolkit/configs/pattern-recognition/test.cfg toolkit/runs/my-pattern-run'
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin pattern_reweight -- toolkit/configs/pattern-reweight/test.cfg toolkit/runs/my-reweight-run'
nix-shell --run 'python3 toolkit/scripts/pattern_summary.py reweight toolkit/runs/my-reweight-run/metrics.csv'
```

## Continual A–B–A rule changes

`continual_rule_change` compares fixed and reader-feedback graphs with recency Markov and oracle-context Markov models as noisy binary transition rules change and return. Frozen probes test retention without updating weights, and 200-step windows show online adaptation. See [EXPERIMENTS.md](../EXPERIMENTS.md) for context asymmetry, reset behavior and post-hoc training-tail probes.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin continual_rule_change -- toolkit/configs/continual-rule-change/test.cfg toolkit/runs/my-continual-run'
nix-shell --run 'python3 toolkit/scripts/continual_rule_change_summary.py toolkit/runs/my-continual-run/probes.csv toolkit/runs/my-continual-run/windows.csv toolkit/runs/my-continual-run/training_tail_probes.csv'
```

## Unlabeled context reuse

`unlabeled_context` compares a fixed two-slot online hidden-state mixture with a single recency Markov model, a known-rule filter and a labelled-context oracle across stationary, A–B–A, and regularly switching streams at two noise levels. It logs prior/posterior beliefs, slot snapshots, frozen probes and online log loss. See [EXPERIMENTS.md](../EXPERIMENTS.md) for additional baseline knowledge and limitations.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin unlabeled_context -- toolkit/configs/unlabeled-context/test.cfg toolkit/runs/my-unlabeled-run'
nix-shell --run 'python3 toolkit/scripts/unlabeled_context_summary.py toolkit/runs/my-unlabeled-run/probes.csv toolkit/runs/my-unlabeled-run/windows.csv toolkit/runs/my-unlabeled-run/slots.csv'
```

## Manifest-driven coordinator

The local coordinator is the first slice of an agent-facing research workflow. It freezes a JSON plan and separate development/evaluation configs; checks required models, comparison fields, seed-role separation, budgets and source fingerprint; runs the registered Rust adapter offline; validates complete paired evidence; and reports whether the **declared operational criteria** were met. It registers `unlabeled_context` and the separate `candidate_context` evaluator. It does not generate hypotheses or modify code itself: an agent can use `next` to decide whether to run development, run evaluation, revise the plan, inspect a failure, or review completed evidence.

```sh
nix-shell --run 'python3 -B -m unittest discover -s toolkit/coordinator/tests -v'
nix-shell --run 'python3 -B -m toolkit.coordinator.runner prepare toolkit/plans/context-reuse.json toolkit/runs/my-coordinated-study'
nix-shell --run 'python3 -B -m toolkit.coordinator.runner next toolkit/runs/my-coordinated-study'
nix-shell --run 'python3 -B -m toolkit.coordinator.runner run toolkit/runs/my-coordinated-study development'
nix-shell --run 'python3 -B -m toolkit.coordinator.runner next toolkit/runs/my-coordinated-study'
# If development criteria are met:
nix-shell --run 'python3 -B -m toolkit.coordinator.runner run toolkit/runs/my-coordinated-study evaluation'
```

After preparation, `nix-shell --run 'python3 -B -m toolkit.coordinator.runner cycle toolkit/runs/my-coordinated-study'` runs the eligible stages automatically and stops at a review, revision, or error. It does not propose the next hypothesis.

The output directory must be new and, if inside the repository, Git-ignored. Preparation copies and hashes the manifest and configs, captures Git HEAD, working-tree status, a hash of tracked/non-ignored source, tool versions and Nix environment. Running rechecks the source and frozen files before and after execution. Each stage preserves stdout/stderr, status, a machine-readable decision and the original adapter evidence; completed stage files are hashed and checked when the workflow resumes. Changed source requires preparing a new plan; failed or interrupted stage directories are retained and never overwritten.

The lock and local artifacts can be edited by anyone with filesystem access; they are not tamper-proof. The evaluation config is visible from preparation onward, and nothing prevents a researcher or agent from inspecting it or repeatedly designing new plans around prior results. Seed separation and frozen criteria are safeguards, not holdout secrecy, preregistration, statistical significance, or scientific correctness. The example tests an already-known model comparison, rather than discovering a new model. Logical-transition and wall-time limits are declared resource safeguards, not peak-memory measurements or fair-compute matching. Nixpkgs is not pinned. Source status is recorded, but dirty source content is fingerprinted rather than archived; commit code before preparing a study if a reconstructible revision is needed.

An agent using this workflow should write the scientific plan first, keep development results and failed runs, avoid altering the frozen plan, and treat `criteria_met` as a prompt to inspect evidence rather than a claim of success. If criteria fail, document the result and prepare a new plan with fresh evaluation seeds when a new hypothesis is ready. See [EXPERIMENTS.md](../EXPERIMENTS.md) for the underlying adapter's design and limitations.

For a quick integration smoke check of both stages, `toolkit/plans/coordinator-smoke.json` uses deliberately permissive thresholds and tiny seed groups. Its `criteria_met` means only that the pipeline processed valid evidence; it is not evidence of a model improvement. The first real example study recorded a development failure; see [RESEARCH.md](../RESEARCH.md). The agent-facing procedure is in [AGENT_WORKFLOW.md](coordinator/AGENT_WORKFLOW.md).

## Autonomous candidate campaigns

The first autonomous campaign lets Codex propose, implement, test and evaluate its own model candidates against the existing hidden-context controls. It is bounded by a maximum number of ideas and per-agent/test wall times. Each idea gets an isolated Git worktree and branch, an agent transcript, a proposed hypothesis and falsification, independent deterministic development/evaluation seeds, tests, a versioned candidate commit, and a coordinator study. Failed development criteria stop that idea's evaluation; later ideas receive the recorded outcome. A failed agent, test or infrastructure step stops the campaign for inspection rather than discarding its work. Nothing is automatically merged.

Ideas in the same campaign share a campaign-owned Cargo target directory for agent commands, tests, and evaluation. Cargo decides which builds can be reused; different source paths or changes may still require recompilation. A completed round's worktree is removed only when its branch still points to the recorded commit, its sealed evidence is intact, and it contains no changes or unexpected ignored files. The candidate branch, commits, logs, and study remain. The shared build cache is removed when the whole campaign completes; it is kept for paused or failed campaigns. Cleanup decisions are recorded in `cleanup.json`. To inspect a candidate again, check out its recorded branch or commit into a new worktree. To apply the same checks to an older campaign, run:

```sh
nix-shell --run 'python3 -B -m toolkit.coordinator.campaign cleanup toolkit/runs/my-campaign'
```

```sh
nix-shell --run 'python3 -B -m toolkit.coordinator.campaign init toolkit/runs/my-campaign --max-ideas 3 --agent-seconds 600'
nix-shell --run 'python3 -B -m toolkit.coordinator.campaign run toolkit/runs/my-campaign'
```

The starting worktree must be clean. The `codex` CLI must be installed and configured. By default it may use a hosted model and incur its usual service costs; a configured local provider can be selected with `--local-provider ollama` or `--local-provider lmstudio` (and `--model` if needed). The candidate model and experiments themselves use local Rust code and no pretrained model or external model API.

The first campaign is intentionally limited to one binary hidden-context task and a fixed interface in `toolkit/src/agent_candidate.rs`. The agent may edit the candidate and its new test, while the separate evaluator and reference controls remain unchanged. The supervisor compares it with the original mixture on noisy A–B–A retention and stationary penalties. The agent can explore different candidate implementations within this interface; discovering broad new language architectures needs additional tasks and evaluator adapters. Automated checks and the coding-agent sandbox are not proof that model code is honest or secure: review candidate code and evidence. The agent is instructed not to use network, pretrained models, rule labels or future inputs; the harness controls the inputs it passes but cannot prove arbitrary code obeys that instruction.

Seeds are generated from a campaign identifier and separated across rounds; there is no global registry guaranteeing distinctness across every possible campaign. Evaluation is gated on the declared development criteria, but seed construction and results are locally visible and repeated campaigns can still overfit. The driver does not enforce a token or monetary budget. It stops on failure and preserves the worktree and logs; if a process is killed, inspect the `.running.lock` and round artifacts before manually clearing the lock. See [AGENT_WORKFLOW.md](coordinator/AGENT_WORKFLOW.md) for research discipline and [RESEARCH.md](../RESEARCH.md) for prior evidence.

To start a separate campaign with the evidence from a previous one in the agent's context, pass `--prior-campaign toolkit/runs/previous-campaign` to `init`. The driver summarizes all final probe conditions, so a narrow success can be seen alongside regressions. For candidate campaigns, the coordinator now also rejects a candidate whose mean final A/B probe accuracy drops more than three percentage points versus the original mixture in any tested regime/noise condition (excluding the unseen B counterfactual after stationary-A training). This is still a limited safeguard, not a universal definition of model quality.

Campaign round results include the coding agent's wall time and, when the CLI reports it, token-usage fields extracted from its event log. They are usage records, not a cost estimate or an enforced token budget. Detailed event logs may include code and experiment information sent to the configured agent provider.

A possible longer-term direction for volunteer compute, and the additional verification and isolation it would need, is described in [DISTRIBUTED_VISION.md](coordinator/DISTRIBUTED_VISION.md). Distributed execution is not implemented.

### Automated prior and adaptation comparison

Run the fixed matched-seed prior ablation and write per-seed metrics, 100-step online windows, aggregate summaries, configuration, and a source-hash manifest with one command:

```sh
nix-shell --run 'python3 toolkit/scripts/run_candidate_ablation.py toolkit/configs/candidate-ablation/matched.cfg toolkit/runs/prior-ablation'
```

The output directory must not exist. The script uses the local Rust runner and no coding-agent API; it does not search ideas or declare success automatically. It compares both original candidate snapshots, neutral equal-slot and seeded random-slot versions, the mixture and single-memory baselines, and a labelled-context oracle. Only the oracle receives context labels. Equal slots test exact symmetry; random offsets test symmetry breaking without encoding repeat/flip rules, while decay targets the neutral prior. Windows are measured from each phase boundary, including the two switches. The tasks remain the four fixed binary transition families; see `EXPERIMENTS.md` for interpretation and limitations.

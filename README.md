# Murmur

Murmur is an experimental research toolkit for exploring alternative machine-learning models and the process of testing them. It combines small Rust experiments in sequence memory and graph dynamics with a coordinator that can propose bounded milestones, run candidate experiments, record and review evidence, and replay paired results.

The work is exploratory. The models have been tested on narrow synthetic tasks and have not demonstrated a competitive or novel general-purpose architecture. The coordinator's agent reviews are fallible; seed allocation and replay improve traceability and reproducibility, not scientific validity. See [RESEARCH.md](RESEARCH.md) and [EXPERIMENTS.md](EXPERIMENTS.md) for observations, negative results, and limitations.

The original graph experiments below run locally without external services. Optional automated campaigns invoke the configured Codex CLI and may use a remote model service, incur usage, and execute agent-generated code. Their input-token threshold is checked between rounds, not a hard spending cap. See the [toolkit documentation](toolkit/README.md) and [agent workflow](toolkit/coordinator/AGENT_WORKFLOW.md) before running them.

Enter the Nix shell with `nix-shell`, then run:

```sh
cargo test
cargo run --release -- 42 results/seed-42
cargo run --release --bin plasticity -- 11 results/plasticity-example 8
cargo run --release --bin probe -- 21 results/probe-example
cargo run --release --bin robustness -- 31 results/robustness-example
cargo run --release --bin engine -- 51 results/engine-example
cargo run --release --bin context -- 61 results/context-example
cargo run --release --bin hidden -- 71 results/hidden-example
cargo run --release --bin discovery -- 81 results/discovery-example
cargo run --release --bin trial_growth -- 201 results/trial-example 40
cargo run --release --bin stress -- 301 results/stress-example
cargo run --release --bin sequential -- 601 results/sequential-example
```

Or use `nix-shell --run 'cargo test'`. `shell.nix` uses the local `nixpkgs` channel and is not pinned; record its revision for cross-machine replication. The original Rust experiments have no external service dependency. Optional CLI arguments are seed, output directory, and the positive number of marker steps (default 1). Other settings are in `Config` in `src/experiment.rs`; the JSON output records all settings. Running into the same directory overwrites its files.

`metrics.json` records accuracy and configuration, `topology.csv` contains directed edges (including possible duplicate and self edges), and `activations.csv` samples unit 0 for the first 16 test trials. The topology file alone is not a full model checkpoint: input weights and trained readout are regenerated only by rerunning with the seed. No structural-change log is produced because topology is fixed.

The `plasticity` binary compares frozen, learning, and one-trial-shifted-feedback graphs; see [EXPERIMENTS.md](EXPERIMENTS.md) for the rule, diagnostics, and limitations.

The `probe` binary compares the original reader with a separately fitted linear probe on frozen graph states, with optional observation perturbations built into the experiment.

The `robustness` binary evaluates varied distractor sequences and noise inside the graph; see [EXPERIMENTS.md](EXPERIMENTS.md).

The `engine` binary demonstrates online transition learning, a rule change, and retention against frozen and cumulative Markov controls. Its reusable API is in `src/sequence_engine.rs`; this initial engine is equivalent to a recency-weighted Markov model.

The `context` binary demonstrates explicitly cued, separate transition memories against context-free and context-aware baselines. The reusable engine is in `src/context_engine.rs`; it does not infer context.

The `hidden` binary uses Bayesian filtering to infer which of two previously labelled-trained contexts is active, with uncertainty and standard baselines.

The `discovery` binary explores unlabeled rule learning and surprise-triggered memory growth, including stationary false-alarm controls.

The `trial_growth` binary compares a temporary new memory with the current model on later observations before committing it.

The `stress` binary varies switching and stream lengths and compares growth decisions with a separate CUSUM-style alarm.

The `sequential` binary explores increasing the penalty after repeated candidate trials; it does not provide formal error control.

Independent exploratory prototypes and their protocols are in [alternatives/local_graph](alternatives/local_graph/README.md), [alternatives/latent_context](alternatives/latent_context/README.md), and [alternatives/change_detection](alternatives/change_detection/README.md). Their metrics are not directly comparable.

The local graph prototype also has a sequence-completion experiment against a lookup baseline; see [its README](alternatives/local_graph/README.md).

Graph-interference diagnostics are described in the [local graph README](alternatives/local_graph/README.md).

[Toolkit proposal](TOOLKIT.md): a reusable workflow and minimal research-runner design for comparing new model ideas.

The [research runner first slice](toolkit/README.md) runs the local-graph sequence comparison from a versioned config and saves matched inputs and per-model results.

[Visual research story](STORY.md): an accessible map of the project’s progression and current evidence.

The [continuous-stream experiment](toolkit/README.md#continuous-stream-delayed-recall) tests short-term memory with and without graph connections.

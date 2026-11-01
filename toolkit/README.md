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

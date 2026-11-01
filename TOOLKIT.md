# A toolkit for experimental model research

The broader toolkit described here is a proposal. A first task-specific runner now exists in [toolkit/](toolkit/README.md); the other pieces remain future work. The goal is to make it easier to invent, implement, test, understand, and compare computational models. Murmur is the first possible user, but a graph, neural network, probabilistic model, or ordinary program should all be able to participate. The toolkit should make negative results and failed implementations as visible as successes.

## The research loop

1. **Name a question.** State what a model is supposed to do, what mechanism might produce it, and which observation would challenge the idea. Link relevant prior work and say what is borrowed.
2. **Make the smallest model and task.** Start with an inspectable implementation and a synthetic problem whose answer is known. Separate what the model sees during training, prediction, and evaluation.
3. **Choose controls before looking.** Include a simple baseline, a mechanism ablation, and conditions that should expose failure. For an adaptive model, include stationary/noisy data as well as actual changes.
4. **Run and inspect.** Record configuration, seeds, data splits, predictions, training signals, failures, and costs. Verify determinism and that the tests exercise real behavior.
5. **Evaluate on separate data.** Use development runs to choose parameters, then use held-out seeds and, when relevant, withheld structures or later time periods. Describe any post-hoc analysis as exploratory.
6. **Interpret and decide.** Compare mechanisms and budgets, document limitations and alternative explanations, and decide whether to extend, revise, or abandon the idea. A result can be useful even when the model fails.

This is a guide, not a mandatory gate or proof of scientific validity. Some exploratory work should happen before an experiment can be fully specified; label it honestly.

## What the toolkit would provide

| Part | Job | Example |
|---|---|---|
| Experiment specification | Describe model, dataset, protocol, seeds, and output location in a versioned file. | Compare Hebbian and Storkey-style updates on identical graphs and cues. |
| Model adapter | Give an experiment a small common way to initialize, train or observe, predict, reset state, and report resource use. | A Rust graph and a Python baseline can each implement an adapter. |
| Dataset and split tools | Generate or import examples, preserve provenance, and check separation. | Keep entire grammar combinations out of training, not merely random rows. |
| Baselines and ablations | Run simple references and controlled variants on the same inputs. | Lookup memory, fixed graph, and graph without internal learning. |
| Runner and artifacts | Execute a named configuration and save inputs, predictions, metrics, logs, snapshots, and execution status. | A run directory that can be inspected or regenerated without a web service. |
| Analysis and reports | Compare runs, show uncertainty and failure cases, and link each claim to its evidence. | Accuracy versus memory load, including per-seed outcomes and cost. |
| Research notebook | Record hypotheses, prior work, decisions, negative results, and next questions. | A human-readable report alongside machine-readable results. |

Adapters should expose only operations the experiment actually needs. Not every model has gradient training, probabilities, a graph, a fixed parameter count, or a serializable state. Optional capabilities should be explicit, and unsupported metrics should be marked unavailable rather than invented. The toolkit must distinguish model state (temporary activity), learned parameters, structural changes, and evaluation-only information.

## A minimal experiment contract

A useful first contract would describe:

- **Question:** the behavior being tested and the comparison that would change our mind.
- **Models:** model type, learning settings, initial seed, and declared capabilities.
- **Data:** generator or local file, exact parameters, separate training/development/test seeds or splits, and any known overlaps.
- **Protocol:** prediction timing, when updates are allowed, whether state resets, and which labels are visible. Explicitly say if evaluation is frozen or online.
- **Measurements:** primary metric, baselines, additional diagnostics, computation and memory measurement method, and failure conditions.
- **Reproduction:** command, source revision when available, environment identifier, output paths, and machine-readable status. Record missing information instead of implying a complete provenance record.

The runner should save the resolved configuration, per-example or per-step predictions when appropriate, aggregate metrics, and error/status information. Training or graph logs and model snapshots should be optional and clearly identified; an edge listing is not a restartable checkpoint unless all required state is present. A timing result should include measurement scope and environment; compare budgets explicitly, and avoid treating different tasks as a leaderboard.

## Design boundaries

- Run locally without external services. A Nix shell should be available, with pinning when cross-machine reproducibility becomes a priority.
- Keep implementations separate from the runner. Avoid requiring a specific neural architecture or Python framework.
- Make the core learning and inference rules inspectable. Use libraries for plumbing when they reduce work without hiding the experiment.
- Treat Hugging Face Datasets as an optional Python adapter for importing/exporting data, not a required runtime, model source, or training method. Local CSV/JSON is enough for the first slice.
- Keep model, dataset, and evaluation seeds separate. Record whether splits are truly disjoint at the level relevant to the claim: new corruptions of known patterns, new sequences, and new compositions test different things.
- State how hyperparameters were selected, including searches and failures. Repeated tests can inflate false alarms; a heuristic threshold is not a formal error guarantee.
- Preserve raw evidence where practical and say when output is ignored by Git, overwritten, incomplete, or too large to retain.

## First vertical slice

Do not build a general plugin system yet. Extract one existing Murmur task into a small, offline reproducible runner:

1. Choose the local-graph sequence-completion task, including its lookup baseline, and freeze its existing behavior as a regression reference.
2. Define a versioned configuration with model/learning rule, dataset parameters, seeds, and evaluation settings. Start with one data format and one command.
3. Run at least two model variants and the lookup on the same generated examples. Keep training examples and evaluation corruptions explicit; do not call them unseen sequences.
4. Write resolved config, per-seed metrics, predictions or errors, and a short summary to a new run directory. Fail clearly on invalid configuration and avoid silently overwriting previous evidence.
5. Verify with tests for split integrity, determinism, and consistent inputs across models; reproduce the existing baseline before adding another learning rule.

If that slice proves useful, add separate held-out composition tasks, a richer model interface, resource measurements, and optional Hugging Face interoperability. If it adds more ceremony than value, simplify it rather than generalizing it.

## First slice delivered

The [sequence-completion runner](toolkit/README.md) reads a versioned local config, compares six variants on paired inputs, refuses to overwrite an existing run, and records resolved config, training sequences, predictions, metrics, summary, provenance gaps, and status. Tests compare it to the legacy evaluator and verify reproducibility and non-overwrite behavior. It is task-specific rather than a common adapter or plugin API.

## What exists in this repository today

Murmur already has seeded generators, Rust model prototypes, per-experiment commands, tests, JSON/CSV outputs, and [RESEARCH.md](RESEARCH.md), [EXPERIMENTS.md](EXPERIMENTS.md), and [LEARNING.md](LEARNING.md). Some models are in separate prototypes and use different protocols. There is one task-specific runner and configuration format but no common multi-task runner or unified model interface; generated results are ignored by Git, and the Nix channel is not pinned. Those gaps are opportunities for a toolkit, not features already delivered.

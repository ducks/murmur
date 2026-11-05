# Experiments

## Cue–marker memory, first milestone

Each independent trial selects a random cue (0 or 1), presents that cue then marker 2, and predicts the cue as the next symbol. State is reset at the start of every trial. Train for 1,000 trials and evaluate 1,000 new trials from a separately seeded PRNG stream; training is online and test weights are frozen. Both possible patterns occur in training and test, so the split evaluates new random draws, not unseen sequence types. The Markov baseline uses the majority target after marker 2; ties choose 0. A separately trained reset-state control erases state between cue and marker. The generator is deterministic SplitMix64; graph, training and test use seeds `s`, `s+1`, and `s+2` respectively. These are distinct deterministic streams, not a formal independence guarantee. Seed and task were not pre-registered.

```sh
nix-shell --run 'cargo test'
nix-shell --run 'for s in 1 2 3 4 5 42; do cargo run --release -- "$s" "results/seed-$s"; done'
```

Observed locally on 2026-10-31, 1,000 test trials per seed:

| Seed | Graph | Reset | Markov |
|---:|---:|---:|---:|
| 1 | 1.000 | 0.470 | 0.470 |
| 2 | 1.000 | 0.483 | 0.483 |
| 3 | 1.000 | 0.510 | 0.490 |
| 4 | 1.000 | 0.493 | 0.493 |
| 5 | 1.000 | 0.514 | 0.486 |
| 42 | 1.000 | 0.511 | 0.511 |

The graph readout learned this one-step delayed association for these seeds; destroying state eliminated that advantage. This does not isolate recurrent edges from unit self-retention. It also does not show generalization beyond the two observed patterns. No timings, memory measurements, growth comparison, or other experiments are claimed at this stage. Model has 512 fixed recurrent edges, 384 fixed input weights, and 258 trainable readout parameters; Markov has two counts. These are not matched budgets. The prescribed falsification would be failure to exceed both controls consistently on this task; the observations do not falsify that narrow memory hypothesis, but do not validate the broader self-organization hypothesis either.

## Delayed memory and edge-free control

Run the same cue task with 1, 2, 4, 8, 12, 16, or 24 repetitions of marker 2 before prediction. The delay is fixed for each run. The edge-free model starts from a clone of the graph, then removes every recurrent edge; its input weights, leak, state transition, training trials and separate readout's learning rule match the full graph. The reset control erases state after the cue and then processes the same marker sequence. Models see identical trial streams within a run. Each row below is the mean over seeds 1–5, with 1,000 test trials each.

```sh
nix-shell --run 'cargo build --release; for d in 1 2 4 8 12 16 24; do for s in 1 2 3 4 5; do target/release/murmur "$s" "results/delay-$d-seed-$s" "$d"; done; done'
```

| Marker steps | Graph | Edge-free | Reset | Markov |
|---:|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | 0.494 | 0.484 |
| 2 | 1.000 | 1.000 | 0.506 | 0.484 |
| 4 | 1.000 | 1.000 | 0.506 | 0.484 |
| 8 | 0.609 | 0.609 | 0.513 | 0.484 |
| 12 | 0.513 | 0.513 | 0.513 | 0.484 |
| 16 | 0.513 | 0.513 | 0.513 | 0.484 |
| 24 | 0.513 | 0.513 | 0.513 | 0.484 |

At delay 8, both graph and edge-free achieved 1.000 only for seed 2 and fell near chance for the other four; averaging conceals this abrupt difference. For every tested seed and delay the two models obtained the same accuracy, though that does not mean their internal states were identical. The original shortest-delay result can be explained without recurrent edges. By delay 12 neither trained readout decoded the cue better than its reset control in these runs. This is a failure of the **trained system**, not proof that the state has no cue information: decaying signals, the training rule, finite data, and readout bias can all matter. Results are exploratory and do not establish that edges never help on other tasks or parameter settings.

## Learning within a fixed graph

This exploratory experiment tests whether a simple edge-update rule improves the cue–marker task. The three models start from identical graph and readout parameters and receive the same training trials. All train their own readout. The frozen graph never updates edges; the learning graph uses current-trial prediction feedback; the mismatched-feedback control uses its own previous trial's feedback, with zero feedback on the first trial. This is a one-trial shift, not a random permutation, and can have temporal correlations; it does not perfectly isolate all possible feedback effects. Trials reset state and traces. Training uses 1,000 trials; evaluation freezes all weights and uses 1,000 fresh trials. Delays 4, 8, and 12 were chosen using the earlier delay sweep. Seeds 11–20 had not been used in that sweep; neither these choices nor the rule were preregistered.

For each graph edge `i → j` and simulation step `t`, the eligibility trace follows `e[t] = 0.8 * e[t-1] + state_i[t-1] * state_j[t]`. At trial end the learning model updates the edge by `0.01 * error * output_weight_1[j] * e`, then clamps the edge to [-0.5, 0.5]. `error = target - sigmoid(class_1_score)` and the output weight and error are measured before the readout learns that trial. This is a heuristic three-factor rule, not backpropagation or a demonstrated gradient; the output feedback is global supervised information. Trace decay and learning rate were chosen for this exploratory run, not optimized. Mismatched feedback follows the same rule with the previous error. Both plasticity models compute their own errors and readout weights; their error distributions and edge-change magnitudes can differ.

```sh
nix-shell --run 'cargo test && cargo build --release --bin plasticity'
nix-shell --run 'for d in 4 8 12; do for s in 11 12 13 14 15 16 17 18 19 20; do target/release/plasticity "$s" "results/plasticity-$d-$s" "$d"; done; done'
```

Means over 10 seeds, each with 1,000 held-out trials:

| Delay | Frozen accuracy | Learning accuracy | Mismatched accuracy | Mean absolute edge change: learning / mismatched |
|---:|---:|---:|---:|---:|
| 4 | 1.000 | 1.000 | 1.000 | 0.0285 / 0.0034 |
| 8 | 0.598 | 0.598 | 0.598 | 0.0244 / 0.0239 |
| 12 | 0.496 | 0.496 | 0.496 | 0.0358 / 0.0364 |

At delay 8, two seeds achieved 1.000 and eight were near chance; each of the three models gave the same accuracy on every individual seed at all three delays. Edge changes are mean absolute final differences from initial weights, not a sum of updates. No improvement in accuracy was observed, despite real weight changes. This result evaluates this rule and budget, not all graph-learning rules.

To investigate, `diagnostics.csv` records the Euclidean distance between graph states after cue 0 and cue 1, and the class-1-minus-class-0 readout score for each cue. In the frozen graph, mean state distance fell from 0.576 at delay 4 to 0.136 at delay 8 and 0.035 at delay 12. At delay 12 the two score margins had the same sign in all ten seeds for all three models, so each model made the same prediction for both cues. The state distance remained nonzero: this supports both a weakening representation and a decision-boundary issue, but does not establish which is the primary cause or how well an optimally trained probe could decode it. The diagnostics were added after observing the accuracy results and are exploratory.

Each output directory contains JSON configuration and metrics, initial/learned/mismatched edge snapshots, a per-100-trial edge-change log, and diagnostics. The snapshots do not contain input or readout weights and are not restartable checkpoints. Existing output directories are overwritten.

## Can a fresh reader recover the cue?

This follow-up uses the original frozen random graph, not the graph trained by the plasticity experiment. First, the original online readout trains on 1,000 clean cue trials. Separately, a linear nearest-centroid probe fits 1,000 clean states generated from a distinct seeded stream: it averages the states for cue 0 and cue 1, then predicts whichever mean is closer in Euclidean distance. Both classes must appear in probe training. This is a linear classifier because the difference of squared distances to two fixed centroids is linear in the input. The graph, both readers, and probe centroids are frozen during evaluation. Both readers evaluate the exact same 1,000 test trials generated from another seeded stream.

For each evaluation state only, independently add uniform noise in `[-a, a]` to each unit's observed value, using `a` below. Noise is generated separately from the cue streams and the same pseudorandom samples are reused across amplitudes. It is added **after** graph computation and is not a simulation of noisy graph dynamics. Graph, online-training, evaluation, probe-training, and noise seeds are respectively `s`, `s+1`, `s+2`, `s+3`, and `s+4`; these are distinct deterministic streams, not a formal independence proof. Both readers train on clean observations. Delay, noise levels, and the seed range were exploratory, not preregistered. The online reader and probe differ in training algorithm, data stream, and objective, so this comparison does not isolate a single cause of their performance difference.

```sh
nix-shell --run 'cargo test && cargo build --release --bin probe'
nix-shell --run 'for s in 21 22 23 24 25 26 27 28 29 30; do target/release/probe "$s" "results/probe-$s"; done'
```

Mean held-out accuracy across 10 seeds, with 1,000 evaluation trials each:

| Marker steps | Noise half-width | Online reader | Centroid probe |
|---:|---:|---:|---:|
| 4 | 0 | 1.000 | 1.000 |
| 8 | 0 | 0.496 | 1.000 |
| 12 | 0 | 0.496 | 1.000 |
| 24 | 0 | 0.496 | 1.000 |
| 12 | 0.01 | 0.496 | 0.996 |
| 24 | 0.0001 | 0.496 | 1.000 |
| 24 | 0.001 | 0.496 | 0.689 |
| 24 | 0.01 | 0.496 | 0.519 |

Other evaluated amplitudes are in `results/probe-*/metrics.csv`. At delay 24 the mean distance between the two clean graph states was about 0.00058, versus 0.032 at delay 12. The clean probe can distinguish the two deterministic states even when their difference is tiny, while the online readout predicts near chance. Thus, on this task, some cue information survives and a different reader can recover it, even though the original reader does not. Added observation noise erases much of that advantage at delay 24. The clean result is particularly easy: each reset trial has one of only two deterministic states, both seen repeatedly in probe training. It is not evidence for robust memory, unseen-pattern generalization, or a benefit from learned graph connections. The experiment used new graph seeds but was selected after earlier results; no claim of preregistered confirmation is made.

## Varied inputs and internal disturbances

This tests a fixed random graph, not the graph-learning rule. Every trial starts with cue 0 or 1, followed by `delay` distractors. The repeated condition always uses symbol 2; the varied condition samples symbols 2, 3, and 4 independently. The graph has five input symbols in both conditions and identical graph initialization for a given seed. State resets at the start of each trial.

For varied sequences, a deterministic FNV-1a hash of the entire distractor sequence assigns it to the test split if `hash % 5 == 0`; all other sequences belong to training. This split depends only on the distractors, not the cue. Consequently no complete distractor sequence can occur in both splits, even with the other cue. Random draws may repeat within a split. The hash partition can cause distribution differences, particularly with only 3^4 possible length-four sequences; this is a stringent but not necessarily representative split. The repeated condition has only one distractor sequence and therefore does not have this separation.

After **every** graph transition, including the cue step, the experiment independently adds uniform noise in `[-a,a]` to each unit and clamps state to [-1,1]. At `a=0` no noise is applied. Fresh random samples are used for each training and test trial; the same seeded samples are reused across noise amplitudes within a condition. Each condition trains both the online readout and centroid probe on the **same** 1,000 graph states, then freezes them and tests both on the same 1,000 separately generated states. Training and test use separate seeded streams for sequences and for noise. The fixed graph is regenerated for each condition. We use 10 previously unused graph seeds (31–40). Delay and noise levels were exploratory, not preregistered.

```sh
nix-shell --run 'cargo test && cargo build --release --bin robustness'
nix-shell --run 'for s in 31 32 33 34 35 36 37 38 39 40; do target/release/robustness "$s" "results/robustness-$s"; done'
```

Mean held-out accuracy across seeds; each cell is online reader / centroid probe:

| Marker steps | Repeated, no noise | Varied, no noise | Repeated, noise 0.01 | Varied, noise 0.01 |
|---:|---:|---:|---:|---:|
| 4 | 1.000 / 1.000 | 1.000 / 0.763 | 1.000 / 1.000 | 1.000 / 0.758 |
| 8 | 0.597 / 1.000 | 0.565 / 0.522 | 0.602 / 1.000 | 0.565 / 0.522 |
| 12 | 0.496 / 1.000 | 0.505 / 0.501 | 0.496 / 0.961 | 0.506 / 0.501 |
| 24 | 0.496 / 1.000 | 0.508 / 0.501 | 0.496 / 0.508 | 0.508 / 0.501 |

At noise 0.001, the repeated-condition probe scored 1.000, 1.000, 1.000, and 0.616 at delays 4, 8, 12, and 24 respectively. Full per-seed metrics are saved in CSV. The probe's earlier clean result does not carry over to longer, unseen varied distractor sequences: the tested readers are near chance by delay 12. The online readout outperformed the centroid probe on the four-step varied task, illustrating that this simple probe is not an upper bound on what a linear classifier can decode. A near-chance probe does not prove no cue information survives. On the repeated task, internal noise undermined probe accuracy substantially at 24 steps; with varied sequences both readers were already near chance at longer delays without added noise. These results do not evaluate other graph parameters, optimal decoding, internal learning, or structural adaptation.

## Online sequence engine: adaptation and retention

The engine is a graph of four symbol units with one directed edge for each ordered pair, including self edges. The current unit is the previous symbol; its strongest outgoing edge predicts the next one. On observing the next symbol, only that unit's outgoing row changes: `w[from,to] += 0.15 * (target[to] - w[from,to])`, where the observed destination has target 1 and the other destinations have target 0. Edges begin uniformly at 0.25. Weights, rather than activity traces, retain learned information between observations. This is a recency-weighted first-order Markov model, equivalent to an exponential moving average of next-symbol frequencies for each source. It is a simple stepping stone, not evidence that the earlier 128-unit dynamical graph learned, and does not claim novelty or self-organization.

The benchmark generates one continuous four-symbol stream. Initially, the likely successor is the next symbol clockwise (e.g. 0→1→2→3→0); after 600 transitions, the likely successor becomes counterclockwise. On each transition, with probability 0.1 the destination is instead sampled uniformly from the three other destinations, including the current symbol. Thus the most likely next symbol has probability 0.9. Models predict **before** receiving the next symbol and update afterward. All see the same stream. The frozen comparison uses the same graph update during the first 600 transitions, then freezes its weights; the ordinary Markov baseline retains cumulative transition counts with one pseudo-count per edge and continues learning, but does not discount old observations. All models retain the immediately preceding symbol at the phase boundary.

Old/new rule recall is evaluated at the end by querying all four possible sources against the deterministic dominant rule for each phase; these are counterfactual queries and do not modify state. Since old and new rules assign different successors to the same source and no context identifies the phase, a single deterministic prediction cannot satisfy both simultaneously. Recovery is the first post-change step at which the trailing 50 predictions contain at least 80% correct; this is a threshold crossing, not a guarantee that performance remains above threshold. `null` means the threshold was not reached within 400 changed transitions.

```sh
nix-shell --run 'cargo test && cargo build --release --bin engine'
nix-shell --run 'for s in 51 52 53 54 55 56 57 58 59 60; do target/release/engine "$s" "results/engine-$s"; done'
```

Means over ten seeds, 600 training and 400 changed transitions each:

| Model | Training accuracy | Post-change accuracy | Old-rule recall after change | New-rule recall after change | Recovery step |
|---|---:|---:|---:|---:|---:|
| Adaptive graph | 0.900 | 0.860 | 0.000 | 1.000 | 62.7 |
| Frozen graph | 0.900 | 0.032 | 1.000 | 0.000 | not reached |
| Cumulative Markov | 0.900 | 0.032 | 1.000 | 0.000 | not reached |

The adaptive graph crossed the recovery threshold in all ten runs (58–67 steps). The cumulative baseline's 600-transition history is longer than the changed phase, so it is unsurprising it did not switch within this specific 400-transition observation period; this does not mean it could never adapt. A recency-weighted Markov model would be mathematically equivalent to the engine. The fixed and adaptive graphs have 16 edge weights; the count baseline has 16 counts. This is not a computation- or memory-matched benchmark, nor a comparison against a general sequence model. No wall-clock or memory claims are made. The experiment is exploratory, and seed, rule, and hyperparameters were not preregistered.

The binary outputs `metrics.json` with configuration and metrics, `predictions.csv` containing each prediction made before learning, and `before_change.csv` / `after_change.csv` for the adaptive edge weights. The files are analysis snapshots, not restartable checkpoints; reuse of an output directory overwrites them.

## Explicit context and retention

The context engine receives an externally supplied context ID **before each prediction**. It maintains a separate set of outgoing transition weights for every `(context, source symbol)` pair; only the row for the current context and source is updated on a transition. It retains one shared previous-symbol state across context switches. The context supplied on `observe(context, symbol)` specifies the transition *into* that symbol. The model does not infer or remember the context cue itself; its caller must supply it for every prediction. This is equivalent to a context-conditioned, recency-weighted first-order Markov model, with no structural growth or learned routing.

Use the same four-symbol process and 10% transition noise as the previous engine experiment. Train 600 transitions under rule A with cue 0, then 400 under rule B with cue 1. Before revisiting A, query each source against both dominant rules without updating weights. Then run four alternating 50-transition blocks A/B/A/B and query both rules again. Each transition is scored before learning, and all models see the same stream. Compare (1) the context engine, (2) the previous context-free recency-weighted engine, and (3) cumulative Markov counts with separate tables for the two explicit contexts and a one-count prior per edge. Models 1 and 3 receive the cue; model 2 does not. This tests the benefit of supplying context, not the isolated effect of parameter count or the advantage of graph learning over an equivalent context-aware baseline. The context-aware models each store 32 weights/counts, compared with 16 for the context-free engine.

```sh
nix-shell --run 'cargo test && cargo build --release --bin context'
nix-shell --run 'for s in 61 62 63 64 65 66 67 68 69 70; do target/release/context "$s" "results/context-$s"; done'
```

Mean over ten seeds; each accuracy is prequential:

| Model | Phase A | Phase B | Alternating blocks | A recall after B | B recall after B |
|---|---:|---:|---:|---:|---:|
| Context graph | 0.893 | 0.900 | 0.899 | 1.000 | 1.000 |
| Context-free graph | 0.893 | 0.866 | 0.601 | 0.000 | 1.000 |
| Context-aware counts | 0.893 | 0.900 | 0.899 | 1.000 | 1.000 |

The context-aware graph and context-aware counts both recalled both dominant rules in every seed immediately after B and again after the alternating phase. The graph's A edges are untouched during B, so A retention here follows directly from the hard routing design. Alternating accuracy for the context-free engine includes adaptation lag on each switch; its final recall reflects the last B block. These are deterministic rule queries, not unseen-combination or language generalization. Because both context-aware models get the correct external cue and separate tables, this result does not demonstrate self-organized context, inference from an unlabeled stream, distributed representations, or superiority to a context-conditioned Markov model. Seed range and protocol were exploratory, not preregistered.

The binary produces `metrics.json`, per-step `predictions.csv`, and graph edge snapshots `after_a.csv` and `after_b.csv`. Snapshots are for analysis, not restartable checkpoints. Outputs in an existing directory are overwritten.

## Inferring an unannounced context

This isolates context inference after **supervised** rule learning. As in the previous experiment, the model first receives 600 labelled training transitions under rule A followed by 400 labelled transitions under B. Its two context-conditioned edge tables are then frozen. A separate context-conditioned count model is fitted on exactly the same labelled training transitions with one pseudo-count per destination and is also frozen. The context-free recency-weighted engine is trained on the same data and continues updating during evaluation; it has no context information. The context oracle uses the graph's frozen tables and receives the true context for each evaluation transition.

Evaluation starts from a fresh random symbol and separate seeded stream. It contains eight 50-transition blocks alternating A/B, with 10% transition noise as before. The graph and count filters are **not given the actual context or block index**. Each starts with equal probability for A and B, assumes a fixed 0.02 probability of switching on any transition, predicts the next symbol using the probability-weighted mixture of the two tables, then updates context probabilities from the actual observed transition. The assumed hazard corresponds to an average of 50 steps in a stochastic switching model, but the test switches deterministically every 50 steps; the filters do not use the deterministic schedule. The underlying rule memories, two-context count, and switching model are provided in advance: this is a simple hidden Markov model, not discovered structure. The true label is used only for the oracle, metrics, and log. Probability tables and graph weights are not updated during evaluation, so this does not test unsupervised rule learning or new-context discovery.

Context accuracy and Brier score are calculated **after observing** each transition, while symbol prediction accuracy is measured **before** seeing it. Brier score averages `(probability of A − actual A label)^2`, so lower is better. The per-step CSV records the graph's probability before and after seeing the symbol, enabling inspection without confusing the two.

```sh
nix-shell --run 'cargo test && cargo build --release --bin hidden'
nix-shell --run 'for s in 71 72 73 74 75 76 77 78 79 80; do target/release/hidden "$s" "results/hidden-$s"; done'
```

Means across ten seeds, each with 400 evaluation transitions:

| Model | Next-symbol accuracy | Context accuracy after observation | Context Brier score |
|---|---:|---:|---:|
| Graph-table hidden-state filter | 0.843 | 0.943 | 0.046 |
| Count-table hidden-state filter | 0.865 | 0.976 | 0.018 |
| Context oracle using graph tables | 0.902 | supplied | — |
| Context-free online engine | 0.613 | not estimated | — |

The hidden filter's very first prediction after each unannounced switch was correct in 6 of 70 switch events (8.6%); it often expects the old rule at that moment. After seeing that first symbol, its most likely context was correct in 44 of 70 events (62.9%). These values refer to different moments: the latter is not a prediction made in advance. Learned graph tables use a recency-weighted update and the count tables use all labelled training data, so their difference does not isolate inference alone. The favorable oracle has access to the true context and the graph tables, and the context-free engine is the only model updating during evaluation. This small synthetic task, fixed block length, chosen hazard, and ten seeds do not establish robustness to unknown or changing rules. The count baseline performed better in this test. Runs are exploratory and not preregistered.

The binary writes `metrics.json`, `predictions.csv` (including true context for later analysis and the filters' probabilities), and `trained_edges.csv`; the latter is an analysis snapshot, not a restartable checkpoint. Reusing an output directory overwrites it.

## Unsupervised rule memories with surprise-gated growth

This is a first test of discovering known *types* of structure without rule labels. The learner starts with one transition table. It predicts using a probability-weighted mixture of its tables, updates the hidden-state posterior after seeing each symbol, and updates each table's row toward the observed transition in proportion to that posterior. It assumes a 0.02 probability of switching states per transition. This is a standard-style online hidden-state mixture, not locally emerging dynamics in the earlier 128-unit graph. Two-state capacity is a hard-coded **experimental safeguard in this prototype**, not a demonstrated general growth architecture; merging and pruning are not implemented.

Growth is a simple heuristic: after the first 100 observations, average the negative log probability assigned to the actual symbol in non-overlapping windows of 25. If two consecutive window means exceed 0.85, add a second table, initialize each row with a randomly favored destination, reset state probabilities to uniform, and log the event. The cap prevents further growth. Updates and growth use only observed symbols, model probabilities, and the specified random seed—never the true rule label. The threshold, rate (0.15), and other settings were chosen for this exploratory experiment, not calibrated on a separate development set.

Compare the growing model with a fixed two-table online hidden-state mixture using the **same implementation** (both tables randomly asymmetric from the start, no growth), and a fixed one-table learner. This fixed-capacity mixture is a conventional hidden-state style baseline but not an independently implemented HMM. Use the same training stream of 1,200 transitions per condition, then freeze all weights and structure, reset beliefs, and evaluate on a separately seeded 400-transition stream. The evaluation filter still updates its beliefs after each observation. The switching condition alternates two true rules every 100 transitions, with 10% noise. Two stationary controls use only rule A with 10% and 25% noise. The training and evaluation streams differ but repeat the same underlying four-symbol rules. True rule labels are logged and used only for analysis, including the best alignment of model IDs to rule IDs; that alignment uses evaluation labels post hoc and is not an online prediction metric. Rule match queries all four sources for both rules, choosing the best permutation after training; it evaluates the dominant transitions, not generalization beyond these known rules.

```sh
nix-shell --run 'cargo test && cargo build --release --bin discovery'
nix-shell --run 'for s in 81 82 83 84 85 86 87 88 89 90; do target/release/discovery "$s" "results/discovery-$s"; done'
```

Ten-seed mean frozen-evaluation accuracy:

| Stream | Growing model | Fixed two memories | Single memory | Growth events |
|---|---:|---:|---:|---:|
| Switching, 10% noise | 0.853 | 0.856 | 0.465 | 10/10 runs |
| Stationary, 10% noise | 0.905 | 0.905 | 0.905 | 10/10 runs |
| Stationary, 25% noise | 0.762 | 0.762 | 0.762 | 10/10 runs |

For switching streams, after label permutation, both two-memory methods matched all eight dominant rule transitions in all ten seeds; mean context alignment **after observing each transition** was 0.945 for the growing model and 0.950 for the fixed model. The latter is not context accuracy before prediction. The first growth event in switching runs occurred between steps 125 and 500. On stationary streams all growth events are false alarms because the generating rule never changes. At 10% noise the trigger falsely grew at steps 250–500; at 25% it did so at steps 125–225. Growth was capped at one event per run, so these are proportions of runs with at least one false alarm, not an unconstrained false-alarm rate. The fixed two-memory model used more capacity throughout, and its accuracy is very similar on this small sample; no statistical superiority is claimed.

An exploratory log inspection for seed 81 found that a stationary 10% noise run triggered at step 400: the two surprise-window means were 1.179 and 1.212, with 10 incorrect predictions across the 50 transitions. This supports the explanation that ordinary noisy observations can push the fixed threshold over its limit; it does not isolate all causes or establish a calibrated threshold. The result is a **failure of the growth criterion**, even though the two-memory learners discovered the two simple transitions when they were present. No claims about unseen grammar, longer memory, useful unsupervised language structure, or advantage over a fully independent HMM are warranted.

`metrics.csv` records settings and outcomes; condition-specific predictions include per-step training surprise, events record growth step and last window mean, and weights record final tables. True contexts in logs are for analysis only. Results are exploratory, not preregistered, and output directories are overwritten.

## Trial period before committing a new memory

This follow-up keeps the prior sustained-surprise condition as an alarm, but an alarm only opens a **candidate trial**. The existing one-memory model is cloned, and a randomly initialized second table is added to the candidate branch. Both branches learn from the same next 100 transitions. Their weights are then frozen for the following 100 transitions while both update their hidden-state beliefs and make predictions. Sum `control log loss − candidate log loss` over that later window. Commit the candidate only if this sum is strictly greater than a fixed penalty; otherwise discard it. After a completed trial there is a 100-transition cooldown. Incomplete trials at the end of training are discarded. Both branches stop updating weights during the comparison window, which also means the control forgoes online learning for those 100 transitions. The candidate's initial beliefs reset to uniform; the control retains its beliefs. A trial contains extra compute and temporary storage even if rejected.

The alarm still uses two non-overlapping 25-transition windows of mean surprise above 0.85, after 100 warmup steps. Each branch uses the same online mixture and rates as the previous discovery experiment. Only one additional memory is permitted. Compare with a fixed two-memory mixture and a fixed single-memory learner, both of which continue updating throughout training, including candidate-comparison windows. These are the same underlying model family, not independently implemented baselines. Labels and switch boundaries are used only to generate/score streams. Train for 1,200 transitions, then freeze all weights and evaluate on a separately seeded 400-transition stream while allowing beliefs to update. Conditions are alternating 100-step rule blocks with 10% noise, one stationary rule with 10% noise, and one stationary rule with 25% noise. This is a familiar four-symbol process, not unseen-grammar generalization.

Development runs used seeds 101–110 and penalties `0, 5, 10, 20, 40, 80` in total log-loss units across 100 validation transitions. After inspecting these exploratory results, 40 was selected: it was the smallest tried penalty with no stationary acceptances in either noise condition and still accepted a candidate in all ten switching runs. This selection rule and candidate list were not preregistered. The trigger, trial lengths, noise levels, and data distribution were not separately tuned or established as general. The held-out test then used seeds 201–220 and penalty 40.

```sh
nix-shell --run 'cargo test && cargo build --release --bin trial_growth'
nix-shell --run 'for p in 0 5 10 20 40 80; do for s in 101 102 103 104 105 106 107 108 109 110; do target/release/trial_growth "$s" "results/trial-dev-$p-$s" "$p"; done; done'
nix-shell --run 'for s in 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220; do target/release/trial_growth "$s" "results/trial-test-$s" 40; done'
```

Separate-seed results (20 runs per condition):

| Stream | Runs accepting a new memory | Frozen-test accuracy, trial model | Fixed two memories | Single memory |
|---|---:|---:|---:|---:|
| Switching, 10% noise | 20/20 | 0.855 | 0.857 | 0.467 |
| Stationary, 10% noise | 1/20 | 0.905 | 0.905 | 0.905 |
| Stationary, 25% noise | 1/20 | 0.751 | 0.751 | 0.751 |

The trial model opened 26, 36, and 64 candidates respectively; 26, 32, and 59 completed their evaluation before training ended. It matched all eight dominant transitions after post-hoc context permutation in all 20 switching tests, as did the fixed two-memory model. Frozen-test accuracy is measured on separate draws, but the same small set of patterns and rule schedules. Acceptance counts are per run under a maximum of one committed candidate; they do not estimate an unlimited-horizon false-alarm rate. The 1/20 false acceptance in each stationary condition is not evidence that the true error rate is 5% exactly, and the two conditions use the same test seeds.

The two stationary false acceptances occurred despite no underlying rule change. The 10% noise example had 9 noisy transitions in its 100-transition selection window and a control-minus-candidate loss of 92.58; the 25% example had 31 noisy transitions and a gain of 45.69. A candidate can improve predictive calibration or fit transient noise without representing a genuinely distinct rule. We did not isolate the cause in either case, so neither result proves a mechanism. The trial and penalty substantially reduced unnecessary commits relative to the earlier trigger-only experiment, but comparisons span different seed sets and the trial introduces delay and extra computation. Neither a formal significance guarantee nor a comparison against an independently implemented change-point detector is provided.

Each condition writes a candidate-trial log (start, end, gain, penalty, decision), per-step training trace, frozen-test predictions, and final weight tables; `metrics.csv` records the configuration. Incomplete trial gains are partial and are never accepted. Snapshots are not restartable checkpoints; existing directories are overwritten.

## Stress test: switching rates and long stationary streams

This repeats the candidate-trial mechanism with the previously selected penalty fixed at 40, the same 100-step candidate-training and 100-step frozen-comparison windows, and the two-memory cap. It varies switching-block length (50, 100, or 200 transitions) and stationary-stream length (1,200 or 5,000 transitions); noise is 10% except the additional 25% stationary condition. Each run then evaluates frozen weights on a separately seeded 400-transition stream with the same condition. The labels are used to generate and score the data, never supplied to the trial model. The model can commit at most once, so reported false alarms are fractions of runs with any commit, not rates per transition.

An independently implemented one-sided **CUSUM-style alarm** serves as a change-detection reference, not as a rule-learning or growth model. It maintains cumulative transition counts with a one-count prior, uses the current count-model probability of the observation as its null and a uniform four-symbol distribution as an alternative, then accumulates `max(0, previous + ln(0.25 / null_probability))`. It begins accumulating after 100 observations and records the first threshold crossing; it neither creates memories nor predicts after an alarm. Its estimated null keeps updating and is not a textbook fixed-null likelihood-ratio test. During fast switching it may already have seen both rules before monitoring begins, so its null can become a mixture. Its binary alarm is not equivalent to the trial model's trained, retained second memory, and its runtime, allocation and complexity are not matched.

Development seeds 401–410 ran alarm thresholds 5, 10, 20, 40, and 80. After inspecting results, threshold 10 was chosen as the smallest tested threshold with at most one alarm in each 10-run, 5,000-step stationary condition (one at 10% noise, zero at 25%). This criterion was set after inspecting the exploratory development results, not preregistered. Separate seeds 301–320 were then used for evaluation. The growth model's penalty of 40 was selected on the previous experiment's development seeds, not retuned here.

```sh
nix-shell --run 'cargo test && cargo build --release --bin stress'
nix-shell --run 'for s in 401 402 403 404 405 406 407 408 409 410; do target/release/stress "$s" "results/stress-dev-$s"; done'
nix-shell --run 'for s in 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320; do target/release/stress "$s" "results/stress-test-$s"; done'
```

Separate-seed results, 20 runs per condition. A commit or alarm is counted if it occurs at any time within the training stream. All switching commits occurred after the first true switch in these runs.

| Stream | Runs committing memory | CUSUM alarms | Trial-model frozen accuracy | Fixed-two-memory accuracy |
|---|---:|---:|---:|---:|
| Switch every 50 steps, 1,200 steps | 20/20 | 0/20 | 0.833 | 0.835 |
| Switch every 100 steps, 1,200 steps | 19/20 | 20/20 | 0.823 | 0.843 |
| Switch every 200 steps, 1,200 steps | 14/20 | 20/20 | 0.718 | 0.848 |
| Stationary, 10% noise, 1,200 steps | 1/20 | 1/20 | 0.897 | 0.897 |
| Stationary, 10% noise, 5,000 steps | 9/20 | 2/20 | 0.897 | 0.897 |
| Stationary, 25% noise, 1,200 steps | 2/20 | 1/20 | 0.740 | 0.749 |
| Stationary, 25% noise, 5,000 steps | 6/20 | 3/20 | 0.741 | 0.749 |

All switching conditions use 10% noise. The 200-step switching condition exposes sensitivity to the trial schedule: a 100-step training window and subsequent 100-step evaluation window can see very different mixtures of rules. That is a plausible explanation for missed commits, not a causal finding established by this experiment. Long stationary runs provide more opportunities for repeated candidate trials, and the fraction with a false commit increased in these samples. This does not estimate an asymptotic false-alarm probability. The CUSUM reference missed all fast-switching runs at its chosen threshold, while the trial method had more false commits on the long stationary runs. Different objectives and mechanics prevent treating this as a controlled head-to-head comparison of complete sequence engines.

`metrics.csv` records the candidate decision, held-out accuracy, and first alarm for all tested CUSUM thresholds, not only the selected threshold. Runs are exploratory and reuse the same seed across conditions, so condition results are paired and not independent. No confidence intervals, compute costs, or comparisons on unknown numbers of rules are claimed.

## Increasing the penalty after repeated attempts

This exploratory test keeps the existing alarm, candidate-training period, frozen comparison, and initial penalty of 40. On the kth candidate attempt, the acceptance threshold becomes `40 + scale × ln(k)`. The first attempt is unchanged; every completed rejection makes the next threshold higher. Incomplete candidates are discarded as before. This is a heuristic inspired by repeated-testing concerns: predictive log-loss differences from adaptive, dependent streams are not shown to be valid e-values or likelihood ratios under a fixed null, so this schedule carries **no formal false-alarm guarantee**.

We test scales 0, 20, 40, and 80 on all conditions, using the same seeded streams for paired comparisons. Switching conditions have 1,200 training transitions with 100- or 200-step rule blocks and 10% noise. Stationary conditions have 5,000 transitions with 10% or 25% noise. Every run freezes its model and evaluates on a separately seeded 400-transition stream of the same type. Run-level commits are capped at one. First, seeds 501–510 were used for exploration; then **all four fixed schedules**, without selecting a winner, were run on separate seeds 601–620. The grid, conditions, and procedure were not preregistered. The base penalty and other hyperparameters were selected for earlier tasks, not recalibrated here.

```sh
nix-shell --run 'cargo test && cargo build --release --bin sequential'
nix-shell --run 'for s in 501 502 503 504 505 506 507 508 509 510; do target/release/sequential "$s" "results/sequential-dev-$s"; done'
nix-shell --run 'for s in 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620; do target/release/sequential "$s" "results/sequential-test-$s"; done'
```

Separate test seeds, number of runs out of 20 that committed a memory:

| Scale | Switch every 100 | Switch every 200 | Stationary, 10% noise | Stationary, 25% noise |
|---:|---:|---:|---:|---:|
| 0 | 19 | 20 | 11 | 3 |
| 20 | 19 | 20 | 1 | 0 |
| 40 | 19 | 20 | 1 | 0 |
| 80 | 19 | 20 | 1 | 0 |

The scale-0 row is the prior fixed penalty, on these new seeds. On the switching tests, the trial model's mean frozen-test next-symbol accuracy was 0.827 (100-step blocks) and 0.846 (200-step blocks) for all four scales; the fixed two-memory baseline scored 0.849 and 0.854. With unchanged streams at 10% and 25% noise, mean trial accuracy was 0.900 and 0.750 for every scale. Since the streams and runs are paired, matching averages do not imply independent replications or equal individual trajectories. In particular, the 200-step result differs from the earlier stress-test seed group: the previous 14/20 versus this 20/20 illustrates seed variability and the small sample. No estimate of long-run guarantees, detection delay bounds, or optimal scale follows from these runs. The selected scale is not a confidence level.

The per-seed `metrics.csv` records outcomes and `trials.csv` records each candidate's actual threshold, loss improvement, and decision. Each scale sees the same input stream but may create and evaluate different candidates after its decisions diverge. The output is an exploratory benchmark, not a validated sequential hypothesis test.

## Parallel model explorations

Three standalone experiments with independent commands and detailed protocols are documented under `alternatives/`. Their reported figures use different tasks, splits, and budgets; they are not a shared leaderboard.

- [Local associative graph](alternatives/local_graph/README.md): across ten exploratory seeds, exact reconstruction of previously stored patterns dropped from 1.000 with two patterns to 0.000 with sixteen, and the sixteen-pattern bit accuracy fell below the untrained cue baseline. This is not sequence generalization; the baseline is weak.
- [Bayesian run-length prediction](alternatives/latent_context/README.md): online next-symbol accuracy on 50/100/200-step switches was 0.796/0.856/0.879 versus 0.836/0.846/0.848 for a fixed two-memory mixture. The model uses substantially more computation and forgets/relearns rather than recalling recurring contexts; these are not frozen-test results.
- [Independent change alarms](alternatives/change_detection/README.md): on 40 held-out seeds, a calibrated Bayesian detector missed all scored switches and alarmed on 6/40 and 5/40 stationary 10%/25% noise runs. The independently implemented CUSUM-style reference caught 42/200 slower switches, none of the faster ones, and had 1/40 stationary alarms at each noise level. These are alarm events within a specified 20-step window, not memory-discovery accuracy.

Each prototype's README records its reproducible commands, conditions, caveats, and established prior-work connection.

## Local graph sequence completion versus lookup

A static associative graph was trained on 2–32 four-symbol sequences. Each symbol was encoded with a seeded, position-specific bipolar code; the first three encoded blocks were observed, and the final block began random. Observed prefix bits were held fixed while the graph updated only the missing block. The full training prefixes were unique. Test queries used the same stored sequences with newly sampled bit corruption, so this is retrieval of familiar sequences, not held-out-sequence generalization or online temporal processing. An exact nearest-neighbor lookup saw the same corrupted prefix and the same stored sequences; untrained-graph and majority-symbol controls were also measured. See [the standalone protocol](alternatives/local_graph/README.md) for ties, seeds, budgets, and reproducible commands.

Across ten seeds with 20% prefix-bit corruption, the graph's missing-symbol accuracy decreased from 1.000 with two sequences to 0.444 with 32; lookup remained at least 0.986. With clean prefixes, the graph still dropped to 0.475 at 32, while lookup was 1.000. This is a negative comparative result for the graph on this task, consistent with interference but not proof of its cause. The graph and lookup have different storage and computational budgets; no runtime measurements or evidence of generalization are claimed.

## Diagnosing local-graph sequence interference

The [local graph diagnostic](alternatives/local_graph/README.md) compared the all-pattern graph against a target-only oracle, an output-feedback ablation, one versus multiple sweeps, and exactly orthogonal codes within each position. It also measured target-aligned fields at the correct full state, which is an oracle diagnostic unavailable during prediction. Conditions used paired seeds 301–310 and familiar stored sequences with fresh cue initializations; codebook changes, loss of output feedback, and oracle target selection are distinct interventions, not matched production models.

At 32 sequences with clean random-coded prefixes, mean symbol accuracy was 0.463 for the full graph and 1.000 for the target-only oracle; 10.6% of output units had a positive target-only field made nonpositive by other patterns at the correct state. Removing output feedback gave 0.484, one sweep gave 0.451, and within-position orthogonal codes gave 0.476. Lookup was 1.000. Target-only performance and the oracle field flips support a role for cross-memory interference. Orthogonalizing within-position symbols did not resolve it, and output feedback removal helped only modestly in this condition. The target-only model knows which target memory to retain and cannot be used as a fair baseline. The bit-level and field results do not establish a single root cause or performance on unseen sequences. Full 20%-noise data and commands are in the standalone protocol.

## Research-runner conversion of sequence completion

The task-specific [research runner](toolkit/README.md) reproduces the prior local graph sequence-completion experiment using a versioned configuration and writes the exact shared noisy inputs, predictions, per-seed metrics, and provenance/status files. The example configuration uses seeds 201–210, 32 stored sequences, 20 corruptions per sequence, 20% prefix-bit noise, 20 sweeps, and 25% edge density. It compares the original trained graph, a trained graph with output-block feedback removed, nearest-prefix lookup, an untrained graph, and the most frequent target symbol. This is a conversion of an existing experiment, not a novel-sequence generalization result or an independent replication.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml'
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml -- toolkit/configs/sequence.cfg toolkit/runs/sequence-first'
```

The destination must not exist. In the first converted run, the trained graph, lookup, untrained graph, and majority baseline matched all 40 corresponding per-seed accuracy values from the previous evaluator. Pooled results were 0.4436, 0.9856, 0.2492, and 0.3469 respectively; the output-feedback ablation scored 0.4830. Those observations reuse the earlier data and must not be treated as new independent evidence. The runner is not a generic model interface and does not record a source/Nix revision, resource costs, or restartable model checkpoints. Runs under `toolkit/runs` are ignored by Git and the Nix channel is unpinned.

## Storkey-style learning versus Hebbian graph

The toolkit runner now trains two graphs from identical initial sparse symmetric topology and the same full encoded sequences: the existing Hebbian update and a Storkey-style local-field update. For each edge `(i,j)`, the new update uses the pre-pattern weights to compute the fields at both endpoints with their direct mutual interaction excluded, then applies `(x_i x_j − x_i h_j(excluding i) − h_i(excluding j) x_j) / N`. All updates for a pattern use pre-update fields. Only existing edges are updated, so this is a sparse-topology adaptation of [Storkey's rule](https://homepages.inf.ed.ac.uk/amos/publications/Storkey1997IncreasingtheCapacityoftheHopfieldNetworkwithoutSacrificingFunctionality.pdf), not a reproduction of its original fully connected capacity results. The two algorithms use the same pattern order. A hand-calculated three-unit test checks the second update and first-pattern equivalence; tests also check symmetry and missing-edge preservation.

Each graph receives exactly the same noisy prefix and random initialization of the missing block as the nearest-prefix lookup. The task, positional codes, four-symbol alphabet, 128 units, 25% edge density, 20% prefix-bit corruption, and maximum 20 asynchronous sweeps are unchanged from the prior stored-sequence completion experiment. The evaluation corrupts prefixes of previously stored sequences; it does **not** test prediction on unseen sequences or temporal processing. Development seeds 11001–11010 and separate test seeds 21001–21020 cover loads 4, 8, 16 and 32. No rule hyperparameters were tuned on the development outcomes; the update scale is the unmodified `1/N`. We inspected development results before running the fixed test comparison, so this is exploratory, not preregistered or a blind confirmation.

```sh
nix-shell --run 'cargo test --manifest-path alternatives/local_graph/Cargo.toml && cargo test --manifest-path toolkit/Cargo.toml && cargo build --release --manifest-path toolkit/Cargo.toml'
nix-shell --run 'for group in dev test; do for k in 4 8 16 32; do toolkit/target/release/murmur-research-runner "toolkit/configs/storkey/$group-$k.cfg" "toolkit/runs/storkey-$group-$k-example"; done; done'
```

Mean missing-symbol accuracy over 20 test seeds, 20 corruptions per stored sequence per seed:

| Stored sequences | Hebbian | Storkey-style | Lookup | Seeds Storkey better / tied / worse |
|---:|---:|---:|---:|---:|
| 4 | 0.909 | 0.951 | 0.999 | 18 / 2 / 0 |
| 8 | 0.698 | 0.770 | 0.996 | 19 / 0 / 1 |
| 16 | 0.558 | 0.602 | 0.991 | 17 / 1 / 2 |
| 32 | 0.449 | 0.436 | 0.983 | 7 / 0 / 13 |

Seed-level outcomes are paired; the many corruptions of each stored sequence are not independent datasets. No confidence intervals or significance tests are claimed. In the target-state oracle diagnostic, the mean fraction of output units whose target-only positive field became nonpositive in the full graph was 0.100 for Hebbian versus 0.022 for Storkey at 32 sequences. The correct missing block is supplied to this diagnostic, not to the models during prediction. Unlike Hebbian updates, Storkey updates depend on prior patterns, so that difference should not be interpreted as a literal additive decomposition of individual patterns. Fewer such flips did not translate to higher decoded accuracy at the largest load; it is one diagnostic, not an explanation of all retrieval errors.

### Resource measurement scope

`resources.csv` contains per-seed wall-clock nanoseconds using `Instant`, measured in a single process in a fixed model order. Graph training time includes encoding and learning the stored sequences, but excludes topology generation; graph prediction time includes clamped graph completion and symbol decoding, excluding cue generation and the separately computed oracle diagnostic. Lookup prediction includes its current implementation's repeated encoding/scan of stored sequences; lookup training time is blank because no separate index is constructed or timed. Timings include measurement and allocation overhead, are not isolated CPU benchmarks, and depend on host/load/compiler. They are useful exploratory observations, not a stable speed ranking or a matched-compute comparison.

At 32 sequences on this local release run, mean graph training time was about 0.624 ms for Hebbian and 1.390 ms for Storkey; prediction was about 13.1 and 13.7 microseconds per query, versus 2.6 microseconds for the lookup implementation. The graph model-owned heap-payload estimate was 147,456 bytes for each graph: capacities of its dense `f64` weight array and `bool` edge mask. The lookup's estimated stored-sequence payload was 1,024 bytes at this load. These figures exclude shared codebooks, graph/Vec headers, allocator overhead, stack and temporary allocations (including Storkey's field workspace), executable size, datasets/cues, process memory, and timing instrumentation. They describe these implementations, not theoretical minimum storage. No peak RAM, GPU, or matched-budget claims are made.

An initial 0.2.0 runner mistakenly treated Rust's `Vec<bool>` mask as bit-packed and undercounted graph payload. Version 0.2.1 fixes this and adds a regression check; the original run directories were retained with `RESOURCE_WARNING.txt`, and the reported measurements come from separate `-corrected` directories. Their accuracy CSVs match the original runs. The runner stores per-trial predictions, per-seed results, oracle-field diagnostics and resource records; run artifacts are ignored by Git, source/Nix revisions remain unrecorded, and results vary with host timing.

## Spreading memories across small graphs

Compare one 128-unit graph storing 32 four-symbol sequences against four independently initialized 128-unit graphs storing eight sequences each. Run both Hebbian and sparse-topology Storkey-style learning. All models receive the same stored sequences, position-specific codebook, noisy prefix, and random missing-block initialization. Single-graph topology uses seed `s`; shard topologies use `s+4000+shard` with wrapping arithmetic and are identical between learning rules. Assignment is fixed by sequence insertion index modulo four. It is an external experimental partition, not learned, balanced by design, and not a growth or self-organization rule.

The **oracle** receives the true shard for the evaluated stored sequence. The actual **router** has only the first three symbols of each stored sequence and its assigned shard; it searches the noisy input prefix for the nearest stored prefix by Hamming distance, choosing the first entry on ties. It receives no target symbol or true shard from the evaluated example. The lookup baseline uses precisely that same nearest-prefix search and directly returns the matched stored final symbol. Thus the router already does most of lookup's work; its high accuracy is not evidence of an autonomous locally learned router. Router, graphs, and lookup receive no evaluation updates. All test cues are fresh corruptions of known stored sequences; this is associative retrieval, not unseen-sequence generalization.

Development seeds 30001–30010 were inspected first; the same settings were then run without tuning on test seeds 31001–31020. Each seed uses 20 queries per stored sequence, 20% independent prefix-bit corruption, 25% graph edge density, and up to 20 asynchronous sweeps. This is exploratory and not preregistered. Seeds, prefix draws, and initial states are paired across conditions; queries sharing a stored sequence or graph are correlated.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml && cargo build --release --manifest-path toolkit/Cargo.toml --bin sharded'
nix-shell --run 'toolkit/target/release/sharded toolkit/configs/sharded/dev.cfg toolkit/runs/sharded-dev-example'
nix-shell --run 'toolkit/target/release/sharded toolkit/configs/sharded/test.cfg toolkit/runs/sharded-test-example'
```

Separate test-seed results, mean over 20 seeds, 640 queries per seed:

| Model | Missing-symbol accuracy |
|---|---:|
| One Hebbian graph | 0.454 |
| Four Hebbian graphs, oracle shard | 0.695 |
| Four Hebbian graphs, prefix router | 0.690 |
| One Storkey-style graph | 0.431 |
| Four Storkey-style graphs, oracle shard | 0.772 |
| Four Storkey-style graphs, prefix router | 0.765 |
| Direct nearest-prefix lookup | 0.986 |

The prefix router selected the originally assigned shard on 0.985 of queries. Routed sharding outperformed the corresponding single graph in all 20 seeds for both learning rules; the routed Storkey variant also exceeded the routed Hebbian variant in all 20. These paired counts are descriptive, not confidence intervals or proof beyond the sampled setup. Oracle and routed predictions match when the router selects the correct shard; wrong-shard predictions can nevertheless coincide with the target symbol. Recurrent feedback, assignment choices, shard size, training order, other capacities and noise levels were not isolated. The four-graph configuration has four times as many units, so it does not establish that partitioning alone beats a single 512-unit graph at matched capacity.

`resources.csv` records graph-learning wall time and total per-query prediction time in fixed execution order using `Instant`. Four-graph routed prediction includes one prefix search plus one selected graph update, but excludes I/O and cue creation. Lookup times the same search plus retrieval of its stored target. At the tested 32-sequence load, mean local release-run measurements were roughly 12.8 µs/query for the single Hebbian graph, 13.5 µs for routed Hebbian, 13.7 µs for single Storkey, 14.0 µs for routed Storkey, and 1.8 µs for lookup. Average learning times were approximately 0.60/0.81 ms for single/four Hebbian graphs and 1.27/1.54 ms for single/four Storkey graphs. These are unisolated timings, include timer and temporary-allocation overhead, exclude graph construction, router/index construction, data preparation, oracle path and I/O, and were measured in a fixed order; they are not stable hardware-independent performance claims.

The model-owned heap-payload estimates count `Vec` capacities for dense `f64` weights and `bool` masks, plus the router's stored `(prefix, shard)` entries for routed variants. They exclude shared codebooks, allocator/struct overhead, transient vectors, process memory, datasets/cues, and lookup index construction. On this 64-bit platform, the one-graph payload was 147,456 bytes; four graphs plus the router were 590,848 bytes; the lookup's stored-sequence payload was 1,024 bytes. Those are implementation-specific estimates, not peak RAM or theoretical storage minima. Topology/weights can be reconstructed from recorded seeds but are not stored as restartable checkpoints.

The `sharded` binary writes configuration, protocol, stored sequences, exact shared cues and predictions, per-seed metrics, resources and execution status. It refuses existing run destinations. Outputs in `toolkit/runs` are ignored by Git; source and Nix revisions are not captured or pinned. The result supports distributing this particular retrieval workload across small graphs given an effective external router and a larger memory budget; it does not demonstrate learned local routing or a practical advantage over direct lookup.

## Four small graphs versus one graph with matched connection storage

The prior sharding experiment compared four 128-unit graphs with one 128-unit graph, so the sharded model had four times the dense edge-array payload. This follow-up compares the same four-graph setup with a **single 256-unit graph**: four 128×128 dense arrays and one 256×256 array have the same number of entries. Both use 25% randomly sampled undirected connections, so their actual active edge counts are close but not identical. Both Hebbian and sparse Storkey-style updates are tested. Four shards still have 512 total units, the single graph has 256, and their topology, code representation and dynamics differ. This matches one implementation-specific connection-storage budget; it does not match unit count, exact active edges, execution time, or all memory costs.

Both models see the same 32 stored four-symbol sequences and the same 32-bit-per-position cues. For the initial large-graph control, each 32-bit code block is **duplicated** to 64 bits: training patterns, noisy visible blocks, and random missing-block initializations each duplicate the same values. Thus it receives no additional independent evidence, but the duplicated features make this particular representation more correlated. The large graph uses seed `s+9000`; its Hebbian and Storkey variants start from identical topology. The four 128-unit graphs receive the original 32-bit blocks, with fixed insertion-index-modulo-four partitioning and graph seeds `s+4000+shard`. The router still searches stored prefixes and the oracle still receives the correct shard. The lookup baseline uses the same prefix search and directly returns the stored target. As before, all evaluated queries are fresh corruptions of sequences seen during training, not unseen-sequence generalization.

We ran 10 development seeds (41001–41010) and 20 distinct test seeds (42001–42020), at 0% and 20% prefix-bit corruption, with 20 query initializations per sequence, 25% density, and at most 20 sweeps. The duplicate-code comparison was inspected before the independently coded sensitivity control was added, so this study is exploratory and not preregistered. The large and sharded variants share sequence and base-cue draws, but their sampled topologies differ. Seed-level comparisons are paired; repeated queries of a stored sequence are not independent datasets.

### Sensitivity to the larger graph's representation

After inspecting the duplicate-code comparison, we added a second 256-unit control with the **same topology and weight-storage budget**, but an independently seeded second 32-bit code for each symbol/position instead of a duplicate. On noisy trials, the added prefix bits are independently corrupted; the large graph therefore receives an additional noisy view of the same underlying symbols that the router and small graphs do not. The extra missing-block initialization is independently random. This makes the control useful for checking sensitivity to the duplicate representation but is not an equal-information comparison. The original prefix and its corruption remain shared, and all models train on the same symbolic sequences. Codebook, noise, and topology seeds and exact cues are recorded in the output. This sensitivity control was designed after the first outcomes, so its test numbers are post-hoc exploratory, not an independently preregistered confirmation.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml && cargo build --release --manifest-path toolkit/Cargo.toml --bins'
nix-shell --run 'for group in dev test; do for noise in clean noisy; do toolkit/target/release/matched_budget "toolkit/configs/matched/$group-$noise.cfg" "toolkit/runs/matched-$group-$noise-example"; toolkit/target/release/independent_encoding "toolkit/configs/matched/$group-$noise.cfg" "toolkit/runs/independent-$group-$noise-example"; done; done'
```

Separate test-seed results, mean over 20 seeds and 640 queries per seed:

| Model | Clean prefix | 20% corruption |
|---|---:|---:|
| Single 256-unit Hebbian, duplicated codes | 0.463 | 0.448 |
| Single 256-unit Hebbian, independent extra codes | 0.466 | 0.455 |
| Four 128-unit Hebbian graphs, prefix router | 0.739 | 0.683 |
| Single 256-unit Storkey-style, duplicated codes | 0.528 | 0.456 |
| Single 256-unit Storkey-style, independent extra codes | 0.548 | 0.490 |
| Four 128-unit Storkey-style graphs, prefix router | 0.850 | 0.756 |
| Direct prefix lookup | 1.000 | 0.985 |

With the oracle shard, four-graph Hebbian/Storkey accuracy was 0.739/0.850 clean and 0.690/0.764 at 20% corruption. The router chose the originally assigned shard 1.000/0.985 of the time. On each of the 20 paired test seeds, routed sharding exceeded the single 256-unit graph for both learning rules and both encodings. This descriptive observation does not establish a general superiority result: the partition and router use stored prefixes, the additional-code control has extra noisy evidence, topology and unit counts differ, and only one family of datasets and settings was tested.

In the 20%-noise test, the mean active undirected edge count was approximately 8,164 for the single large graph versus 8,068 across four shards. Graph-owned dense edge-array payload was exactly 589,824 bytes in both; the sharded router added an estimated 1,024 bytes for stored prefix/assignment entries. These estimates exclude codebooks (including the independent extra codebook), model/Vec headers, allocator overhead, scratch/state vectors, dataset/cue storage and process memory. Lookup's stored-sequence payload was 1,024 bytes, excluding its codebook and overhead. These numbers are implementation-specific, not theoretical limits or peak memory.

For context only, the 20%-noise duplicate-code local run measured roughly 54.5 µs/query for single-large Hebbian and 13.7 µs for routed-four Hebbian; 60.0 and 14.2 µs for Storkey; lookup was 1.8 µs. The larger graph's timed call includes constructing duplicated input blocks; the sharded routed call includes prefix search plus a selected-graph call. Data generation, graph and index construction, I/O, and other graph queries are excluded. Timers run in fixed order in an unisolated process and include timer/allocation effects; no stable or compute-matched speed claim follows. The independent-code sensitivity runs measured similar scopes but are separate executions. `resources.csv` records individual seed totals and sample counts.

The matched and sensitivity runners write versioned config, protocol, exact base/extra cues and predictions, per-seed metrics, resource estimates, and status in new directories. The duplicate and independent runs used runner versions 0.4.0 and 0.4.1 respectively; source/Nix revisions remain unrecorded and the Nix channel is unpinned. Run artifacts are ignored by Git. Neither model learns the router, grows shards, or operates token by token.

## A supervised learned router

This exploratory comparison asks whether a small decision model can route noisy prefixes to the existing graph shards without scanning a stored-prefix index. The four 128-unit graphs and 32 sequences use the same balanced insertion-order assignment as the previous sharding experiment. A separate linear multiclass softmax router is trained **from scratch** with those externally assigned shard labels. It has 96 binary prefix inputs, four outputs, and 388 `f64` coefficients including biases. It trains for 100 epochs with shuffled stored prefixes and newly sampled 20%-bit-flip augmentation, using learning rate 0.15 with updates normalized by 96 input features. It receives no missing target symbol when routing. The graph memories and router are frozen during evaluation. This is supervised routing to a supplied partition, not unsupervised assignment, structural growth, or routing learned by local graph units. It is a small conventional classifier used as an auxiliary router, not the graph's learning mechanism.

Compare the learned router to the oracle true shard, the previous nearest-stored-prefix router, and direct nearest-prefix lookup. All graph variants see the same stored sequences, topology, noisy evaluation prefixes and missing-block initializations for each seed. Router training uses its own deterministic augmentation stream (`s+12000`); evaluation corruptions use the task's separate stream (`s+3000`). Training and evaluation use the same underlying stored prefixes with new noise, so neither unseen-sequence nor compositional generalization is tested. The 32 sequences are assigned round-robin by insertion order, without regard to similarity; the labels are an arbitrary supervised target. Development seeds 51001–51010 were inspected, and the unchanged settings were evaluated on test seeds 52001–52020. No hyperparameter search was performed; the experiment was exploratory, not preregistered.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml && cargo build --release --manifest-path toolkit/Cargo.toml --bin learned_router'
nix-shell --run 'for group in dev test; do toolkit/target/release/learned_router "toolkit/configs/learned-router/$group.cfg" "toolkit/runs/learned-router-$group-example"; done'
```

Mean accuracy across 20 test seeds (640 noisy queries per seed):

| Metric | Accuracy |
|---|---:|
| Correct shard, supervised linear router | 0.554 |
| Correct shard, nearest-prefix router | 0.986 |
| Single Storkey-style graph, final symbol | 0.432 |
| Four graphs, supervised router, final symbol | 0.511 |
| Four graphs, nearest-prefix router, final symbol | 0.764 |
| Four graphs, oracle shard, final symbol | 0.771 |
| Direct nearest-prefix lookup, final symbol | 0.985 |

The router correctly classified 0.625 of clean training prefixes on average after noisy augmentation. As a post-hoc investigation on **development seeds only**, `router_diagnostic` trained the same linear model on clean prefixes for longer: average clean training accuracy was 0.622 after 100 epochs, 0.625 after 1,000, and 0.628 after 5,000; routing accuracy on noisy cues was 0.528, 0.504, and 0.494 respectively. These exploratory runs suggest that simply training longer did not fix this setup and that arbitrary shard assignments may not fit this additive linear decision rule well. They do not prove mathematical non-separability, optimal convergence, or that other routers or shard assignments would fail. These diagnostics changed augmentation and are not a held-out test.

```sh
nix-shell --run 'cargo run --release --manifest-path toolkit/Cargo.toml --bin router_diagnostic -- toolkit/configs/learned-router/dev.cfg toolkit/runs/router-diagnostic-example'
```

The learned router's parameter array uses 3,104 bytes (`4 × 97 × 8`) versus 1,024 bytes for the nearest-prefix index at this load on the current 64-bit platform; four graphs' dense weight/mask payload is 589,824 bytes for both. These figures exclude common codebooks, model/Vec headers, training data, allocator/stack overhead beyond the router's coefficient array, temporary data and process memory. On the local unisolated release run, learned-router training averaged 3.58 ms per seed; route-only prediction averaged 0.31 µs versus 1.81 µs for nearest-prefix search. These timings use fixed execution order and include timer/temporary overhead; training covers augmentation and router learning only, not graph or index construction. They are exploratory implementation-specific timings, not matched-budget performance guarantees. The learned router is quicker in this run but made substantially more routing mistakes.

The runner saves configuration, externally assigned training labels, per-query predictions and cues, router coefficients, per-seed metrics, timing/payload estimates and status. Outputs under `toolkit/runs` are ignored by Git, source/Nix revisions are unrecorded, and the channel remains unpinned. Output destinations must be new. No automatic router assignment, context growth, or recurrent sequence-learning claim follows.

## Directed local prediction-error graph

This experiment explores an established error-correcting idea in the fixed sparse graph; it is not an architectural novelty claim or a complete predictive-coding model. The first 96 units hold a clean encoded three-symbol prefix during training; the final 32 form a recurrent output block with random initial states. After up to three asynchronous training sweeps, each output unit calculates `error = target_bit − tanh(local_field)`. Each of its existing incoming connections is updated by `0.5 × error × source_activity / incoming_degree`, with weights clipped to [-1,1]. The target output bits are explicitly supervised during training. Updates are directed: they do not preserve symmetry, and no Hopfield energy or convergence guarantee applies. Training shuffles stored sequences for 30 epochs, with deterministic random output initializations. Evaluation freezes the weights, supplies a newly corrupted prefix and random output initialization, and permits at most 20 sweeps. A trial is marked converged only if a full sweep has no changes; trials that hit the limit are not necessarily permanently unstable.

A feedback ablation removes connections among the output units, keeping the same input-to-output connectivity, algorithm, order, and initial-state random stream. In that ablation outputs depend directly on clamped inputs, so it is effectively a conventional supervised linear-style decoder with hard threshold dynamics; its role is a control, not evidence that a recurrent graph is needed. Compare both variants with the same-task one-pass Hebbian and sparse Storkey-style graphs and nearest-prefix lookup. All see the same stored sequences and evaluation cues. The predictive graphs receive 30 presentations per sequence while the original graph rules receive one, so the first comparison is not training-budget matched. Per-output target bits and global training supervision are available during learning; the earlier Hebbian rules also see complete encoded patterns, but the methods do not share identical learning signals or computation.

Development seeds 71001–71010 were inspected, then the same fixed settings were run on separate test seeds 72001–72020 at loads 8, 16, and 32, 20% prefix-bit corruption and 20 queries per sequence. No parameter tuning was performed in this comparison; it remains exploratory, not preregistered. Each evaluation cue corrupts a **previously stored sequence**; it does not test unseen compositions, temporal sequence processing, unlabelled learning, automatic routing, or graph growth. The same seed offsets reproduce graph topology, codebook, stored data, and evaluation cues; topology is identical for the full predictive graph and Storkey/Hebbian graphs, with output-feedback edges explicitly removed for the ablation.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml && cargo build --release --manifest-path toolkit/Cargo.toml --bin predictive_graph --bin training_budget'
nix-shell --run 'for group in dev test; do for k in 8 16 32; do toolkit/target/release/predictive_graph "toolkit/configs/predictive/$group-$k.cfg" "toolkit/runs/predictive-$group-$k-example"; toolkit/target/release/training_budget "toolkit/configs/predictive/$group-$k.cfg" "toolkit/runs/budget-$group-$k-example"; done; done'
```

Separate test-seed mean missing-symbol accuracy:

| Stored sequences | Local predictive graph | Predictive, no output feedback | Hebbian, one pass | Storkey, one pass | Lookup |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.879 | 0.900 | 0.692 | 0.774 | 0.994 |
| 16 | 0.722 | 0.742 | 0.558 | 0.630 | 0.991 |
| 32 | 0.526 | 0.539 | 0.440 | 0.443 | 0.985 |

The full predictive graph exceeded one-pass Storkey in all 20 paired seeds at each tested load, but the no-feedback ablation had higher mean accuracy at every load. With 32 stored sequences, only 0.566 of full-graph queries settled within 20 sweeps, versus 1.000 for the feedback-free version. This does not prove all remaining queries cycle indefinitely, but it is a material dynamics limitation. Queries sharing a seed and stored sequence are correlated; no confidence intervals or general superiority claims are made.

To check exposure count, `training_budget` presents each sequence to the Storkey baseline for the same 30 epochs in the same deterministic shuffled order. It consumes the same random initialization draws to preserve subsequent order; Storkey itself does not use output-state initializations. This repeated baseline uses its normal sparse Storkey rule without clipping or per-epoch retuning. It is stopped if any weight becomes nonfinite or exceeds an absolute value of 1,000,000; stopped runs have no scored prediction. All 20 test seeds completed in these runs. Mean repeated-Storkey accuracy at loads 8/16/32 was 0.276/0.095/0.194, with median maximum absolute weights approximately 1.55/37.69/43,264 after the last epoch. This is a negative result for **naively repeating this implementation**; Storkey is normally an incremental rule, and other rates, normalization or stopping policies were not tested. The predictive graph's clipping and target-dependent error make it a different algorithm, so equal presentations do not imply equal compute or equal optimization effort.

`predictive_graph` saves per-query predictions and convergence, output-bit metrics, last-epoch online local squared error, directed output-edge weights and measured training time. The diagnostic error is measured during the last changing-weight epoch, not a frozen loss or an independently validated objective. `training_budget` saves repeated-baseline metrics and predictions; its timer includes checking the numerical guard after every presentation and is not directly comparable with the predictive training timer. Resource usage, unseen-sequence generalization, online language modeling, continual retention and self-organization are not measured. Outputs are ignored by Git, source/Nix revisions remain unrecorded, and run destinations must be new.

## Post-hoc settling analysis

This is a descriptive follow-up to the directed local prediction-error experiment, not a new model run. The new `settling_analysis` utility groups the *existing* held-out prediction rows by whether the full recurrent graph reached a no-change sweep within its 20-sweep limit. It reports results for that graph and the feedback-free control on the **same queries**, both per seed and pooled. The grouping uses the full graph's outcome after inference, so it is a post-hoc selection, not information available before choosing a model. Queries share seeds and stored sequences; pooled counts do not imply independent samples.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml'
nix-shell --run 'cargo run --quiet --release --manifest-path toolkit/Cargo.toml --bin settling_analysis -- toolkit/runs/predictive-test-32/predictions.csv'
```

At 32 stored sequences, on the previously reported 20 test seeds:

| Full graph within 20 sweeps | Queries | Full graph accuracy | Feedback-free accuracy on same queries |
|---|---:|---:|---:|
| Settled | 7,243 | 0.570 | 0.579 |
| Did not settle | 5,557 | 0.468 | 0.487 |

The feedback-free control was also less accurate on the unsettled group. This is consistent with those inputs being harder for both models, so it does not establish that failing to settle caused the full graph's prediction errors. Likewise, a model reaching a fixed point does not guarantee a correct answer, and reaching the step limit does not prove the graph would never settle. We did not intervene on the dynamics, examine trajectories, or estimate uncertainty; these are post-hoc descriptive results. The input runs are ignored by Git and must be regenerated using the commands in the preceding experiment if absent. The parser accepts only the runner's unquoted CSV format, validates required fields, and emits per-seed and pooled counts to standard output.

## Frozen-weight interventions on graph dynamics

This experiment changes **only inference dynamics**, keeping each trained full predictive graph's directed weights, clamped input prefix, target, and initial output state fixed across interventions. The graph is retrained deterministically from the same configuration/seed as the preceding experiment; `forward20` is checked against that experiment's prediction and convergence on every query. The existing feedback-free predictor, one-pass Storkey graph, and nearest-prefix lookup are recorded on the same cues. Training remains supervised, for 30 epochs on the stored sequences. Evaluation is on fresh corruptions of those same sequences, not novel-sequence generalization.

The six update modes are:

- `forward20`: original fixed ascending unit order, zero threshold, at most 20 sweeps.
- `forward100`: the same deterministic order and rule, at most 100 sweeps.
- `reverse20`: fixed descending order, zero threshold, 20 sweeps.
- `shuffled20`: a seeded random permutation each sweep, zero threshold, 20 sweeps. One random-order trajectory is sampled per query; this is not an average over random orders.
- `margin005` and `margin020`: ascending order, 20 sweeps; a unit only changes if its local field is above +0.05/+0.20 or below −0.05/−0.20. A value inside the interval keeps its prior state.

A mode **converges by its own rule** when a whole sweep makes no changes. `strict_stable` separately checks whether every final output unit would also remain unchanged under the original zero-threshold rule; reaching a margin-based fixed point does not necessarily satisfy that stricter condition. A shuffled trajectory can end at a strict fixed point on its last allowed sweep without a subsequent no-change sweep to confirm convergence. For deterministic fixed-order modes, exact repetition of the output state at a sweep boundary is logged as a cycle; the clamped prefix, weights and order are fixed, so a repeated boundary state will repeat under those conditions. Cycle detection is intentionally omitted for shuffled order because the future update order changes. The simulator retains the final state at the sweep limit even after detecting a cycle, making the output depend on which cycle phase is reached.

Development seeds 71001–71010 and test seeds 72001–72020 are the same seed groups used in the earlier predictive-graph experiment. Both groups were already inspected in the earlier study, so they are separate from one another but **not fresh confirmatory test seeds** for this follow-up. The selected thresholds and modes are exploratory and were not preregistered. Loads 16 and 32 use 20% prefix corruption and 20 query initializations per stored sequence. Graph topology, learned weights, cues, and evaluation labels are paired across modes; only dynamics change. Shuffled-order RNG uses `(seed+25000 wrapping) XOR (sequence_index × 0x9e3779b97f4a7c15 wrapping) XOR trial`. The ground-truth target does not enter inference.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml && cargo build --release --manifest-path toolkit/Cargo.toml --bin dynamics'
nix-shell --run 'for group in dev test; do for k in 16 32; do toolkit/target/release/dynamics "toolkit/configs/predictive/$group-$k.cfg" "toolkit/runs/dynamics-$group-$k-example"; done; done'
```

At 32 stored sequences, pooled across the 20 previously inspected test seeds, 12,800 queries total:

| Mode | Missing-symbol accuracy | Converged by own rule | Strictly stable under original rule | Mean sweeps |
|---|---:|---:|---:|---:|
| forward20 | 0.526 | 0.566 | 0.566 | 10.74 |
| forward100 | 0.525 | 0.566 | 0.566 | 45.47 |
| reverse20 | 0.530 | 0.564 | 0.564 | 10.76 |
| shuffled20 | 0.524 | 0.590 | 0.590 | 10.60 |
| margin005 | 0.529 | 0.691 | 0.432 | 8.69 |
| margin020 | 0.531 | 0.938 | 0.174 | 4.46 |

For `forward20`, all 5,557 queries that failed to reach a no-change sweep had already repeated an exact sweep-boundary state; 4,597 were two-sweep cycles. Repeating the identical deterministic update for 100 sweeps produced no additional converged cases and almost unchanged accuracy; a different cycle phase can change a prediction. The no-feedback model scored 0.539 on the same queries. The stronger margin greatly increased stopping under its **modified** rule but had little effect on accuracy, and most final states were not stable under the original rule. This is not evidence that recurrence is useful or that thresholding fixes the memory. A no-feedback graph is a simpler control, and none of these differences establish a statistically significant improvement.

At load 16, `forward20` accuracy/convergence was 0.722/0.962; `forward100` remained 0.722/0.962; `margin020` was 0.711/0.999 by its own rule but only 0.454 strictly stable; no-feedback accuracy was 0.742. The results are paired but queries sharing stored sequences and seeds are correlated; no confidence intervals or compute/time benchmark are reported. `metrics.csv` retains per-seed counts and total sweeps; `predictions.csv` records each intervention's output, convergence, strict stability and fixed-order cycle length. No trajectories, averaged cycle predictions, cycle interventions during training, memory growth or unseen-language tasks were tested. The output is local, ignored by Git, and is not a model checkpoint.

## Symmetric feedback during local-error learning

The directed prediction-error graph updates incoming weights for each output unit independently; its output-output connections can become asymmetric, allowing fixed-order cycles. This comparison changes the *training constraint*, not just the stopping rule. After each pattern's usual directed local-error update and clipping, every existing output-output edge pair is replaced with the average of its two proposed weights. Both endpoints' updates contribute. Prefix-to-output weights remain directed; prefix states are clamped during inference. This preserves a symmetric, zero-diagonal output block without adding or removing connections. The two graphs begin with the same sparse topology and use the same stored patterns, shuffled training order, random initial output states, local targets, 30 training epochs and learning rate. Their trajectories and learned weights can differ after the first update.

For frozen symmetric output weights and a fixed clamped prefix, asynchronous one-unit-at-a-time sign updates with ties preserving state do not increase the corresponding Hopfield-style energy in exact arithmetic, and eventually settle in a finite state space. This does **not** guarantee correctness, convergence within an arbitrary 20-sweep limit, stability while weights are changing, or the same property for the directed graph. A unit test checks symmetry, sparsity and bounded weights after learning; another checks non-increasing restricted energy over sampled frozen-graph sweeps. Floating-point comparisons are tested with a tolerance, not proven for all possible numerical inputs.

The feedback-free control removes output-output edges and trains with the same local-error rule. The one-pass Storkey-style graph and nearest-prefix lookup are additional references, but have different learning algorithms and training presentations. All models use identical sequence sets, noise, and query initialization per seed. Evaluation corrupts sequences previously shown during training; it does not measure unseen-sequence generalization or temporal processing. No model receives the evaluation target for prediction.

Development seeds 81001–81010 were inspected, then unchanged settings were evaluated on separate seeds 82001–82020. Each seed has 8, 16, or 32 stored sequences, 20 newly corrupted queries per sequence, 20% prefix-bit noise, and a limit of 20 evaluation sweeps. The experiment is exploratory and not preregistered; repeated queries share stored sequences and graphs. No hyperparameter search or formal significance analysis was performed.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml && cargo build --release --manifest-path toolkit/Cargo.toml --bin symmetric_feedback'
nix-shell --run 'for group in dev test; do for k in 8 16 32; do toolkit/target/release/symmetric_feedback "toolkit/configs/symmetric/$group-$k.cfg" "toolkit/runs/symmetric-$group-$k-example"; done; done'
```

Mean missing-symbol accuracy over 20 test seeds:

| Stored sequences | Directed | Symmetric | No feedback | One-pass Storkey | Lookup |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.871 | 0.874 | 0.902 | 0.758 | 0.996 |
| 16 | 0.686 | 0.681 | 0.706 | 0.607 | 0.993 |
| 32 | 0.541 | 0.551 | 0.545 | 0.436 | 0.987 |

For 32 sequences, the directed graph reached a no-change sweep within 20 sweeps on 0.572 of queries; both the symmetric and no-feedback graphs did so on 1.000. The symmetric model took a mean of 3.12 sweeps, with a maximum of 7, on these queries. These are observed counts, not a guarantee that every possible input settles within 20 sweeps. Across paired seeds at load 32, symmetric accuracy exceeded directed in 14 of 20 and no-feedback in 12 of 20, with two ties against no-feedback. The observed differences are small and no statistical superiority is claimed; at lower loads the no-feedback mean was higher.

Thus symmetry addressed the observed non-settling in these runs, while leaving retrieval accuracy close to the controls and well below lookup. This does not establish that recurrent feedback is beneficial, nor that symmetry makes learning self-organizing. `predictions.csv` and `metrics.csv` record per-query outcomes and per-seed counts; `weights.csv` records learned symmetric output-row weights for inspection, not a complete restartable model. Source and Nix revisions are unrecorded; generated runs are ignored by Git.

## Continuous delayed-bit recall

This experiment asks whether fixed recurrent connections help represent an ongoing sequence, rather than whether a static graph can complete a stored pattern. A deterministic generator emits a stream of pseudorandom binary symbols. After consuming `x[t]`, each model predicts the **previously observed** bit `x[t−delay]`. The first `delay` steps are warm-up and are not scored. A separate model/readout is trained for each delay.

The connected model is the original fixed sparse leaky reservoir from `src/graph.rs`: 128 units, four incoming sampled edges per unit (512 directed edge entries, including possible self/duplicate edges), leak 0.35, fixed random input weights and graph weights, and one synchronous update per input symbol. Only the two-output online logistic readout learns (258 trainable weights including biases). The edge-free control is a clone with all graph edges removed; it retains the same input weights, leaky self-state, and an independently trained readout of the same size. The reset-state control uses the original graph but clears all state before each input, so it cannot carry earlier input through the graph. Each readout trains separately on the same stream and target timing. The recurrent graph's weights **do not learn**; this is reservoir computing, not the later symmetric local-error model, self-organization, or graph growth.

A causal shift register stores the last `delay` observed symbols and returns the oldest after seeing the current symbol. It knows the task delay, is not trained, and does not receive future input. The majority control predicts the most frequent delayed target in the training stream. Both are useful references but are not parameter-, memory- or compute-matched with the graph; the shift register has no learned weights but does consume memory for history. All models observe the same input stream.

Each graph/readout sees 6,000 training input symbols, makes an online supervised update for every step from `delay` onward, then has its state reset. Evaluation uses a separate 2,000-symbol stream with frozen readout weights and no within-stream reset (except the explicit reset-state ablation). The buffer also starts empty. The runner logs the generated streams and every scored test prediction. The task uses pseudorandom bits from deterministic SplitMix64 streams; distinct seeds are a reproducibility measure, not a mathematical proof of statistical independence.

Development model seeds are 91001–91010; final test model seeds are 192001–192020. Input streams use wrapping seed offsets `s+31000` for training and `s+32000` for evaluation. The supplied configurations are tested to ensure that none of these derived development stream seeds overlap the derived final test stream seeds. The runner also rejects a configuration whose own training and test stream seeds collide across runs. An earlier exploratory test group 92001–92020 had training stream seeds that overlapped development evaluation stream seeds and was **discarded**; its run directory contains `SEED_OVERLAP_WARNING.txt` and is not used for the results below. The groups and delay settings were not preregistered, and the test group was changed after detecting the seed issue.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml && cargo build --release --manifest-path toolkit/Cargo.toml --bin temporal_stream'
nix-shell --run 'for group in dev test; do toolkit/target/release/temporal_stream "toolkit/configs/temporal/$group.cfg" "toolkit/runs/temporal-$group-example"; done'
```

Mean accuracy over 20 final test seeds, with `2000−delay` scored inputs per seed:

| Delay | Connected graph | Edge-free graph | Reset state | Shift register | Majority |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.695 | 0.660 | 0.498 | 1.000 | 0.500 |
| 2 | 0.687 | 0.596 | 0.497 | 1.000 | 0.500 |
| 4 | 0.659 | 0.525 | 0.497 | 1.000 | 0.500 |
| 8 | 0.543 | 0.502 | 0.497 | 1.000 | 0.500 |

The connected graph exceeded the edge-free graph in all 20 paired seeds at every tested delay. At delay four the per-seed accuracy advantage ranged from approximately 0.083 to 0.169. This is evidence that the fixed recurrent connections contribute to the trained readout's short-term memory *on this task*, beyond the leaky-unit-only control. It does not isolate a particular path or show that graph connections learned. The history buffer is exact, much simpler for this task, and makes clear that the graph is not competitive as a bit-memory implementation. Adjacent queries share history and are correlated, and no confidence intervals, wall-clock or peak-memory measurements, continual-learning tests, novel-rule generalization, or language results are reported. Comparing graph and buffer as equal-resource models would be misleading.

`resolved.cfg`, `protocol.txt`, `streams.csv`, `predictions.csv`, `metrics.csv`, `summary.md`, and `status.txt` are written to a new output directory. Existing destinations are refused. The runner records graph edge counts and readout parameter counts, not the buffer's actual memory consumption; zero learned parameters does not mean zero memory. Runs under `toolkit/runs` are ignored by Git, and source/Nix revisions remain unrecorded and unpinned.

## Temporal connection plasticity

Can changing recurrent edges improve continuous delayed-bit recall? This experiment uses the preceding stream task and controls, with delays 4 and 8. Every graph has the same initial 128-unit, 512-edge topology and a separately trained 258-parameter readout. The adaptive graph also trains 512 edge weights; the fixed graph does not, so their trainable parameter budgets differ. No connections or units are added or removed.

After each synchronous graph step, each edge maintains a decaying eligibility trace: `trace = 0.8 * trace + leak * (1 - tanh_field²) * previous_source_state`. The graph receives a scalar error, `target - sigmoid(class_1_score)`, broadcast through fixed random destination signs scaled by `1/sqrt(128)`. Its edge update is `0.02 * error * destination_feedback * trace`, clipped to [-0.5, 0.5]. This is an approximate local-activity/error heuristic, not backpropagation through time or an established gradient. Its feedback is shared, not wholly local. The unrelated-feedback control uses the same inputs, initial graph, feedback signs and true-label readout learning, but supplies separately seeded random bits as the edge-learning targets; those bits can coincide with true targets. Consequently this is not an exact error permutation, and its weight changes need not match the adaptive model's magnitude.

All models train for 6,000 symbols and evaluate on a separate 2,000-symbol stream, after resetting state and freezing weights. The first `delay` symbols are not scored. Development seeds 291001–291010 and final seeds 391001–391020 use roles `s` for the graph, `s+31000` for train, `s+32000` for test, `s+33000` for feedback signs and `s+34000` for unrelated targets, with wrapping arithmetic. Tests check cross-group seed-role collisions; distinct deterministic pseudorandom seeds do not prove independence. The test settings were held fixed after observing the development results, but the whole protocol was not preregistered.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test temporal_plasticity && cargo build --release --manifest-path toolkit/Cargo.toml --bin temporal_plasticity'
nix-shell --run 'for group in dev test; do toolkit/target/release/temporal_plasticity "toolkit/configs/temporal-plasticity/$group.cfg" "toolkit/runs/plasticity-$group-example"; done'
```

Observed pooled accuracy on 20 final seeds (1,996 or 1,992 scored symbols per seed):

| Delay | Fixed graph | Task feedback | Unrelated feedback | Edge-free | Reset state | Shift register |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.6579 | 0.6596 | 0.6657 | 0.5198 | 0.5009 | 1.0000 |
| 8 | 0.5353 | 0.5398 | 0.5529 | 0.5015 | 0.4986 | 1.0000 |

Task feedback beat fixed in 12/20 seeds at delay 4 (one tie) and 17/20 at delay 8 (three ties), but beat unrelated feedback in only 4/20 (one tie) and 3/20 (one tie), respectively. At the end of training, mean absolute edge changes averaged 0.031 and 0.040 for task feedback, versus 0.193 and 0.100 for unrelated feedback, at delays 4 and 8. No task-feedback edges hit the clip limit; unrelated feedback averaged 11.45 and 0.05 clipped edges out of 512. The magnitude difference is a confound: this experiment does not isolate whether the unrelated signal or simply a larger perturbation helped. Adjacent test examples are correlated; these descriptive differences are not statistical significance claims. We do not measure runtime or peak memory here. The perfect shift register knows the delay and stores history; it is not resource-matched.

The result does not establish useful task-specific recurrent learning: indiscriminate changes did better under this setup. A useful follow-up is to match edge-change magnitudes, include frozen random perturbations, and examine whether the signal changes memory persistence or readout conditioning before designing a more complex learning rule. The logs contain resolved config, protocol, per-symbol predictions, metrics, training checkpoints, edge weights, feedback signs, and run status. They are not restartable snapshots; source and Nix revisions are not recorded and `toolkit/runs` is ignored by Git.

## Matched frozen perturbations

This follow-up asks whether the directions of the task-trained edge changes matter when their sizes are held constant. For each seed and delay, run the earlier online task-feedback and unrelated-feedback learners. Take the task learner's final graph and build a second graph from the same initialization: for each edge use the *absolute value of that edge's trained displacement* with a separately sampled random sign. If the chosen sign would exceed the existing weight limit, flip it; no flips were needed in these final runs. Thus each edge has exactly the same absolute displacement in the two frozen graphs, while its direction may differ. Magnitudes are derived from task training, so the control is not independent of the trained model. Signs are separately seeded, not guaranteed statistically independent.

Freeze both final graphs, reset their state, and train a fresh readout on the same 6,000-symbol training stream before evaluation on the separate 2,000-symbol stream. The fixed baseline also trains a fresh readout on that stream. This gives both frozen graph conditions the same readout procedure, but the task-trained graph has already seen the training stream once during edge learning: this is a two-stage training process, not a matched total-compute or data-exposure comparison with the fixed graph. The online models learn edges and readouts together in one pass. Evaluation always resets state and freezes graph and readout weights. The shift register is an exact task-specific, non-resource-matched reference.

Development seeds 591001–591010 and final test seeds 791001–791020 are distinct from the previous experiment's seed ranges. Seed roles are graph `s`, training `s+31000`, evaluation `s+32000`, feedback signs `s+33000`, unrelated targets `s+34000`, and perturbation signs `s+35000`, using wrapping arithmetic. Tests check cross-group role collisions. We held settings fixed between the development and test runs, but did not preregister the overall protocol.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test matched_perturbation && cargo build --release --manifest-path toolkit/Cargo.toml --bin matched_perturbation'
nix-shell --run 'for group in dev test; do toolkit/target/release/matched_perturbation "toolkit/configs/matched-perturbation/$group.cfg" "toolkit/runs/matched-$group-example"; done'
```

Pooled test accuracy across 20 paired seeds:

| Delay | Fixed graph | Online task | Online unrelated | Frozen task | Frozen matched random | Shift register |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.6523 | 0.6537 | 0.6596 | 0.6547 | 0.6538 | 1.0000 |
| 8 | 0.5346 | 0.5401 | 0.5699 | 0.5461 | 0.5401 | 1.0000 |

The frozen task graph beat its matched random-direction partner in 10 of 20 seeds at delay 4 (three ties), and 17 of 20 at delay 8. The mean paired advantages were 0.0009 and 0.0059 accuracy, respectively. Mean absolute per-edge changes were 0.0323 and 0.0405. This suggests the directions of task-trained changes may contribute a small benefit at delay 8 under this particular control, but no confidence intervals or causal mechanism are established; adjacent queries are correlated. The online unrelated-feedback learner still performed better on average, with a different, substantially larger weight displacement in the preceding experiment. This frozen intervention therefore does not resolve why online unrelated feedback helped or match its training trajectory and displacement. No growth, continual learning, language generalization, timing, or peak-memory claims follow.

Run directories contain resolved configuration, protocol, per-query predictions, per-seed accuracy, original/learned/random edge weights, displacement diagnostics and status. Runs are ignored by Git, are not restartable checkpoints, and source and Nix revisions are not recorded.

## Matching online update sizes

The prior frozen-graph intervention matched final per-edge displacement, but the original unrelated-feedback learner changed connections online and moved them farther overall. Here the task-feedback learner and unrelated-feedback control train concurrently on the same stream. At each scored step, first measure the L2 norm of the task learner's realized 512-edge update. Rescale the unrelated control's proposed update vector to that norm before applying it. Each graph has its own state, weights, eligibility traces and separately trained true-target readout; both use the same initial graph and fixed destination feedback signs. Clipping at ±0.5 could prevent exact matching. The control depends on the other model's update size, so this is a nonlocal experimental intervention, not an implementable independent local learning rule. Since the unrelated target is a bit and its prediction is strictly between 0 and 1, normalization effectively removes the magnitude of its scalar error, retaining its sign; this intervention changes more than a single fixed learning rate. Matching per-step global L2 norms does not match each edge, update direction, accumulated trajectory, or final weight displacement.

We also run the original fixed graph, task-feedback, and unnormalized unrelated-feedback learners and the causal shift register on identical streams. All models see 6,000 training symbols, are reset, and then evaluated with frozen weights on a distinct 2,000-symbol stream. The shift register knows the delay and is not resource matched. Development seeds 991001–991010 and held-out test seeds 1191001–1191020 use graph `s`, train `s+31000`, test `s+32000`, feedback `s+33000`, and unrelated targets `s+34000` (wrapping arithmetic). Distinct seed roles are checked within and across the provided groups. The protocol was not preregistered; configuration was not changed after observing development results.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test matched_online && cargo build --release --manifest-path toolkit/Cargo.toml --bin matched_online'
nix-shell --run 'for group in dev test; do toolkit/target/release/matched_online "toolkit/configs/matched-online/$group.cfg" "toolkit/runs/online-matched-$group-example"; done'
```

Pooled test accuracy across 20 paired seeds:

| Delay | Fixed | Task feedback | Unrelated feedback | Norm-matched unrelated | Shift register |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.6566 | 0.6579 | 0.6646 | 0.6565 | 1.0000 |
| 8 | 0.5490 | 0.5556 | 0.5733 | 0.5496 | 1.0000 |

Task feedback beat norm-matched unrelated feedback in 12/20 paired seeds at delay 4 (one tie) and 18/20 at delay 8. At delay 8, mean final absolute drift from initialization was 0.0406 for task feedback and 0.0277 for the matched control, despite equal per-step update norms. Across all test seeds, the maximum per-step norm mismatch was below `5e-17` and no edges were at the clipping limit at checkpoints. The unnormalized unrelated model again did better, but its updates and trajectory are different; this intervention supports investigating update magnitude, not a causal proof that magnitude alone explains its benefit. Examples within each stream are correlated, and these counts and accuracy differences are descriptive without confidence intervals. There are no measurements of compute, memory, changing tasks, generalization to language, or learned topology.

The output directory contains resolved config, protocol, predictions, metrics, cumulative update-norm and displacement diagnostics at 1,000-step checkpoints, and status; it does not save restartable weights. Runs are ignored by Git. Source and Nix revisions are not recorded.

## Fixed connection strength and memory persistence

Does increasing recurrent connection strength improve memory, and could that help explain the earlier unrelated-feedback result? This experiment starts from the same random 128-unit, 512-edge graph per paired seed, multiplies every initial edge weight by 0.5, 1, 1.5, 2 or 3 and clamps it to [-0.5, 0.5]. The wiring, input weights, leak and reader size remain the same. Each graph stays frozen while its own readout trains online; there is no graph learning in the scaled conditions. At 3×, clipping changes the distribution as well as the scale: 170.9 of 512 edges on average are clipped in the test group. The 1× condition is checked against the prior fixed-graph runner on every prediction. Separately, the prior online task-feedback and unnormalized unrelated-feedback models and the causal, task-specific shift register use the same stream.

The task again uses 6,000 training bits and 2,000 held-out test bits, resetting state at the boundary and freezing weights/readouts for testing. For an independent diagnostic, create two copies of each frozen scaled graph with zero state; feed one a 0 and the other a 1, then feed both the same pseudorandom continuation. Measure the root-mean-square distance between their states immediately and after 1–16 more inputs, averaging across 64 continuations. This is a finite-horizon perturbation response from a reset state, not a general stability guarantee or a direct measurement of the trained reader's information. Delays in the recall task count back from the current input, so impulse lag 8 is an illustrative comparable horizon, not the same test sequence.

Development seeds 1391001–1391010 and held-out seeds 1591001–1591020 use graph `s`, train `s+31000`, test `s+32000`, feedback `s+33000`, random targets `s+34000`, and impulse continuation `s+35000`, with wrapping arithmetic. The runner checks within-config seed roles and tests check cross-group roles. The scales and delays were fixed before observing the development results; the overall study was not preregistered.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test connection_strength && cargo build --release --manifest-path toolkit/Cargo.toml --bin connection_strength'
nix-shell --run 'for group in dev test; do toolkit/target/release/connection_strength "toolkit/configs/connection-strength/$group.cfg" "toolkit/runs/strength-$group-example"; done'
```

Pooled accuracy across 20 held-out seeds:

| Delay | 0.5× | 1× | 1.5× | 2× | 3× | Online task | Online unrelated | Shift register |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.6120 | 0.6489 | 0.6554 | 0.6594 | 0.6649 | 0.6522 | 0.6593 | 1.0000 |
| 8 | 0.5219 | 0.5417 | 0.5647 | 0.5978 | 0.6209 | 0.5482 | 0.5635 | 1.0000 |

At delay eight, 3× beat 1× and each online learner in all 20 paired seeds. The mean initial impulse RMS distance was 0.2397 for all scales. After eight common inputs, it was 0.0110 for 1×, 0.0209 for 2× and 0.0308 for 3×; after 16 it was 0.00071, 0.00303 and 0.00690, respectively. Longer-lasting state differences are consistent with improved temporal memory, though the experiment does not establish the mechanism of the accuracy increase or show that every preserved difference is useful.

As a post-hoc check of the *earlier* temporal-plasticity test's delay-eight `weights.csv`, mean absolute edge weight across its 20 seeds was 0.1241 at initialization, 0.1292 after task feedback and 0.1537 after unrelated feedback. Thus the unrelated learner happened to increase average edge magnitude in those runs. This comparison is descriptive, comes from a different seed group, and is not evidence that strength alone caused its advantage. A direct intervention on the trained graphs, or matched strength and trajectories, is needed. Neither the fixed scaling nor this diagnostic demonstrates graph self-organization, changing-task adaptation, language capability, statistical significance, or resource parity with a shift register. Adjacent query outcomes are correlated; no compute, peak memory, or confidence intervals are reported.

The output directory records resolved config, protocol, predictions, per-seed metrics, and finite-horizon impulse curves with clip counts. It is not a restartable graph snapshot; runs are ignored by Git and source/Nix revisions are not recorded.

## Intervention on trained graph strength

The previous experiment showed that scaling fixed initial weights improves delayed recall, but did not intervene on a trained graph. Here each seed and delay first trains the same random-target-feedback graph as in temporal plasticity, while its readout learns the true delayed bit. From that run we create five frozen graphs:

- **Initial**: original random graph.
- **Initial strength-matched**: original graph scaled to the trained random-feedback graph's mean absolute edge weight.
- **Random trained**: final random-target-feedback graph.
- **Random normalized**: that same trained graph scaled back to the original graph's mean absolute edge weight.
- **Task trained**: final task-feedback graph, included as an additional comparison.

The two scaling interventions use a single nonnegative multiplier for all weights, with clipping to [-0.5, 0.5]; bisection finds the target mean per seed and delay. Signs and topology are preserved, but the full distribution, correlations, spectral properties and individual edge changes are not matched. The random target is unrelated, but the online reader that supplies the prediction score is trained on true targets, so the entire edge-learning process is not independent of task information.

Each frozen graph gets a newly initialized readout trained on the same 6,000-symbol training stream; its state is reset before a frozen 2,000-symbol test stream. Thus the trained graphs have seen the training sequence during edge learning and again during readout training; the initial graph does not get the same graph-training pass. The online random learner and a causal shift register are also evaluated; unlike the frozen readers, the online learner's reader trained while its graph changed, and the shift register knows the delay and is not resource matched. The experiment measures whether the final frozen graph has useful properties after retraining the reader, not the direct effect of changing weights while keeping the original trained reader fixed.

We additionally run the previous zero-state, one-bit impulse probe: 64 pairs per graph, with first inputs 0 and 1, then identical continuations, recording RMS state distance for lags 0–16. This is a finite-horizon response from a reset state, not a stability guarantee or a direct measure of decodable memory.

Development seeds 1791001–1791010 and held-out seeds 1991001–1991020 use graph `s`, train `s+31000`, test `s+32000`, feedback signs `s+33000`, unrelated target `s+34000`, impulse continuation `s+35000` (wrapping `u64`). The runner rejects role collisions within a config and tests check cross-group roles. The experiment was not preregistered; the supplied settings did not change after the development run.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test trained_strength && cargo build --release --manifest-path toolkit/Cargo.toml --bin trained_strength'
nix-shell --run 'for group in dev test; do toolkit/target/release/trained_strength "toolkit/configs/trained-strength/$group.cfg" "toolkit/runs/trained-strength-$group-example"; done'
```

Pooled held-out accuracy, 20 paired seeds:

| Delay | Initial | Initial strength-matched | Random trained | Random normalized | Task trained | Online random | Shift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.6515 | 0.6582 | 0.6656 | 0.6568 | 0.6539 | 0.6566 | 1.0000 |
| 8 | 0.5300 | 0.5417 | 0.5797 | 0.5543 | 0.5398 | 0.5577 | 1.0000 |

At delay eight, random trained beat its normalized version in all 20 seeds (mean difference 0.0254), and beat the strength-matched initial graph in all 20 (0.0380). Random normalized beat initial in 19 seeds, with one tie (0.0243). The initial and normalized graphs had mean absolute weight 0.1244, and the strength-matched initial and random-trained graphs 0.1616. Mean impulse RMS after eight common inputs was 0.0116 initial, 0.0140 initial strength-matched, 0.0261 random trained, and 0.0189 random normalized. These findings support both a contribution from the scaling intervention and additional differences in the trained graph that a single average-strength statistic does not capture. They do not identify the additional property or prove that impulse persistence causes recall. Removing strength changes the entire graph uniformly, not only the portion learned during training; per-example outcomes within each stream are correlated and counts are descriptive, not formal significance tests.

Per-seed metrics, predictions, impulse curves, scaling/clip diagnostics, and all five edge-weight sets are recorded in the ignored output directory. Input weights and readers are not saved as restartable snapshots, and source/Nix revisions are not recorded. No task-change retention, growth, language generalization, runtime or memory comparison is measured.

## Where do the trained weights matter?

The preceding intervention found that the graph trained with unrelated feedback retains an advantage after matching average weight strength. This experiment tests whether *where* its signed weights are placed matters. Each seed and delay trains the same online random-target-feedback graph and freezes it. We compare it with:

- The original initial graph and the trained graph normalized to the initial graph's mean absolute edge weight.
- Three **global shuffles**, each independently permuting all trained signed weights across the existing directed edge entries.
- Three **destination shuffles**, each independently permuting trained weights only among edges entering the same destination unit. These preserve every destination's incoming signed-weight multiset in addition to the global multiset.

All shuffle controls keep every edge's endpoints, the exact signed weight multiset, input weights, and leak; destination shuffles additionally keep each destination's incoming weight distribution. They can break relationships among edges, their source and destination units, input weights, and the original initialization. Duplicate/self edges are kept. Thus a difference establishes sensitivity to these placements under this intervention, not a specific learned circuit or a mechanism. Fisher–Yates shuffles use a modulo mapping of a 64-bit pseudorandom integer, with negligible modulo bias; three separately seeded shuffles are used per control. As in the previous experiment, the random-target edge update is modulated by a reader trained on true task targets, so this is not unsupervised or task-independent learning.

After constructing each frozen graph, separately initialize and train a reader on the same 6,000-symbol stream, reset state, and evaluate its frozen weights on a separate 2,000-symbol stream. The trained graph has already seen that training stream during edge training; the initial graph has not. This is a two-pass protocol for trained graphs, and does not compare equal total training effort against the initial graph. A zero-state impulse probe measures mean RMS state distance between pairs given different first bits and then the same continuation for 16 inputs; it is not a direct measure of decodable memory or a stability guarantee.

Development graph seeds 2191001–2191010 and held-out seeds 2391001–2391030 use offsets `s` for graph, `+31000` train, `+32000` test, `+33000` feedback, `+34000` unrelated targets, `+35000` impulse, `+36000/+38000/+40000` global shuffles and `+42000/+44000/+46000` destination shuffles (wrapping `u64`). Configuration validation rejects role collisions within a group; tests check within- and cross-group roles. Configurations and three shuffle replicates were fixed before viewing development results, but the broader investigation and bootstrap contrasts were not preregistered.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test weight_organization && cargo build --release --manifest-path toolkit/Cargo.toml --bin weight_organization'
nix-shell --run 'for group in dev test; do toolkit/target/release/weight_organization "toolkit/configs/weight-organization/$group.cfg" "toolkit/runs/placement-$group-example"; done'
nix-shell --run 'python3 toolkit/scripts/weight_organization_summary.py toolkit/runs/placement-test-example/metrics.csv'
```

Pooled accuracy across 30 held-out graph seeds; shuffle columns average three per seed:

| Delay | Initial | Trained | Normalized | Globally shuffled | Shuffled within destination |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.6524 | 0.6645 | 0.6558 | 0.6578 | 0.6592 |
| 8 | 0.5262 | 0.5723 | 0.5462 | 0.5414 | 0.5477 |

At delay eight, trained beat the average of the three global shuffles in 29/30 seeds, and beat the average of the destination shuffles in 29/30. Mean paired advantages were 0.03085 and 0.02451. A descriptive 10,000-resample percentile bootstrap, sampling entire graph seeds with replacement after averaging their three shuffles, gave 95% intervals [0.02426, 0.03775] and [0.01851, 0.03082]. At delay four the paired advantages were 0.00665 [0.00453, 0.00879] and 0.00533 [0.00336, 0.00730]. The deterministic analysis script reproduces these intervals. They quantify seed variation under this protocol, not uncertainty across tasks or a preregistered confirmatory claim; multiple contrasts were examined. Adjacent examples are correlated, so individual predictions are not treated as independent samples.

For illustration, on delay-eight-trained graphs the mean impulse RMS at lag eight was 0.02183 trained, 0.01340 for global shuffle 0 and 0.01497 for destination shuffle 0; those are one of three shuffle replicates and don't establish causation. The result is evidence that the placement of trained weights contributes to performance on this delayed-bit task beyond the preserved weight multisets. It does not establish useful language representation, structural growth, biologically local learning, or that unrelated feedback itself is preferable to task feedback under matched resource and training budgets. Output includes all per-symbol predictions, per-seed metrics, impulse curves and edge-weight sets; runs are ignored by Git, are not restartable checkpoints, and do not record source/Nix revisions, runtime or peak memory.

## Is the random-feedback signal actually centered?

The random-feedback graph's edge scalar was `random_bit - sigmoid(class_1_score)`. The bit is a separately seeded pseudorandom target, but the score comes from a reader trained on the true delayed bits. Algebraically this scalar is `(random_bit - 0.5) + (0.5 - probability)`: it contains a reader-dependent component. The mean scalar can be close to zero while that component still correlates with graph state or eligibility traces. This study intervenes on the learning rule rather than treating the whole signal as random.

The **pseudo** learner retains the original signal; the **centered** learner instead uses `random_bit - 0.5`. They begin from the same graph, use the same feedback signs and pseudo bits, trace rule, rate, clipping, inputs, and true-label readout learning. They evolve separate graph states and readers, and each learns online for 6,000 bits. Pseudorandom bits are deterministically generated from a separate seed; this is not a proof of mathematical independence. The two signals can cause very different edge-update norms and trajectories, so the intervention does not isolate a direction while holding the amount of learning fixed. The task-feedback and initial frozen graphs are references.

We evaluate both their original online readers and fresh readers trained on the same training sequence after freezing each graph. The fresh-reader setup uses the training sequence once for graph learning and again for readout learning, unlike the fixed initial graph. All test results reset state and freeze weights for a separate 2,000-bit stream. A causal shift register is an exact, task-specific, non-resource-matched reference. The finite-horizon impulse probe starts two copies from zero state, feeds different first bits then 16 identical inputs, and averages RMS state distances across 64 continuations; it is not a direct measure of readable memory or stability.

Development seeds 2591001–2591010 and held-out seeds 2791001–2791030 use graph `s`, training `s+31000`, test `s+32000`, fixed feedback signs `s+33000`, pseudorandom targets `s+34000`, and impulse continuation `s+35000`, wrapping `u64`. The runner checks within-config role collisions and tests check within- and cross-group roles. Configurations were fixed before the development result, but this exploratory study and its multiple comparisons were not preregistered.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test centered_feedback && cargo build --release --manifest-path toolkit/Cargo.toml --bin centered_feedback'
nix-shell --run 'for group in dev test; do toolkit/target/release/centered_feedback "toolkit/configs/centered-feedback/$group.cfg" "toolkit/runs/centered-$group-example"; done'
nix-shell --run 'python3 toolkit/scripts/centered_feedback_summary.py toolkit/runs/centered-test-example/metrics.csv'
```

Pooled held-out accuracy, 30 graph seeds:

| Delay | Initial, fresh reader | Pseudo, fresh reader | Centered, fresh reader | Task, fresh reader | Pseudo online | Centered online | Shift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.6543 | 0.6663 | 0.6544 | 0.6558 | 0.6612 | 0.6540 | 1.0000 |
| 8 | 0.5316 | 0.5765 | 0.5325 | 0.5423 | 0.5548 | 0.5319 | 1.0000 |

At delay eight, pseudo with a fresh reader beat centered with a fresh reader in 30/30 paired seeds, with mean difference 0.04399. A 10,000-resample percentile bootstrap over graph seeds gives a descriptive 95% interval [0.03747, 0.05075]; at delay four the advantage was 0.01189 [0.00872, 0.01528], 28/30 seeds. The script reproduces the intervals. They describe seed variation on this task; correlated per-step examples are not treated as independent and the exploratory contrasts are not a preregistered significance claim.

At delay eight, mean absolute edge drift from initialization was 0.09790 for pseudo and 0.01995 for centered; mean absolute final weight was 0.15385 and 0.12708 respectively. The pseudo learner's mean class-one probability across training was 0.49904, so its global mean probability alone would miss the effect. Mean impulse RMS at lag eight was 0.02238 pseudo, 0.01098 centered and 0.01088 initial. This strongly implicates the reader-dependent term in the observed differences between these rules, but does not prove why it matters: changes in trajectory, amount of weight drift and state/eligibility correlations remain entangled. It is not evidence of language learning, self-organization, or that random targets are useful. A follow-up can compare the reader-dependent component alone and match cumulative updates or final drift.

Outputs include resolved config, protocol, per-symbol predictions, per-seed metrics, cumulative signal and drift diagnostics, impulse curves, and edge weights. They are not restartable checkpoints; output is ignored by Git, and source/Nix revisions, runtime and peak memory are not recorded.

## Reader-only feedback

The earlier edge signal is `random_bit - probability`, where `probability` is the true-label-trained reader's class-one output. Algebraically, `random_bit - probability = (random_bit - 0.5) + (0.5 - probability)`. The centered-feedback experiment removed the second term and lost the improvement. Here we ask the complementary question: can the **reader-only** signal `0.5 - probability` produce the improvement without the random bit?

The initial graph, input weights, fixed destination feedback signs, eligibility trace, 0.02 edge rate, clipping, training sequence and reader-learning rule are shared across paired models. The comparison includes the original combined rule, centered-random-only rule, reader-only rule, actual-task-target feedback, a fixed initial graph, and a causal shift register. Each learning graph evolves its own state, weights and true-label-trained reader; its probability is computed from its own trajectory. Therefore the component identity holds for the same probability at one step, but final graph changes are not expected to add linearly across separately evolving models. The centered random bits use a separate deterministic seed; the reader-only rule does not use those bits.

All models train on 6,000 symbols. Frozen-graph comparisons initialize a fresh reader and train again on the same 6,000-symbol stream after graph learning, then reset state and score the frozen reader and graph on a separate 2,000-symbol stream. This measures the final graph's utility for a fresh reader, not equal total training exposure against the initial graph. We also score each model's original online-trained reader; it adapts while the graph changes. The shift register knows the delay and is not resource matched. An impulse probe uses 64 reset-state trajectory pairs with different first bits and identical continuations; it measures finite-horizon state differences, not directly readable memory.

Development seeds 2991001–2991010 and held-out seeds 3191001–3191030 use graph `s`, train `s+31000`, test `s+32000`, feedback `s+33000`, random bit `s+34000`, and impulse continuation `s+35000`, wrapping `u64`. Within-group role collisions are rejected and tests check cross-group collisions. Configurations were fixed before seeing development results; this exploratory work and its contrasts were not preregistered.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test reader_signal && cargo build --release --manifest-path toolkit/Cargo.toml --bin reader_signal'
nix-shell --run 'for group in dev test; do toolkit/target/release/reader_signal "toolkit/configs/reader-signal/$group.cfg" "toolkit/runs/reader-$group-example"; done'
nix-shell --run 'python3 toolkit/scripts/reader_signal_summary.py toolkit/runs/reader-test-example/metrics.csv toolkit/runs/reader-test-example/weights.csv'
```

Pooled accuracy across 30 held-out seeds:

| Delay | Initial frozen | Combined frozen | Centered frozen | Reader-only frozen | Task frozen | Combined online | Reader-only online | Shift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.6564 | 0.6722 | 0.6568 | 0.6722 | 0.6589 | 0.6661 | 0.6658 | 1.0000 |
| 8 | 0.5359 | 0.5832 | 0.5364 | 0.5850 | 0.5454 | 0.5613 | 0.5628 | 1.0000 |

At delay eight, reader-only beat initial and centered on all 30 paired seeds. Its mean advantage over initial was 0.04911, with descriptive 95% seed-level percentile bootstrap interval [0.04088, 0.05763] from 10,000 resamples. Reader-only minus combined averaged 0.00186, interval [-0.00013, 0.00427] (16 wins, 13 losses, one tie). The latter interval includes zero and the experiment does not establish statistical equivalence; it shows the random component is not required to obtain a similar mean improvement in this tested setting. Adjacent query outcomes are correlated and are not treated as independent observations; intervals reflect seed variation on this task, not generalization to other tasks.

At delay eight, mean absolute edge drift from initialization was 0.10385 for reader-only and 0.10673 for combined, versus 0.02562 for centered. The cosine similarity of the 512-edge change vectors for reader-only and combined averaged 0.9532 across seeds (range 0.7919–0.9997); these graph changes are similar, not identical. Mean impulse RMS at lag eight was 0.02433 reader-only, 0.02388 combined, 0.01131 centered and 0.01114 initial. The reader-only signal is a real feedback signal from a supervised reader, not an independent random or unsupervised rule. It does not directly transmit the correct answer at the current step, but the reader's weights depend on previous true targets. The result does not identify which graph pathways carry memory, prove causality for the impulse measurement, or demonstrate language learning, growth, generalization to changing tasks, or resource-matched superiority. Source and Nix revisions, runtime and peak memory are not recorded; output files are ignored by Git and are not restartable checkpoints.

## Cross-delay transfer of reader-shaped graphs

Does reader-only graph feedback produce features useful to another memory task on different data? First train the graph using the reader-only signal `0.5 - probability` and a true-label reader on delayed-bit recall, with source delay either 4 or 8. Freeze the resulting graph, reset its state, then train a **new** reader for target delay 4, 8 or 12 on a separate stream. Finally reset state and evaluate on a third stream, with all weights frozen. Target delay 12 was never a graph-training target. However, the new reader is explicitly supervised on delay 12, so this is representation transfer within a family of related tasks, not zero-shot recall, general language ability, or inference without task-specific training.

Each source-delay group compares:

- The original fixed graph.
- The graph pretrained with reader feedback while its reader learns true delayed labels.
- A control pretrained by the same rule while its reader learns separately seeded pseudorandom labels on the same input stream. These labels are not a permutation and may match true labels by chance.
- The original graph scaled to match the true-pretrained graph's per-seed mean absolute edge strength, with clipping.
- The original fixed graph with all weights multiplied by 2 and clipped to [-0.5, 0.5].
- A causal shift register that knows the target delay and stores the required history; it is an exact reference and not resource-matched.

All graphs share the initial wiring, input weights and leak. Each graph receives its own fresh target-specific reader trained on the same reader-training sequence. Pretraining sees an additional 6,000 inputs and supervised labels; the initial and fixed controls receive no comparable graph-training compute, so budgets are not matched. Separately seeded random streams provide reproducibility, not a mathematical proof of independence. A role-collision check rejects overlapping seeds within configs and tests compare development and test roles.

Development seeds 3391001–3391010 and test seeds 3591001–3591020 use graph `s`, graph-pretraining inputs `s+31000`, evaluation inputs `s+32000`, feedback signs `s+33000`, random-label control `s+34000`, and fresh-reader training inputs `s+37000`, using wrapping `u64`. Graph pretraining and reader training each use 6,000 symbols; evaluation uses 2,000, with first `target_delay` steps unscored. Source delays and target delays were configured before development results. The overall investigation and reported comparisons were not preregistered; multiple source, target and control comparisons are exploratory.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test reader_transfer && cargo build --release --manifest-path toolkit/Cargo.toml --bin reader_transfer'
nix-shell --run 'for group in dev test; do toolkit/target/release/reader_transfer "toolkit/configs/reader-transfer/$group.cfg" "toolkit/runs/transfer-$group-example"; done'
nix-shell --run 'python3 toolkit/scripts/reader_transfer_summary.py toolkit/runs/transfer-test-example/metrics.csv'
```

Pooled held-out accuracy across 20 paired graph seeds:

| Graph training delay | Reader target delay | Initial | True-label graph | Random-label graph | Strength-matched | Fixed 2× | Shift |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 4 | 0.6463 | 0.6598 | 0.6471 | 0.6527 | 0.6535 | 1.0000 |
| 4 | 8 | 0.5443 | 0.6258 | 0.5455 | 0.5890 | 0.5968 | 1.0000 |
| 4 | 12 | 0.5015 | 0.5715 | 0.5018 | 0.5170 | 0.5249 | 1.0000 |
| 8 | 4 | 0.6463 | 0.6526 | 0.6471 | 0.6505 | 0.6535 | 1.0000 |
| 8 | 8 | 0.5443 | 0.5913 | 0.5456 | 0.5537 | 0.5968 | 1.0000 |
| 8 | 12 | 0.5015 | 0.5194 | 0.5018 | 0.5035 | 0.5249 | 1.0000 |

For source 4 and target 12, true-label pretraining beat initial, random-label and strength-matched controls in all 20 paired seeds. The mean advantage over strength-matched was 0.05455; a descriptive 10,000-resample seed-level percentile bootstrap interval was [0.04540, 0.06476]. It also beat fixed 2× in all 20 (mean 0.04658, interval [0.03697, 0.05755]). Per-symbol results are correlated, so the bootstrap samples graph seeds, not individual queries. These are exploratory intervals over seeds on this task family, not preregistered claims or uncertainty across tasks.

Mean absolute graph drift for true-label pretraining was 0.2003 at source 4 and 0.1057 at source 8; the random-label control drifted about 0.0228 for either. The unequal changes prevent attributing results solely to the label *content* rather than its effect on weight-update magnitude or trajectory. At source 8, the fixed 2× graph exceeded the true-pretrained graph on average for targets 8 and 12, underscoring that the learned graph is not universally superior even here. The observed cross-delay benefit supports reusable temporal features within this bit-recall family; it does not establish continual-learning retention, robustness to distribution shifts, novel grammar generalization, or a language model.

Run directories record the config, protocol, per-symbol predictions, per-seed metrics, graph edge weights and strength/clip diagnostics. Outputs are ignored by Git, do not contain restartable input weights/readers, and do not record source/Nix revisions, runtime or peak memory.

## Transfer to pattern detection

This experiment asks whether a graph shaped by reader-only feedback on four-step bit recall helps on a different sequence objective: after seeing the current bit, predict whether the last three bits are exactly `101`. Every window after the first two is scored, including overlapping windows. Independent fair bits make positives occur about one eighth of the time. Accuracy alone is therefore misleading; report confusion counts, positive recall, specificity, balanced accuracy and F1 as well. The exact-pattern reference keeps the last three bits and applies the known rule; the always-negative reference always predicts no match. Both use task-specific knowledge and are not resource-matched.

The graph is pretrained on the earlier four-step recall task using 6,000 bits, then frozen. A fresh logistic reader is trained online on `101` labels from a separately seeded 6,000-bit stream, and tested on another 2,000-bit stream after clearing state; the graph and reader are frozen in evaluation. Compare the original graph, the true-label-pretrained graph, a graph pretrained while its reader received separately seeded random labels, the initial graph scaled to match true-pretrained mean absolute weight, and the initial graph with weights scaled 2×. Each graph has its own reader and sees the same pattern-training and evaluation inputs. Pretrained graphs use additional training and compute. The two output logits learn independently; classification chooses the higher of the two output scores. No threshold tuning, positive weighting, or repeated examples were used in this first test.

Development seeds 3791001–3791010 and test seeds 3991001–3991030 use graph `s`, graph pretraining stream `s+31000`, evaluation `s+32000`, feedback signs `s+33000`, random-label control `s+34000`, and pattern-reader training `s+37000`, wrapping `u64`. Streams are deterministically seeded and seed-role collisions are checked; this is not a proof of statistical independence. The task and settings were fixed before running development and test, though the broader research was exploratory and not preregistered.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test pattern_recognition && cargo build --release --manifest-path toolkit/Cargo.toml --bin pattern_recognition'
nix-shell --run 'for group in dev test; do toolkit/target/release/pattern_recognition "toolkit/configs/pattern-recognition/$group.cfg" "toolkit/runs/pattern-$group-example"; done'
nix-shell --run 'python3 toolkit/scripts/pattern_summary.py original toolkit/runs/pattern-test-example/metrics.csv'
```

Pooled results across 30 held-out seeds:

| Model | Accuracy | Recall | Specificity | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| Initial graph | 0.8598 | 0.0000 | 0.9820 | 0.4910 | 0.0000 |
| True-label-pretrained graph | 0.8585 | 0.0007 | 0.9805 | 0.4906 | 0.0012 |
| Random-label-pretrained graph | 0.8588 | 0.0000 | 0.9808 | 0.4904 | 0.0000 |
| Strength-matched initial graph | 0.8600 | 0.0000 | 0.9823 | 0.4911 | 0.0000 |
| Fixed 2× graph | 0.8600 | 0.0000 | 0.9823 | 0.4911 | 0.0000 |
| Always negative | 0.8755 | 0.0000 | 1.0000 | 0.5000 | 0.0000 |
| Exact pattern detector | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

The graph readers failed to detect almost all positives under this training protocol; the always-negative baseline had higher accuracy. The development run had 0.0505 true-pretrained recall, but this did not replicate on the first held-out group. This is a negative result for the original reader setup, not proof that the graph states contain no pattern information.

### Follow-up: rare-class training

After observing the failed result, we ran a separate, exploratory intervention on fresh seed groups: keep the same binary task and graph pretraining but repeat each positive reader update seven consecutive times (negative windows receive one update). The reader's weights change between repeats, so this is sequential replay, not exactly a single gradient multiplied by seven. Compare that schedule with one update per window for the true-pretrained and initial graphs. Add a reader with the same two-output logistic rule given the three actual observed bits directly, trained under both schedules. The direct-history model has only 8 reader weights and an explicit buffer of three bits; graph readers have 258 reader weights and 128 state units plus 512 edges, so they are not matched in memory or compute. The direct model is an important task-specific baseline, not a general model.

The follow-up was chosen after inspecting the first test and is not a confirmation of its original protocol. New development seeds 4191001–4191010 and test seeds 4391001–4391030 use the same seed offsets and disjoint groups. The seven-update choice was fixed before running these new groups; no threshold tuning was performed. Graph-pretraining, reader-training and test streams are distinct in the same way as above.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test pattern_reweight && cargo build --release --manifest-path toolkit/Cargo.toml --bin pattern_reweight'
nix-shell --run 'for group in dev test; do toolkit/target/release/pattern_reweight "toolkit/configs/pattern-reweight/$group.cfg" "toolkit/runs/reweight-$group-example"; done'
nix-shell --run 'python3 toolkit/scripts/pattern_summary.py reweight toolkit/runs/reweight-test-example/metrics.csv'
```

Pooled follow-up accuracy over 30 new held-out seeds:

| Model | Accuracy | Recall | Specificity | Balanced accuracy | F1 |
|---|---:|---:|---:|---:|---:|
| Initial graph, one update | 0.8338 | 0.0000 | 0.9528 | 0.4764 | 0.0000 |
| Pretrained graph, one update | 0.8491 | 0.0304 | 0.9659 | 0.4982 | 0.0480 |
| Initial graph, seven positive updates | 0.6285 | 0.6537 | 0.6249 | 0.6393 | 0.3054 |
| Pretrained graph, seven positive updates | 0.6737 | 0.6771 | 0.6732 | 0.6751 | 0.3414 |
| Direct three-bit reader, one update | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Direct three-bit reader, seven positive updates | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Always negative | 0.8751 | 0.0000 | 1.0000 | 0.5000 | 0.0000 |
| Exact pattern detector | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Extra positive updates let the graph readers detect many matches, with substantial false positives. On balanced accuracy, the true-pretrained graph beat the initial graph in 18/30 paired seeds (two ties) under the seven-update schedule; mean per-seed advantage 0.03608, descriptive 10,000-resample seed bootstrap 95% interval [0.00336, 0.06805]. F1 advantage was 0.03731 [0.00669, 0.06800], 18 wins, seven losses and five ties. These are exploratory comparisons after a failed initial protocol; intervals describe seed variation and do not make individual, overlapping windows independent. The new schedule uses more training updates and changes the recall/false-positive tradeoff. The direct-history reader's perfect result shows a simple supervised model solves this task with access to the last three bits. The result does not demonstrate language generalization, meaningful semantic representations, or a competitive pattern detector.

Both runners log resolved config, protocol, per-seed confusion metrics and per-window predictions; the first also logs graph weights and strength diagnostics. These ignored outputs are not restartable checkpoints; source/Nix revisions, runtime and peak memory are not recorded.

## A–B–A rule changes and retention

Can the reader-feedback graph adapt to a new sequence rule and retain the old one? This experiment uses two noisy binary transition processes. In A, the next bit repeats the current bit with probability 0.9; in B, it flips with probability 0.9. A model predicts the next bit, then receives the correct answer and updates. It sees 4,000 transitions of A, 4,000 of B, then 4,000 of A again, each from a separately seeded stream. Graph state and eligibility traces are cleared at each phase boundary, while learned edge and reader weights persist. This isolates weight retention; it does not test a continuously running graph that infers an unannounced switch without reset. Models other than the oracle do not receive the rule label.

Compare the fixed recurrent graph and its online true-label reader with the same graph and reader trained with reader-only graph feedback `0.5 - class1_probability`; the graph-feedback model has 512 additional trainable edge weights. Both readers update from the true next bit. A recency-weighted Markov model updates one transition row with the same numerical rate 0.05, though its update and parameter count differ, so this is not a matched-compute or matched-learning-rate comparison. A second, **oracle-context Markov** model is given the correct rule label and keeps separate transition tables; it has extra information unavailable to the other models. Prediction always occurs before the corresponding update.

After each phase, clone each model, reset its state, and evaluate frozen weights on two independently generated 2,000-transition probe streams, one for A and one for B. The exact same A and B probes are reused at each checkpoint to make paired comparisons. The probes never update model parameters or feed information into later training; their repeated use during exploratory research is not an untouched final benchmark. Evaluating rule B immediately after A is a counterfactual before B training, not retention of a learned B. Oracle probes receive their true rule labels. Unlike ordinary training, frozen probes do not update the reader, though the graph's unit state continues to evolve within the probe.

Development seeds 4591001–4591010 and held-out test seeds 4791001–4791020 use `s` for graph, `s+33000` feedback, `s+40000` phase A, `s+42000` phase B, `s+44000` returning A, `s+46000` A probe and `s+48000` B probe, using wrapping `u64`. Within-config seed collisions are rejected and tests check cross-group role collisions. Seeded pseudorandom streams are reproducible but do not prove statistical independence. Model and protocol settings were fixed before development and held-out runs; the study and the comparisons were exploratory, not preregistered.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test continual_rule_change && cargo build --release --manifest-path toolkit/Cargo.toml --bin continual_rule_change'
nix-shell --run 'for group in dev test; do toolkit/target/release/continual_rule_change "toolkit/configs/continual-rule-change/$group.cfg" "toolkit/runs/continual-$group-example"; done'
nix-shell --run 'python3 toolkit/scripts/continual_rule_change_summary.py toolkit/runs/continual-test-example/probes.csv toolkit/runs/continual-test-example/windows.csv toolkit/runs/continual-test-example/training_tail_probes.csv'
```

Pooled frozen-probe accuracy across 20 held-out seeds:

| Checkpoint | Probe rule | Fixed graph | Feedback graph | Markov | Oracle-context Markov |
|---|---|---:|---:|---:|---:|
| After A | A | 0.7578 | 0.6846 | 0.9016 | 0.9016 |
| After A | B | 0.2682 | 0.3466 | 0.0975 | 0.4991 |
| After B | A | 0.1853 | 0.1898 | 0.0984 | 0.9016 |
| After B | B | 0.8429 | 0.8528 | 0.9024 | 0.9024 |
| After A returns | A | 0.7736 | 0.6662 | 0.9016 | 0.9016 |
| After A returns | B | 0.2822 | 0.3276 | 0.0975 | 0.9024 |

Both graph models' A accuracy dropped substantially after B; the feedback graph was 0.00445 above the fixed graph on that probe, with descriptive 10,000-resample paired-seed bootstrap 95% interval [-0.00005, 0.00860] and 13 wins, six losses and one tie. It had already scored 0.07315 *below* the fixed graph on A after A training, and it scored 0.10740 below after A returned. These observations do not meet the proposed criterion of a clear retention and learning advantage. The oracle retains both rules because it is given the rule identity and separate tables; this is not evidence that an unlabelled graph discovered context. With 10% transition noise, a model that knows the active rule has 90% expected next-bit accuracy, rather than 100%.

Online windows show a different aspect of adaptation. During the first 200 transitions of B, mean accuracy was 0.6522 for feedback, 0.6205 fixed, 0.7853 Markov and 0.8895 oracle; during the last 200 it was 0.8972, 0.8760, 0.8982 and 0.8982. In the first 200 after A returned, feedback scored 0.7358 versus 0.7770 fixed, 0.7893 Markov and 0.9055 oracle. Online predictions keep updating their readers and graph where applicable; they are not directly equivalent to frozen probes. Counts and intervals describe exploratory seed variation on this task; overlapping transition windows and multiple inspected comparisons are not independent preregistered tests.

**Post-hoc probe diagnostic.** The final online A-window accuracy looked better than frozen A-probe accuracy, so after observing the original test we added a frozen replay of the last 2,000 training transitions at each checkpoint and reran the same seeds. For example, after A, fixed graph scored 0.7570 on the frozen training tail and 0.7578 on the independent A probe; feedback scored 0.6841 and 0.6846. That makes a difference between the two streams an unlikely explanation for this gap in these runs; continuing online updates or other dynamics may matter, but this intervention does not isolate the cause. The replay is in-sample and is not a new held-out result.

`predictions.csv` records online guesses, `windows.csv` summarizes 200-step windows, `probes.csv` records frozen probe results, `training_tail_probes.csv` records the post-hoc replay, and `weights.csv` saves edge weights at checkpoints. Output is ignored by Git and is not a restartable checkpoint; source/Nix revisions, runtime and peak memory are not recorded.

## Pivot: unlabeled context reuse

The previous graph feedback rule did not retain rule A after training on B. This experiment pivots to an established hidden-state mixture of transition models, to measure how much context inference and separated memories can do before designing a graph mechanism. It is **not** a new architecture, a graph experiment, a growth mechanism, or proof of self-organized units. It uses two fixed memory slots from the start, a fixed switch hazard and global Bayesian responsibilities.

For each input bit, each slot has a learned probability of the next bit being 1. Before the next bit arrives, the model mixes the slots according to their current inferred context probabilities, adjusted by a fixed 0.01 switching hazard. After seeing the bit, it updates the context probabilities and changes each slot's relevant row in proportion to its posterior responsibility (rate 0.05, probabilities clipped to [0.001, 0.999]). The two tables are independently initialized from seeded random values in [0.2, 0.8), breaking symmetry without supplying rule labels. The model receives the input and target symbols, not the true context. Predictions and log loss use the prior before the target; logs also record the posterior after observation.

Each model sees a continuous 3,000-transition stream. A normally repeats the current bit and B normally flips it; independent per-transition noise flips the intended outcome with probability 0.10 or 0.25. Test three regimes: A–B–A with 1,000 transitions per phase; stationary A; and switches between A and B every 200 transitions. The mixture is not told block boundaries or reset between phases. Compare a single recency Markov model; the two-slot mixture; a **known-rule filter** that gets the exact A/B transition probabilities and the correct noise level but not the current context; and an **oracle-context Markov model** that is told the true context and keeps separate learned tables. The two references have additional information. The known-rule filter also uses the same 0.01 hazard, which does not match deterministic switch timing. Models have different parameter and compute costs; the comparison is not budget matched.

After each 1,000 transitions, frozen probes use separate 1,000-transition A and B streams, resetting beliefs to uniform before each probe. Slot probabilities remain frozen while context beliefs update after observed transitions. Thus the mixture can infer context during a probe but cannot use the true rule label; the oracle does receive the label. The same probe streams are reused across checkpoints and conditions for paired comparisons. The stationary B probe is deliberately a counterfactual on an unobserved rule, not a retention test. A second B memory in this fixed-capacity model is not a growth event, and slot disagreement is not a calibrated false-alarm metric.

Development seeds 4991001–4991010 and test seeds 5191001–5191030 use `s+40000` for training stream, `s+46000` for A probe, `s+48000` for B probe and `s+51000` for initial slot weights; `s` is reserved as a seed role, all offsets wrap `u64`. Seed-role collisions are rejected and tests check role separation between supplied configs. Conditions share corresponding seeds and random draws; different deterministic seeds do not prove statistical independence. The settings were held fixed between development and test but are exploratory and not preregistered.

```sh
nix-shell --run 'cargo test --manifest-path toolkit/Cargo.toml --test unlabeled_context && cargo build --release --manifest-path toolkit/Cargo.toml --bin unlabeled_context'
nix-shell --run 'for group in dev test; do toolkit/target/release/unlabeled_context "toolkit/configs/unlabeled-context/$group.cfg" "toolkit/runs/unlabeled-$group-example"; done'
nix-shell --run 'python3 toolkit/scripts/unlabeled_context_summary.py toolkit/runs/unlabeled-test-example/probes.csv toolkit/runs/unlabeled-test-example/windows.csv toolkit/runs/unlabeled-test-example/slots.csv'
```

Pooled frozen-probe accuracy after all three phases, 30 held-out seeds:

| Training regime | Noise | Probe | Single Markov | Unlabeled two-slot | Known-rule filter | Context oracle |
|---|---:|---|---:|---:|---:|---:|
| A–B–A | 10% | A | 0.8997 | 0.8916 | 0.8893 | 0.8997 |
| A–B–A | 10% | B | 0.1046 | 0.8778 | 0.8849 | 0.8954 |
| A–B–A | 25% | A | 0.7439 | 0.7383 | 0.7338 | 0.7439 |
| A–B–A | 25% | B | 0.2535 | 0.4245 | 0.7376 | 0.7465 |
| Stationary A | 10% | A | 0.8997 | 0.8997 | 0.8893 | 0.8997 |
| Stationary A | 10% | B, unseen | 0.1046 | 0.1046 | 0.8849 | 0.5005 |
| Stationary A | 25% | A | 0.7439 | 0.7439 | 0.7338 | 0.7439 |
| Stationary A | 25% | B, unseen | 0.2535 | 0.2535 | 0.7376 | 0.5003 |
| Switching every 200 | 10% | A | 0.8997 | 0.8922 | 0.8893 | 0.8997 |
| Switching every 200 | 10% | B | 0.1046 | 0.8891 | 0.8849 | 0.8954 |
| Switching every 200 | 25% | A | 0.7439 | 0.7289 | 0.7338 | 0.7439 |
| Switching every 200 | 25% | B | 0.2535 | 0.7191 | 0.7376 | 0.7465 |

On A–B–A with 10% noise, final frozen B recall exceeded 0.65 in 29/30 two-slot seeds (the threshold is a post-hoc descriptive count). It was near the known-rule filter's mean, while the single model mostly predicted the last A rule. At 25% noise, it exceeded 0.65 in only 4/30 seeds. Mean B-probe accuracy was 0.7461 immediately after B training but fell to 0.4245 after A returned. Mean per-seed absolute difference between the two slot transition-probability rows was 0.0534 after the first noisy A phase, 0.1344 after B, and 0.2401 after A returned. These diagnostics are consistent with the slots becoming similar early and later learning an imperfect separation, but they do not establish the cause of lost B recall. More frequent 200-step alternation at the same noise produced 0.7191 final B accuracy; the mean slot difference was 0.4459. Under stationary A, mean final slot differences were 0.015 at 10% noise and 0.004 at 25%, and both slots predicted the seen rule; no B rule was demonstrated to be learned. These fixed slots should not be mistaken for correct model selection or calibrated detection of a new context.

Online prediction tells a different story from a final counterfactual probe. At 10% noise in the 200-step-switching regime, single and mixture mean online accuracy were 0.7961 and 0.8787; mean log loss was 0.4820 and 0.3761. At 25% noise they were 0.6812 and 0.7209 accuracy, with log loss 0.6253 and 0.5990. On the long A–B–A stream, the single model had time to adapt, scoring 0.8840 versus 0.8854 at 10% noise and 0.7392 versus 0.7326 at 25%. These are exploratory comparisons on simple streams and fixed hyperparameters, with no formal significance or compute claim. The reference filter has knowledge of both rules and noise, and the oracle has true context.

The outcome provides a useful baseline for this pivot: explicit hidden-state inference and two preallocated memories can preserve and reuse the low-noise rules without context labels, but can collapse or forget under a noisier long-block schedule. The task does not require local graph computation, structural growth, language generalization or unknown numbers of contexts. A next experiment should study the noisy long-block failure and compare fixed-capacity mixture initialization/learning against a truly independent baseline before adding growth.

Output records per-step prediction probabilities, prior/posterior belief, window accuracy/log loss, frozen probes and slot snapshots. Outputs are ignored by Git, are not restartable checkpoints, and do not record source/Nix revisions, runtime or peak memory.

## Autonomous three-idea campaign

This was an operational test of the bounded agent workflow, not a preregistered model-discovery study. A three-idea campaign started from `c0671b0` and received the preceding failed candidate's results. Each agent edited the candidate and its tests in a separate worktree; the supervisor ran tests, committed the candidate, generated new development and evaluation seed groups, and applied the fixed `candidate_context` plan. Each later idea inherited the preceding candidate commit and saw the prior round's outcome. The candidates are therefore sequential refinements, not independent ideas. Evaluation seeds are deterministic and visible locally, and the agent could inspect the task generator; no claim of blind evaluation or novelty follows.

The fixed criterion required a mean accuracy improvement of at least 0.05 and wins on at least 60% of development seeds for the final B probe after noisy (25%) A–B–A, while limiting stationary online regression to 0.02 and mean regression in other final A/B probes to 0.03. Evaluation used 20 new seeds per idea after development passed. The original two-slot mixture, single-memory baseline, exact-rule/noise filter and labelled-context oracle remained unchanged. The known-rule and labelled-context references have extra information, and candidate and baseline compute budgets are not matched. The benchmark has complementary repeat/flip rules with equal noise; its generator and both noise levels are public to the agent.

| Idea | Candidate commit | Approach | Noisy B probe: candidate | Mixture | Paired wins/losses/ties |
|---:|---|---|---:|---:|---:|
| 1 | `f3d7c18d` | Complementary rule hypotheses with shared learned reliability | 0.7446 | 0.4394 | 17/3/0 |
| 2 | `6da9499` | Three context-switch timescales, shared reliability | 0.7439 | 0.4701 | 19/0/1 |
| 3 | `336ae71` | Bounded run-age distribution with changing switch hazard | 0.7524 | 0.4037 | 20/0/0 |

All three met the fixed development and evaluation criteria. Inspection matters: every candidate hard-coded complementary repeat/flip hypotheses and initialized shared reliability to `0.75`, which matches the 25%-noise target condition before learning. They thus exploit the structure of this benchmark, and none demonstrates discovering an unknown model family. The agents could inspect previous rounds' complete results; fresh per-round seeds reduce direct reuse but do not eliminate adaptive benchmark overfitting. The third model tracks 256 age/context masses and is not compute-matched to the two-slot baseline. Before making a broader claim, evaluate on asymmetric transitions, unequal or unseen noise, additional rules, and tasks whose structure was not exposed during candidate design; include fair priors and resource costs.

The three agent attempts took approximately 134 seconds in total according to the driver. The CLI reported 305,952 input tokens including 234,496 cached input tokens and 4,901 output tokens; reported token categories may overlap, and no monetary cost is calculated or capped. The campaign directory occupied approximately 1.2 GiB at inspection, including three worktrees and build artifacts; this is not a model-memory benchmark. The campaign completed without a reported agent, test or coordinator failure. It created local candidate branches but did not merge them. Raw logs and per-seed results remain under the ignored `toolkit/runs/autonomous-three-ideas/`; the worktrees were subsequently cleaned up and can be recreated from the candidate branches. Source/Nix reproducibility, untrusted-code review and holdout limitations remain as described in the toolkit README.

## Two-idea shared-cache workflow check

A two-idea campaign starting at `fdf358d` imported the previous three-idea campaign's results. Each sequential agent saw earlier results, and each round used fresh, different development and evaluation seeds with the same fixed criteria and public task generator described above. This was an operational check and exploratory comparison, not a blind or independent test of general model discovery.

| Idea | Commit | Approach | Evaluation noisy B accuracy | Mixture accuracy | Paired wins/losses/ties |
|---:|---|---|---:|---:|---:|
| 1 | `abdeb10` | Two independently learned transition tables with opposing priors and soft routing | 0.7375 | 0.3245 | 20/0/0 |
| 2 | `376d4c0` | Prior candidate with predictive-belief-gated learning and forgetting | 0.7475 | 0.4522 | 18/2/0 |

Both met development and evaluation criteria. Different evaluation seeds mean these numbers do not establish which candidate is better. The agents explicitly used opposing repeat/flip priors tailored to the task, so success does not establish that either discovers the rules from neutral assumptions. Future comparisons need matched seeds, neutral or equally informative priors, and tasks with different rule structures.

Reproduce the protocol with a fresh campaign identifier and hence different seeds:

```sh
nix-shell --run 'python3 -B -m toolkit.coordinator.campaign init toolkit/runs/cache-check-reproduction --max-ideas 2 --agent-seconds 600 --prior-campaign toolkit/runs/autonomous-three-ideas'
nix-shell --run 'python3 -B -m toolkit.coordinator.campaign run toolkit/runs/cache-check-reproduction --ideas 1'
nix-shell --run 'python3 -B -m toolkit.coordinator.campaign run toolkit/runs/cache-check-reproduction --ideas 1'
```

The first worktree was removed after its completed round, while the shared Cargo target directory remained at 310 MiB. After the second round, both worktrees and the shared cache were absent; the campaign retained 147 MiB of logs and evidence, and both candidate branches remain. The agent's first Cargo test compiled the workspace in 7.07 seconds and the second in 5.18 seconds, but both logs show workspace crates recompiling from their different worktree paths. This is not a controlled timing comparison or proof of cross-worktree build reuse. The supervisor's subsequent tests finished in 0.02 and 0.03 seconds after the agent's tests in each respective worktree. CLI usage was 357,172 input tokens including 282,624 cached and 3,361 output tokens; categories may overlap, and no cost cap or monetary estimate was applied. Raw artifacts are under the ignored `toolkit/runs/autonomous-cache-check/`.

## Matched candidate transfer and adaptation

This exploratory follow-up compares the two exact source snapshots from `abdeb10` (posterior-routed learning) and `376d4c0` (predictive-belief-gated learning) side by side against the original mixture, single online Markov model, and an oracle Markov model given the true context. All models see identical observations per seed. Every prediction precedes its target; only the oracle sees the context. After A–B–A training with 1,000 transitions per phase, separate A and B probes each use 1,000 new transitions, cloned model parameters, reset context beliefs, and frozen parameter learning; beliefs may update on probe observations. Online accuracy includes learning during each phase. Thirty deterministic seeds were selected before this run; each model sees the same stream and no agent participates or tunes on the result.

The transition pairs below list `P(next=1 | current=0), P(next=1 | current=1)` for A then B. Complementary is the original 25% noise condition; the other three were selected as simple tests of the candidates' opposing repeat/flip priors, not sampled from a distribution of tasks:

- Complementary: A `(0.25, 0.75)`, B `(0.75, 0.25)`.
- Asymmetric: A `(0.15, 0.70)`, B `(0.65, 0.25)`.
- Constant bias: A `(0.10, 0.10)`, B `(0.90, 0.90)`.
- Unequal noise: A `(0.05, 0.90)`, B `(0.70, 0.30)`.

Final frozen B-probe accuracy after A–B–A, mean across the same 30 seeds:

| Family | Single | Mixture | Posterior learning | Predictive gating | Context oracle |
|---|---:|---:|---:|---:|---:|
| Complementary | 0.2477 | 0.4630 | 0.7452 | 0.7463 | 0.7523 |
| Asymmetric | 0.2995 | 0.4177 | 0.6897 | 0.6899 | 0.7005 |
| Constant bias | 0.0995 | 0.8257 | 0.8213 | 0.8323 | 0.9005 |
| Unequal noise | 0.2987 | 0.3199 | 0.6905 | 0.6911 | 0.7013 |

Frozen recall alone hides the adaptation cost. Mean online accuracy in the B phase for those same families was respectively 0.7360/0.7272/0.7199/0.6798, 0.6818/0.6704/0.6608/0.6277, 0.8781/0.8708/0.7910/0.7980, and 0.6832/0.6711/0.6742/0.6735 for single/mixture/posterior/predictive. On constant bias, both candidates lost around eight percentage points against the mixture during B, while their final B probes were near it. Predictive gating in particular also lagged the mixture by about 4.7 and 4.3 points during B on complementary and asymmetric streams. These observations describe averages, not a causal explanation of the lag.

On the final B probes, predictive minus posterior mean paired accuracy was +0.0011, +0.00027, +0.01093 and +0.00063 respectively; wins/losses/ties across seeds were 16/4/10, 12/9/9, 18/0/12 and 13/7/10. These exploratory contrasts do not establish statistical significance or that predictive gating is generally superior. The asymmetric and unequal-noise pairs still resemble repeat versus flip, while constant bias is a more substantial departure. All are binary first-order Markov tasks with two fixed contexts and known phase lengths; the candidates still start with task-informed priors and are not compared at matched compute. Future tests should add neutral-prior ablations, more than two contexts, non-Markov structure, and switching-speed curves with matched resources.

```sh
nix-shell --run 'cargo test --offline --manifest-path toolkit/Cargo.toml --bin candidate_transfer'
nix-shell --run 'cargo run --offline --release --manifest-path toolkit/Cargo.toml --bin candidate_transfer -- toolkit/configs/candidate-transfer/matched.cfg toolkit/runs/candidate-transfer-reproduction'
python3 toolkit/scripts/candidate_transfer_summary.py toolkit/runs/candidate-transfer-reproduction/metrics.csv
```

The committed [per-seed metrics](toolkit/experiment-results/candidate-transfer/metrics.csv) and [aggregate summary](toolkit/experiment-results/candidate-transfer/summary.csv) capture the observed run. Candidate snapshots are byte-identical to the recorded commits. The simulation config and rule definitions are versioned; the Nix shell's nixpkgs is not pinned and wall time/memory were not measured.

## Automated neutral-prior and switching-window ablation

A follow-up uses the matched A–B–A protocol and same four rule families as the preceding transfer study, with 30 new configured seeds. The original candidates and baselines see identical streams alongside four ablations. Two start both memory tables at identical neutral `[2,2]` counts. Two start each context/input row with independently seeded `2±offset` counts, offset uniformly selected between -0.4 and 0.4; the row's total remains four, matching the originals. Decay in all four variants targets `[2,2]` rather than the task-informed opposing repeat/flip priors. The posterior and predictive variants retain their respective original update/forgetting rules. This simultaneously removes both initialization and persistent decay priors; it does not isolate those two effects individually. All receive no context label; the oracle alone does.

Selected mean accuracies (30 matched seeds):

| Family / measure | Mixture | Original posterior | Equal neutral posterior | Random neutral posterior | Original predictive | Equal neutral predictive | Random neutral predictive |
|---|---:|---:|---:|---:|---:|---:|---:|
| Complementary: final frozen B | 0.3963 | 0.7421 | 0.2498 | 0.7445 | 0.7435 | 0.2498 | 0.7447 |
| Asymmetric: final frozen B | 0.3536 | 0.6848 | 0.3053 | 0.6692 | 0.6844 | 0.3445 | 0.6773 |
| Constant bias: final frozen B | 0.8733 | 0.8294 | 0.8198 | 0.8236 | 0.8366 | 0.8198 | 0.8573 |
| Unequal noise: final frozen B | 0.3365 | 0.6880 | 0.3013 | 0.6875 | 0.6882 | 0.3013 | 0.6827 |
| Complementary: first 100 steps after A → B | 0.5817 | 0.5177 | 0.2390 | 0.2397 | 0.2590 | 0.2390 | 0.2390 |

Equal neutral slots remain exactly equal by construction under these deterministic, symmetric updates; they cannot develop two distinct memories without some symmetry-breaking mechanism. The seeded random initial differences permit different memories and substantially restore final recall on the repeat/flip-like tasks, without explicitly encoding those rules. They also adapt very poorly in the first 100 steps after the complementary switch. Constant bias has different input frequencies and is not explained solely by that final-recall pattern. The windows are descriptive online averages, not causal proof or a general result about symmetry breaking. The random-offset scale is a single design choice, no compute matching or statistical claim was made, and all tasks remain public two-context binary processes.

Run the entire local experiment and generate per-seed metrics, aggregate summaries, switch-window tables, and a provenance manifest with:

```sh
nix-shell --run 'python3 toolkit/scripts/run_candidate_ablation.py toolkit/configs/candidate-ablation/matched.cfg toolkit/runs/prior-ablation-reproduction'
```

The [generated report](toolkit/experiment-results/prior-ablation/report.md), [summary](toolkit/experiment-results/prior-ablation/summary.csv), [compressed per-seed metrics](toolkit/experiment-results/prior-ablation/metrics.csv.gz), and [manifest](toolkit/experiment-results/prior-ablation/manifest.json) are versioned. The recorded run was generated from clean commit `a292039`; the metrics matched an earlier independent invocation byte-for-byte. The manifest hashes uncompressed metrics and source/config files, but does not pin nixpkgs or measure time and memory. The runner does not invent experiments or apply a success threshold.

## First retention-and-adaptation scorecard campaign

A new milestone protocol compares an agent candidate with the original mixture on four development rule families. Its fixed criterion requires the mean paired candidate accuracy to be no more than 0.02 below the mixture for each family's final frozen B probe and the first 100 online predictions after each A→B and B→A switch: 12 checks in total. Development uses 10 seeds and evaluation, if eligible, uses 20 separate seeds and four different, fixed rule pairs. The evaluator and seeds are locally visible, so these are not blind holdouts. Counts and model/seed coverage are validated; passing non-inferiority does not demonstrate novelty, as simply copying the mixture can pass. A local scripted smoke candidate copied the baseline and achieved equality on all 12 development and evaluation checks; that is an infrastructure test, not a research result.

The first live two-idea campaign began from `aaa4a28` and imported the previous two-idea campaign history. The second agent received the first candidate and its complete development scorecard. Candidate commits are preserved as local branches:

| Idea | Candidate | Development | Failed checks | Evaluation |
|---|---|---|---|---|
| 1 | `e22d28c` | 8/12 checks pass | Both complementary switches and both asymmetric switches | Not run: development failed |
| 2 | `9c8f02b` | 11/12 checks pass | Unequal-noise B→A: 0.864 candidate vs 0.889 mixture, difference -0.025 | Not run: development failed |

The first used two learned transition tables with squared predictive-belief updates; the second used posterior-weighted updates discounted by assignment uncertainty. Their development seeds differ, so the check counts do not establish that the second mechanism is better. This is exploratory feedback, not a validated advance. All final B retention checks passed for both on their respective seeds; several switch checks failed. Before interpreting this as improvement, compare both models on identical seeds and evaluate genuinely withheld tasks or rules.

The first/second agent reported 148,695/119,362 input tokens including 116,224/86,144 cached, and 1,537/1,396 output tokens. Token fields may overlap; no spending cap or monetary estimate is enforced. Both completed worktrees and the shared build cache were removed, while ignored detailed campaign evidence remains under `toolkit/runs/milestone-first/`. The committed [campaign summary](toolkit/experiment-results/milestone-first/campaign-summary.json) preserves scorecards, resource diagnostics, ideas and usage; the frozen plans and development configurations are stored alongside it. The scorecard enforces stage wall time, seed count and logical transitions, while per-model training time and inline size are descriptive only; heap and peak memory are not measured and compute is not matched. The task families are small, fixed, public and vulnerable to repeated adaptive tuning.

```sh
nix-shell --run 'python3 -B -m toolkit.coordinator.campaign init toolkit/runs/my-milestone --max-ideas 2 --agent-seconds 300 --prior-campaign toolkit/runs/milestone-first'
nix-shell --run 'python3 -B -m toolkit.coordinator.campaign run toolkit/runs/my-milestone'
```

Campaign identifiers produce different seeds, and a coding agent is nondeterministic; this reproduces the protocol, not the generated candidate or exact scores. Hosted-agent usage may incur costs. Review candidate source before trusting claims, since the evaluator does not sandbox arbitrary model code as a complete security boundary.

## Matched validation of milestone candidates

The two candidate commits from the first milestone campaign, `e22d28c` and `9c8f02b`, were compared on the same 30 new development seeds. The candidate revisions, seed configuration, four development rule families, 12 non-inferiority checks, and paired descriptive bootstrap procedure were frozen in commit `1b1eb45` before the comparison. The runner checked that the evaluator and baseline files agree between the candidate commits and verified every per-seed control metric matched across runs. Both candidates received the same inputs, initialization seeds and scoring rules. The threshold remained a maximum 0.02 mean accuracy drop versus the mixture on final B recall and the first 100 steps after each switch, per family.

Idea 1 passed 8/12 checks and idea 2 passed 12/12. Selected matched mean accuracies:

| Development condition | Mixture | Idea 1 | Idea 2 | Idea 2 − idea 1 |
|---|---:|---:|---:|---:|
| Complementary: frozen B recall | 0.3988 | 0.3871 | 0.7228 | +0.3357 |
| Complementary: first 100 after A→B | 0.5557 | 0.4453 | 0.6067 | +0.1613 |
| Asymmetric: first 100 after B→A | 0.6867 | 0.6350 | 0.7673 | +0.1323 |
| Unequal noise: frozen B recall | 0.3407 | 0.4658 | 0.3949 | -0.0709 |

Because idea 2 passed development, both unchanged candidates were then run on 30 different seeds and four separate, already-public evaluation rule families. The evaluation seed list and protocol were committed as `5094f43` after the development result and before the follow-up. Seed roles are disjoint between these two runs. Idea 1 passed 9/12 checks; idea 2 passed 12/12. Selected means:

| Separate-rule condition | Mixture | Idea 1 | Idea 2 | Idea 2 − idea 1 |
|---|---:|---:|---:|---:|
| Mixed rules: frozen B recall | 0.4819 | 0.4966 | 0.6162 | +0.1196 |
| Reversed asymmetric: frozen B recall | 0.3688 | 0.3542 | 0.6854 | +0.3311 |
| Reversed asymmetric: first 100 after A→B | 0.5893 | 0.5163 | 0.6503 | +0.1340 |
| Similar rules: frozen B recall | 0.8479 | 0.8479 | 0.8469 | -0.0010 |

Idea 2 shows a consistent advantage over idea 1 on several tested switch and recall measures, but not all: its development unequal-noise B recall was lower. Some means conceal high seed variability: on separate-rule shifted-bias B recall, idea 2's mean advantage over idea 1 was 0.1490 even though it won on 14 seeds and lost on 15. The versioned summary includes per-measure wins/losses/ties and 10,000-draw seed-paired percentile bootstrap intervals; these are descriptive, unadjusted for multiple comparisons and not evidence of general superiority. The test families, evaluator and candidate source were publicly visible, and evaluation was selected after development inspection. These are not blind holdouts. Both candidates use the existing two-slot mixture structure, so the outcome does not establish architectural novelty. The non-inferiority threshold could be passed by an exact copy of the baseline.

The same four development and four evaluation pairs cover only selected binary first-order processes. Candidate-level training time is approximate, includes per-step clock overhead, and excludes probe computation; inline size excludes heap memory. No compute or memory matching was enforced. Next, a useful follow-up is a controlled ablation of the two update rules on genuinely held-out or broader task families, with multiple initialization seeds and switch rates, before expanding the agent search.

Reproduce using local Git commits and a Nix shell (new output directories are required):

```sh
nix-shell --run 'python3 -B toolkit/scripts/compare_milestone_candidates.py toolkit/configs/milestone-comparison/protocol.json toolkit/runs/milestone-matched-reproduction'
nix-shell --run 'python3 -B toolkit/scripts/compare_milestone_candidates.py toolkit/configs/milestone-comparison/evaluation-protocol.json toolkit/runs/milestone-evaluation-reproduction'
```

[Development summary](toolkit/experiment-results/milestone-matched/development/summary.json) and [evaluation summary](toolkit/experiment-results/milestone-matched/evaluation/summary.json) contain full scorecards and paired comparisons. Compressed per-seed metrics and resource diagnostics are versioned beside them. The runner uses temporary detached Git worktrees with one shared Cargo target directory and removes them after completion. Timing results are not expected to reproduce byte-for-byte. The Nix package set remains unpinned.

## Update-rule and switch-interval sweep

The frozen protocol and evaluator in `846c2b3` compared the original mixture, the two exact candidate snapshots (`e22d28c`, `9c8f02b`), a stronger-discount variant, a single Markov model, and a context-labelled oracle. The candidate snapshots were checked byte-for-byte against the recorded commits. A unit test independently verifies that idea 2 follows the mixture's inference and parameter update with its posterior responsibility raised to 1.25. The stronger-discount variant changes that exponent to 2; it is an ablation, not an optimized model.

Thirty fresh matched seeds were used for five binary first-order rule families and A–B–A phases of 100, 400 or 1,000 transitions each. The families include no rule change, a subtle change, high noise, a new asymmetric pair, and an uneven symbol bias. The frozen [protocol](toolkit/configs/update-rule-sweep/protocol.json) lists their transition probabilities. For each condition, the simulator reports phase online accuracy and log loss, the first 100 predictions of each phase, and frozen A/B probes after each phase. Different phase lengths also change the amount of training, so this is not a controlled isolation of switching frequency. All candidates receive the same stream and initialization seed; only the oracle receives the context. The test families are hand-selected and public, and the seeds are not a blind holdout.

Selected mean accuracies over the same 30 seeds:

| Family, phase length, measure | Mixture | Idea 1 | Idea 2 (power 1.25) | Power 2 |
|---|---:|---:|---:|---:|
| No change, 400, final B probe | 0.802 | 0.802 | 0.801 | 0.793 |
| High noise, 400, final B probe | 0.435 | 0.412 | 0.521 | 0.544 |
| New asymmetric, 1,000, final B probe | 0.523 | 0.479 | 0.630 | 0.518 |
| Uneven bias, 400, B probe immediately after B | 0.809 | 0.812 | 0.810 | 0.810 |
| Uneven bias, 400, B probe after A returns | 0.786 | 0.794 | 0.686 | 0.408 |
| Uneven bias, 400, first 100 after B→A | 0.898 | 0.900 | 0.885 | 0.841 |

The uneven-bias condition shows a negative result: both mixture and idea 2 had similar frozen B accuracy immediately after B, but idea 2 lost more after returning to A. For final B recall, its paired difference against the mixture was -0.1002, with 4 wins, 15 losses and 11 ties. A post-hoc seed-paired 10,000-resample percentile interval was [-0.1758, -0.0355]; it is descriptive, unadjusted for examining many conditions, and is not a formal significance claim. Stronger discounting was worse in this particular condition, while it improved high-noise B recall. These patterns do not support simply increasing the exponent across tasks.

After seeing that result, an inspection copy of idea 2 was added in `b2246cd`. It asserts prediction equality with the exact saved candidate at every simulation step and logs the mixture's transition tables and beliefs after each phase. The diagnostic rerun reproduced the original per-seed metrics byte-for-byte; the observation and all following analysis are post-hoc. For each seed on uneven bias at phase length 400, a heuristic selects whichever slot has the higher mean P(1|input) across both input symbols at the end of B and tracks that same slot after A. This is not a true context label. In 7/30 idea-2 runs, that slot's P(1|1) ended below 0.5; the mixture had 0/30 under the same heuristic, and power 2 had 23/30. Idea 2's mean B-recall difference versus the mixture was -0.3829 in its seven below-threshold runs and -0.0141 in the other 23. This correlation suggests a testable failure mechanism, but the slot selection and threshold were chosen after seeing the data, and probe-time beliefs can affect predictions. It does not prove that the slot threshold caused forgetting.

The compressed [per-seed metrics](toolkit/experiment-results/update-rule-sweep/metrics.csv.gz), [logged states](toolkit/experiment-results/update-rule-sweep/states.csv.gz), [compressed aggregate summary](toolkit/experiment-results/update-rule-sweep/summary.csv.gz), [post-hoc drift analysis](toolkit/experiment-results/update-rule-sweep/drift-analysis.json) and [manifest](toolkit/experiment-results/update-rule-sweep/manifest.json) are versioned. Reproduce with:

```sh
nix-shell --run 'cargo test --offline --manifest-path toolkit/Cargo.toml --bin update_rule_sweep'
nix-shell --run 'python3 -B toolkit/scripts/run_update_rule_sweep.py toolkit/configs/update-rule-sweep/protocol.json toolkit/configs/update-rule-sweep/matched.cfg toolkit/runs/update-rule-sweep-reproduction'
python3 -B toolkit/scripts/summarize_memory_drift.py toolkit/runs/update-rule-sweep-reproduction uneven_bias 400
```

No runtime or memory matching was measured in this sweep, the Nix package set remains unpinned, and hyperparameters and task choices were not surveyed exhaustively. The next discriminating test is to intervene on the selected memory or its routing after B, then measure recall on fresh paired seeds while controlling the intervention; a held-out task family would help distinguish this mechanism from a condition-specific artifact.

## Controlled memory-restoration intervention

Following the post-hoc uneven-bias observation above, a fixed intervention was committed in `623c2ca` before running 30 fresh seeds. The exact saved idea-2 candidate was run alongside an inspectable equivalent, with prediction equality asserted on every training step. A subsequent verification also checks equality throughout a frozen B probe after resetting both models; it reproduced the recorded metrics exactly. After 400 transitions each of A and B, the protocol saved both transition tables and selected the table with higher mean P(1|input), with ties going to slot 0. After another 400 transitions of A it compared frozen A and B probes for: unchanged idea 2 (sham); restoring the selected table to its end-of-B parameters; restoring the *other* table; and the unmodified mixture. Every arm reset its context belief before each probe. Two public high-P(1) B rules were tested, including the original uneven-bias family and a shifted pair; all arms shared the same streams and 30 seeds.

| Family, final B probe | Mixture | Sham idea 2 | Restore selected slot | Restore other slot |
|---|---:|---:|---:|---:|
| Uneven bias | 0.7643 | 0.7278 | 0.8112 | 0.7152 |
| Shifted bias | 0.8596 | 0.8378 | 0.8526 | 0.8370 |

On uneven bias, restoring the selected table changed mean B accuracy by +0.0834 versus sham, while restoring the other table changed it by -0.0126. The selected-slot change won/lost/tied on 12/12/6 seeds, its median was 0, and its range was -0.018 to +0.670. On shifted bias the mean selected-slot change was +0.0148, with 6/21/3 wins/losses/ties and median -0.0065. The mean benefits therefore came from large improvements on some seeds, not consistent gains for most seeds. On the A probe, restoring the selected table changed mean accuracy by -0.0045 and +0.0017 respectively. Descriptive, unadjusted paired bootstrap intervals and all per-seed outcomes are in the [summary](toolkit/experiment-results/memory-intervention/summary.json) and [metrics](toolkit/experiment-results/memory-intervention/metrics.csv).

This is a direct intervention: it supports the conclusion that selected-slot parameter drift can contribute to reduced B recall on some runs under these chosen conditions. It does not establish that drift explains every failure, and the shifted condition is less convincing. The intervention has a privileged end-of-B checkpoint, is applied after training, and selects the slot using a high-P(1) rule tailored to these two task families; it is not a deployable online learner or a general context-discovery method. The hypotheses came from post-hoc analysis, and all tasks are public. A follow-up should test an online policy against sham and random-slot controls across fresh rules and switch schedules, while measuring A adaptation and computational costs.

```sh
nix-shell --run 'cargo test --offline --manifest-path toolkit/Cargo.toml --bin memory_intervention'
nix-shell --run 'cargo run --offline --release --manifest-path toolkit/Cargo.toml --bin memory_intervention -- toolkit/configs/memory-intervention/matched.cfg toolkit/runs/memory-intervention-reproduction'
python3 -B toolkit/scripts/summarize_memory_intervention.py toolkit/runs/memory-intervention-reproduction/metrics.csv
```

## Online inactive-slot protection

After the privileged restoration study, the fixed [protocol](toolkit/configs/online-protection/protocol.json) and simulator were committed as `daa1cc3` before running 30 fresh paired seeds. The proposed online variant starts from idea 2's two-memory model. Before seeing each target it computes the predicted context belief; when one slot exceeds 0.8 probability it freezes the other slot for that update. It receives no true context label and stores no privileged end-of-B checkpoint. The idea-2 candidate is the unchanged saved source. The random-slot control instead chooses a uniformly random slot to freeze whenever *its own* predictive context belief exceeds 0.8. Each control has its own changing belief, so its freeze times and counts are not exactly matched to the proposed policy. The mixture is a further baseline.

Each model saw identical streams on four public binary rule families, with A–B–A phases of 400 and 1,000 transitions. Changing phase length also changes training exposure. Final frozen probes and the first 100 online predictions after both switches were measured. Selected means:

| Family, phase length, measure | Mixture | Idea 2 | Protect inactive | Random-slot control |
|---|---:|---:|---:|---:|
| Uneven bias, 400, final B recall | 0.7928 | 0.7543 | 0.7233 | 0.7927 |
| Uneven bias, 1,000, final B recall | 0.2481 | 0.3270 | 0.2930 | 0.5091 |
| Complementary, 1,000, final B recall | 0.4269 | 0.7006 | 0.6161 | 0.6373 |
| Shifted bias, 1,000, final B recall | 0.7482 | 0.8297 | 0.7793 | 0.8487 |
| No change, 1,000, final B recall | 0.800 | 0.800 | 0.800 | 0.800 |

This first protection rule did **not** meet its hypothesis: it reduced mean B recall relative to idea 2 in the first four rows, and the random-slot control had higher mean recall in those same conditions. On uneven bias at length 400, protection versus idea 2 had a mean paired difference of -0.0310 and 6/10/14 wins/losses/ties. Protection froze one slot on 88.1% of training steps on average; the random control froze one on 89.8%. At length 1,000, the rates were 80.5% and 86.9%. This makes overly broad protection or confidently wrong routing plausible failure explanations, not established causes: we did not record whether the inferred assignment was correct or isolate freeze timing. The random control's independently evolving state also prevents a strictly step-matched comparison.

The [summary](toolkit/experiment-results/online-protection/summary.json) reports paired differences, wins/losses/ties, unadjusted descriptive bootstrap intervals, adaptation metrics and suppression fractions for every condition. [Per-seed metrics](toolkit/experiment-results/online-protection/metrics.csv.gz) and [suppression counts](toolkit/experiment-results/online-protection/suppression.csv) are preserved. A second invocation reproduced both raw files byte-for-byte. The sample and task set are small and public; the fixed 0.8 threshold was not optimized; no compute matching or formal multiple-comparison correction was used.

```sh
nix-shell --run 'cargo test --offline --manifest-path toolkit/Cargo.toml --bin online_protection'
nix-shell --run 'cargo run --offline --release --manifest-path toolkit/Cargo.toml --bin online_protection -- toolkit/configs/online-protection/matched.cfg toolkit/runs/online-protection-reproduction'
python3 -B toolkit/scripts/summarize_online_protection.py toolkit/configs/online-protection/protocol.json toolkit/configs/online-protection/matched.cfg toolkit/runs/online-protection-reproduction
```

## Automated failure diagnosis campaign

A three-idea live campaign on the milestone task started at `a89188f` with prior campaign results and the current research-memory notes. After any failed development or evaluation scorecard, a separate read-only agent was asked to distinguish the measured observation from competing hypotheses and propose a controlled follow-up. Its output was explicitly marked unverified, preserved with its logs, and included in the next agent's context. Calls were bounded to 300 seconds per candidate and 120 seconds per diagnosis; no token or monetary budget was enforced.

Idea 1 (`ca4df50`) combined the uncertainty-discounted memories with a recency readout. Development failed one of 12 checks: unequal-noise B→A first-100 accuracy was 0.880 versus the mixture's 0.901. The diagnosis hypothesized that the recency readout interfered, named memory updates or seed variation as alternatives, and proposed an ablation. Idea 2 (`071ebf0`) implemented that ablation with and without the recency readout on identical streams and checked that the underlying memory trajectories matched. Its ten-seed agent-authored replay reported switch-to-A scores of 0.886 without recency, 0.885 with it, and 0.899 for the mixture; this offers little support for the specific recency-interference explanation on that sample. The replay used the agent's seeds 100–109, not the preceding campaign round's seed group, and its experimental design was agent-authored rather than independently validated. Its raw per-seed output was recorded in the agent event log; subsequent supervisor tests passed, although their captured stdout was not displayed under the then-current test command.

Idea 2 removed the recency contribution and passed the fixed development and separate-seed, separate-public-rule evaluation scorecards. Its default memory behavior is essentially the earlier uncertainty-discounted idea, so this is not an independent discovery. Idea 3 (`24e084b`) added gradual per-entry consolidation and an ablation control; it also passed both scorecards, but its generated test establishes only a simple mechanism property, not that consolidation caused a general improvement. All three candidate commits and detailed logs remain available, and the completed worktrees and shared cache were removed. The [campaign summary](toolkit/experiment-results/autonomous-diagnosis-first/campaign-summary.json) preserves the ideas, scorecards, diagnostic proposal, reported usage, and cleanup. This is an operational test of a diagnosis→follow-up loop; visible rules and seeds, agent-authored tests, unmatched budgets and repeated adaptive design preclude claims of blind validation or novelty.

The coding agents reported 262,271 + 314,615 + 161,797 input tokens and 2,156 + 2,457 + 1,878 output tokens. The diagnostic agent additionally reported 114,566 input and 658 output tokens; input totals include cached tokens and no monetary estimate or hard cost cap was recorded. Following this run, the supervisor was updated to preserve successful test stdout and accept optional structured follow-up counts, clearly labeled as agent-authored and not independently validated. A malformed or absent record is not treated as experimental evidence.

## Follow-up autonomous diagnosis campaign

A second three-idea campaign started at `79d3985` with the preceding diagnosis campaign in context. Idea 1 (`560b1c8`) combined parallel context filters with consolidation and missed three development switch checks. A read-only agent proposed distinguishing routing effects from learned-memory effects using paired frozen comparisons. Idea 2 (`2dc04c0`) implemented that control and an agent-authored ten-seed replay. Its development scorecard passed, but evaluation failed reversed-asymmetric switch-to-B by 0.0255 versus the mixture; a second diagnosis proposed crossing training history with inference mode. Idea 3 (`8698ad7`) implemented a crossed follow-up and surprise-sensitive reversible consolidation. It passed the configured development and evaluation scorecards. These candidates were generated sequentially, use different seed groups, and were evaluated against public rules; pass counts do not establish causal improvement or novelty. The agent's frozen-routing tests and simple consolidation test do not establish that the new consolidation caused the scorecard outcome.

The supervisor preserved `MURMUR_FOLLOWUP` test output, but initially kept only the first 50 structured records in its state. Reprocessing the intact test logs recovered 400 and 560 agent-authored records for ideas 2 and 3. They are archived as [idea 2 follow-ups](toolkit/experiment-results/autonomous-diagnosis-followup/idea-002.followups.json.gz) and [idea 3 follow-ups](toolkit/experiment-results/autonomous-diagnosis-followup/idea-003.followups.json.gz), alongside the [campaign summary](toolkit/experiment-results/autonomous-diagnosis-followup/campaign-summary.json). The extraction validates counts and arithmetic, not the test code or experimental design. The supervisor has since been changed to preserve up to 10,000 records per round, retain a bounded preview, and aggregate recognized per-seed labels while keeping the originals. The candidate worktrees and shared build cache were removed; detailed local logs remain under `toolkit/runs/autonomous-diagnosis-followup/`.

Reported candidate input tokens were 315,549, 643,943 and 564,562; the two diagnoses reported 77,745 and 100,345. These counts include any cached input as reported by the CLI, do not establish cost, and were not subject to a hard token cap. Subsequent campaigns may specify an optional reported-input threshold checked between rounds; it can be exceeded by a single round and does not cap money or output tokens. The next scientific step is an independent matched-seed comparison of the reversible-consolidation candidate and its controls, including tasks outside the public campaign scorecard.

## Bounded autonomous campaign with follow-up preservation

A further campaign began at `2394d1b`, carrying the previous diagnosis campaign and research-memory index into the coding agents' context. Its limit was five ideas, with a 1,500,000 reported-input-token stop threshold checked only *between* rounds. Four sequential candidates completed and passed the existing public development and evaluation scorecards:

| Idea | Commit | Agent-described change | Structured follow-up records |
|---|---|---|---:|
| 1 | `9ce5038` | Fast/slow error traces for consolidation release | 0 |
| 2 | `7a67b08` | Evidence-limited sharing between transition rows | 0 |
| 3 | `cf10ca9` | Learned same/opposite row relationship for sharing | 0 |
| 4 | `1b1bab5` | Release guard based on the other memory's prediction | 0 |

The candidates were sequential changes and their scorecards used different seed groups. The first used the prior reversible-consolidation candidate as inspiration, and later candidates accumulated earlier mechanisms. All four passed development and public-rule evaluation, so no diagnostic agent was invoked; the tests checked local mechanism properties but did not produce matched quantitative ablation records. Thus this run does not show that any added mechanism improved accuracy or caused the passes. A reviewer should compare the accumulated candidate against its controls on identical fresh seeds and more varied tasks before drawing a research conclusion; growing code complexity without a measured causal contribution is a risk.

After four rounds the CLI had reported 1,803,489 input tokens, including 1,539,072 cached input tokens, and 7,215 output tokens. The first three reported 1,354,299 input tokens, below the threshold; the fourth round then crossed it by 303,489. This illustrates that the threshold is not a hard cap or monetary budget. The driver stopped before idea 5 and removed the shared cache after validating completed-round cleanup. The [campaign summary](toolkit/experiment-results/autonomous-budgeted-01/campaign-summary.json) records the candidates, agent-reported usage, scorecards and cleanup; detailed local logs remain in `toolkit/runs/autonomous-budgeted-01/`. Changing the configured threshold or starting another campaign would incur further model usage.

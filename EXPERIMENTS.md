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

# Learning from Murmur

Murmur is both an experiment and a way to explore machine learning. Start with [EXPERIMENTS.md](EXPERIMENTS.md), run one experiment, and change one thing at a time. Results are evidence about a particular setup, not proof about all neural networks.

## The pieces in the current experiment

- **Example and target:** each trial presents a random cue (0 or 1), followed by one or more identical markers. The target is the original cue. The markers make the answer depend on earlier input.
- **State:** each unit holds a number that persists from one step to the next. Its update blends its previous value with a bounded response to the current input and messages from other units. This is working memory, not a learned parameter; it is reset between trials.
- **Weights:** fixed input and edge weights determine how signals affect units. Output weights are parameters that change through training. Unlike state, those weights persist between trials.
- **Prediction and learning:** the readout combines the unit states into scores. After seeing the correct answer, it changes each output weight in proportion to the corresponding unit's activity and the output's prediction error. This is an online supervised delta rule.
- **Training and evaluation:** training adjusts weights; evaluation freezes them and uses separately generated trials. Here both cue patterns appear in training, so held-out trials test repeated sampling, not unseen combinations.
- **Baseline and ablation:** the Markov baseline uses only the most recent marker. The reset control removes memory of the cue. The edge-free control preserves unit memory but removes communication between units. Comparing them helps identify what caused a result.

## A useful observation

At short delays the edge-free version performs as well as the full graph. Individual units can carry traces of the cue without exchanging messages. At longer delays performance falls toward chance. This illustrates why a positive result needs a control: accurate predictions alone do not tell us which mechanism caused them.

Try running `cargo run --release -- 2 results/my-run 8` inside `nix-shell`. Change the final argument to `4` and then `12`, and compare `metrics.json` files. Each run trains from scratch. For more reliable conclusions, repeat across seeds as shown in [EXPERIMENTS.md](EXPERIMENTS.md).

Next we will examine **credit assignment**: when a prediction is wrong, how can a connection inside the graph tell whether changing its weight would help? An eligibility trace is one possible way to connect earlier activity to a later feedback signal, but its usefulness must be tested rather than assumed.

## What happened when graph connections learned?

Run `cargo run --release --bin plasticity -- 11 results/my-plasticity 8` inside `nix-shell`. Compare `frozen_accuracy`, `learned_accuracy`, and `shuffled_accuracy` in `metrics.json`. Open `edge_changes.csv` to see that the learning and mismatched-feedback models can change their connections even when accuracy stays the same. The word *shuffled* in the filenames means feedback shifted by one trial; it is not a random shuffle.

An **eligibility trace** is a temporary record of a connection's recent activity. Here it combines activity at the sending and receiving units and gradually decays. At the end of a trial, the rule multiplies that trace by a prediction error and a readout weight. Intuitively, the trace says which connections were active, while the error provides feedback about the result. This rule is a hypothesis: activity alone does not tell us whether a connection helped or hurt, and the rule does not compute the true effect of each connection on the loss.

The result so far is negative: connection weights changed, but prediction accuracy did not improve relative to either control. In `diagnostics.csv`, `cue_state_distance` tells us how different the final graph states are for the two cues. `cue0_margin` and `cue1_margin` compare the two output scores: a negative margin predicts 0, a positive margin predicts 1. At the longest delay tested, the states still differed, but both margins had the same sign in every tested seed. That distinction illustrates why **a representation containing some information** and **a trained readout using that information** are different questions.

## A different reader

A **probe** is a separate, simple model used to ask what information is present in another model's internal state. In this experiment, the probe learns the average graph state for cue 0 and for cue 1. When it sees a new state, it guesses the cue whose average is closer:

```text
Is this state closer to the typical state for 0 or for 1?
```

At 24 markers, the original reader scored around chance, but the probe recovered the cue from every clean test state in these runs. So the graph retained a faint cue signal that the original reader did not use. This is not proof of robust memory: the test repeats just two deterministic patterns. When we added small artificial noise to what the readers see, the probe became less reliable. The noise was added to the final observation, not inside the graph.

Run `cargo run --release --bin probe -- 21 results/my-probe` in `nix-shell` and inspect `metrics.csv`. Compare the readers at delay 24 as `noise_half_width` increases. The experiment illustrates the difference between **information being present** and **a particular model being able to use it**.

## What if the intervening inputs vary?

The repeated-marker task looked like `1 → 2 → 2 → 2`. The new task uses a mix such as `1 → 4 → 2 → 3`; the goal is still to recall the initial cue. The reader is evaluated on complete distractor sequences it did not see during training. We also try adding a small random disturbance to each graph unit after every step.

With four varied inputs, the online reader still succeeded but the simple centroid probe made mistakes. With twelve varied inputs, both readers were near chance. This does **not** prove the cue has vanished from the graph: another decoder might recover it. It does show that our earlier success on repeated markers was an easier situation than remembering a cue through varied input.

Run `cargo run --release --bin robustness -- 31 results/my-robustness` in `nix-shell` and examine `metrics.csv`. Compare `varied=false` and `varied=true`, then compare the `internal_noise_half_width` values. Both readers learn and are evaluated under the same noise level, and noise changes the graph state at every step rather than only changing the final observation.

## The online sequence engine

Imagine four units, one for each symbol. Every unit has an arrow to every possible next symbol. An arrow's weight says how strongly that transition is expected. After the engine sees `0 → 1`, it nudges the arrow `0 → 1` upward and the other arrows leaving `0` downward. It does not need to retrain the whole graph.

```text
predict from the current symbol
observe what actually came next
adjust only the current symbol's outgoing arrows
repeat
```

This is mathematically the same idea as a **recency-weighted Markov model**: the current symbol determines the prediction, and recent examples count more than older ones. The connection weights persist, but the engine has no memory of a longer context.

We trained it on one rule and then switched to another. It adapted to the new rule while the frozen graph and ordinary cumulative-count baseline stayed with the old rule during this experiment. However, after adapting it answered according to the new rule for every tested symbol and had forgotten the old answers. That conflict is unavoidable for a single answer per symbol unless the engine receives or infers which rule is active.

Run `cargo run --release --bin engine -- 51 results/my-engine` in `nix-shell`. `predictions.csv` lets you inspect each guess before the model learned the answer. Compare `before_change.csv` and `after_change.csv` to see the arrows change. These files are snapshots for analysis, not saved models you can resume.

## Giving the engine context

Previously the engine had to use the same arrows for both rules. Now the caller supplies a label before each prediction:

```text
context A + symbol 0 → use A's arrows → predict 1
context B + symbol 0 → use B's arrows → predict 3
```

The model changes only the arrows belonging to the active context. As a result, learning B leaves A's arrows intact. In the experiment it remembered both rules, including when we switched between them. A standard context-aware Markov model did just as well, which tells us the main useful ingredient here is the **provided context label and separate memories**.

This is like giving someone a label that says which game is being played. The harder question is how to work it out when nobody provides the label. We have not built that part yet.

Run `cargo run --release --bin context -- 61 results/my-context` in `nix-shell`. Compare `after_a.csv` and `after_b.csv`: context A's arrows do not change while context B is being taught. `predictions.csv` shows the context, each guess, and the actual next symbol.

## Guessing which context is active

We removed the context label during a new test stream. Instead, the engine keeps a probability for each familiar rule. Before a prediction it asks: “How likely is each rule, and what would each one predict?” After seeing the next symbol it updates those probabilities. This is an example of a **hidden Markov model**: the active rule is hidden, while its effects on the observed symbols provide clues.

For example, if rule A is currently likely but the next few symbols fit B much better, the probability gradually shifts toward B. The engine cannot reliably know a switch happened before seeing any evidence. In the experiment it did well overall but usually missed the first prediction after an unannounced switch. A conventional count-based version of the same filter performed better.

The two rules were still taught with labels before this test. The engine is choosing between known rules; it is not discovering new rules on its own. Run `cargo run --release --bin hidden -- 71 results/my-hidden` in `nix-shell` and look at `predictions.csv`. `prior_context_a` is the estimate before the symbol; `posterior_context_a` is the estimate after it. Compare them just after steps 51, 101, and 151, when the hidden rule switches.

## Can it discover rules without labels?

In the latest experiment, the model did not receive labels saying which rule was active. It started with one memory, made predictions, and measured **surprise**: how unlikely it thought the observed symbol was. If surprise stayed high for two windows, it created a second memory. Each memory then learned in proportion to how likely it was to have produced the observed symbol.

On streams that alternated between two rules, the two memories learned the rules. But there was a catch: on streams where the rule **never changed**, random noise also triggered a second memory in every run. The model could learn useful memories after growing, but its decision **when to grow** was unreliable.

This illustrates a common research lesson: a system can get a useful answer for the wrong reason. We need to measure both detection of real changes and **false alarms**. Run `cargo run --release --bin discovery -- 81 results/my-discovery` in `nix-shell`; compare `switching-events.csv` with `stationary-events.csv`. The files also contain the surprise values and the final learned transition weights.

## Making a new memory earn its place

Surprise still opens a trial, but it no longer adds a permanent memory immediately. The model makes a temporary copy with an extra memory. The original and the candidate both learn from the next 100 observations, then stop changing their weights while they predict 100 later observations. If the candidate predicts those later observations enough better to overcome a chosen penalty, it is kept; otherwise it is discarded.

The comparison uses **log loss**, which rewards assigning higher probability to what actually occurs. The penalty expresses a cost for extra complexity. We chose the penalty on one group of random seeds, then tested on different seeds. In the separate test, the model accepted a second memory in all 20 streams that really switched rules, but also in 1 of 20 unchanged streams at each of two noise levels. That is an improvement on the earlier trigger, which added a memory on every unchanged run, but false alarms remain and the tasks are small.

Run `cargo run --release --bin trial_growth -- 201 results/my-trial 40` in `nix-shell`. Open `switching-trials.csv` and `stationary-trials.csv`: each row records the prediction improvement, penalty, and decision. The extra memory is not necessarily a real new rule merely because it helped a short prediction test.

## A harder test of memory growth

The trial-period approach looked good on short examples, so we ran longer unchanged streams and changed how often the rules switch. The result: an unchanged 5,000-step stream caused a false memory commit in 9 of 20 runs at 10% noise, compared with 1 of 20 over 1,200 steps. More attempts create more chances for an apparently convincing result by accident. This is the **repeated testing** problem.

The model also accepted a new memory in only 14 of 20 runs when rules switched every 200 steps. The fixed-length training and comparison windows may see different parts of a rule cycle; this is a hypothesis to test, not a proven cause.

A separate change detector uses a running score of evidence that observations fit an alternative distribution better than its learned normal distribution. It had fewer alarms on long unchanged streams at the selected threshold, but missed the rapid-switching case and does not learn memories. Run `cargo run --release --bin stress -- 301 results/my-stress` inside `nix-shell` and inspect `metrics.csv`. The comparison illustrates the tradeoff between missing changes and raising false alarms.

## Accounting for repeated attempts

One simple idea is to make each later candidate meet a higher bar. If the first attempt must improve prediction by 40 log-loss units, the next attempt must improve it by a little more, and so on. On a separate small test group, that reduced false memories in long unchanged streams without reducing the number of switching runs that committed. But this formula is a **heuristic**, not a proven way to control false alarms. It still admitted a false memory, and results varied across seed groups. A formally valid sequential test would require assumptions and evidence we have not established.

Run `cargo run --release --bin sequential -- 601 results/my-sequential` and inspect `trials.csv` to compare the thresholds and decisions for different scales.

## Exploring several ideas at once

Three prototypes explored different questions in parallel. One uses a graph of interacting units to clean up a corrupted stored pattern; it works for a few patterns but struggles as memories interfere. Another maintains uncertainty over how long the current rule has been active; it adapts to some changes but uses much more computation. A third tries to raise an alarm when the rule changes; its tested settings missed many changes.

These are different tasks, so their percentages are not a direct contest. The useful next question is whether the graph that cleans up patterns can also help with **sequences**, and whether it does so better than a simple lookup-based memory.

## Can the local graph complete a sequence?

We gave the graph stored sequences like `0, 2, 1, 3`, then hid the last symbol and asked it to recover it. Each position and symbol had a random signal; the graph saw the first three signals, with some bits flipped, and tried to settle on the missing one. A lookup table got the same noisy prefix and searched the stored examples for the closest match.

At 20% bit corruption, the graph recovered every missing symbol with two stored sequences, but only about 44% with 32. The lookup stayed near 99%. Even with no corruption, the graph struggled as more sequences were stored. This is evidence that the graph can perform a small retrieval task, but the straightforward lookup was substantially more accurate here.

The graph saw the complete stored sequences during training; the test used fresh corruptions of those same examples. It did not learn to predict new sentences or process a stream token by token. Run `cargo run --release --manifest-path alternatives/local_graph/Cargo.toml --bin sequence -- alternatives/local_graph/results/sequence` inside `nix-shell` to reproduce the comparison.

## Why did the graph struggle with more sequences?

The strongest clue is **interference**. We repeated each test after teaching a graph only the particular sequence being queried. That graph recovered the answer in every tested case. Of course, it had an unfair advantage: we told it which memory to keep. The comparison nevertheless shows that the same graph can represent each example in isolation.

We also inspected the “push” on each unit when the graph was placed in the correct answer. With 32 stored sequences, other memories reversed or canceled the target memory's helpful push for about 11% of the missing-symbol units. This is a diagnostic with access to the answer, not something the graph receives while predicting.

We tried making symbol signals perfectly distinct within each position. That helped only slightly. Removing connections among the missing-symbol units also helped modestly in one condition. So the issue is not explained solely by accidental similarity between random symbol codes or by the graph taking too many update steps. Many sequences share parts, and their learning updates are mixed into the same connections. The next experiment can test whether subtracting common patterns before learning reduces that interference.

## A different connection-learning rule

The original graph strengthens connections based on which units are active together. The new Storkey-style rule also considers what input the units already receive from their other connections. The hope is to avoid adding a new memory as though the graph were empty.

On the same stored-sequence task, this helped when storing 4, 8, or 16 sequences. At 8, for example, missing-symbol accuracy rose from about 70% to 77%. At 32 it fell slightly, from about 45% to 44%. The lookup baseline was still around 98% at that largest load. The rule reduced one measured form of conflicting signal, but that alone did not ensure that the graph settled on the correct symbol.

The toolkit now saves graph and lookup timing, approximate model-owned storage, and interference diagnostics. These numbers describe the current code and exclude important costs such as temporary allocations; they are not a fair resource-matched benchmark. The experiment still tests recall of sequences seen during training, not understanding new sequences.

## Spreading memories across graphs

We tested the idea of letting several small graphs each handle fewer memories. With 32 stored sequences, four graphs held eight each. Using the Storkey-style rule, the original single graph got about 43% of missing symbols right; the four-graph system got about 77% when the correct graph was provided, and about 76% when a router selected it from the noisy prefix.

There is an important catch: the router finds the nearest stored prefix, which is also nearly all the work the direct lookup baseline needs to answer. Lookup got about 99% correct on this task. And four graphs use about four times the graph storage. This supports the idea that spreading out memories reduces interference here, while leaving the harder question of how to route locally and efficiently unanswered.

## Does splitting still help with a similar storage budget?

We gave the single graph a larger budget: 256 units, enough connection-array space to match the four 128-unit graphs. On noisy cues, the larger single graph recovered about 46% of missing symbols with Storkey-style learning; the four routed graphs recovered about 76%. We also tried a different way of encoding inputs for the larger graph, which raised its result to about 49% but did not close the gap. Direct lookup still scored about 98%.

The comparison is useful but not perfectly matched: the four-graph system has more total units and uses a router that searches stored prefixes. It is still retrieving examples seen during training. The evidence is that this particular way of spreading the memories helps under a similar connection-storage budget—not that it has solved learning or routing.

## Can a small decision model choose the graph?

We trained a small decision model to select one of the four graphs from a noisy prefix. The assignment was supplied during training; the router learned from those labels. At test time it saw the prefix, but not the correct graph or missing symbol.

It chose the assigned graph about 55% of the time, compared with about 99% for searching stored prefixes. As a result, the four graphs recovered about 51% of missing symbols with the learned router, compared with 76% using prefix search. The decision model itself was faster in the local timing, but its mistakes lost much of the benefit of splitting memories.

Why? The assignments were arbitrary and the router made a simple linear decision, so the labels may be hard to express using that rule. Training much longer on clean examples did not reliably fit the labels in a small diagnostic, but that alone doesn't prove the model is incapable. A next experiment can give related inputs related graph assignments, then test whether the same router does better.

## A graph that learns from local prediction errors

We tried a different connection update. Each output unit looks at the signal it receives, compares its own prediction with the correct target bit during training, and adjusts its incoming connections. This is related to established error-correcting learning, not a newly invented principle. The output units also exchange signals with one another in the full version.

At 32 stored sequences, it predicted the missing symbol correctly about 53% of the time, compared with 44% for the earlier one-pass graph rules and 98% for lookup. But when we removed the connections between output units, it scored about 54%. That tells us the supervised error rule is useful on this task, but the extra recurrent interactions have not shown a benefit. The full graph also failed to reach a no-change state within the 20-step limit on roughly 43% of those queries.

The new graph practiced each stored sequence 30 times; the original rules saw each once. Repeating the existing Storkey update 30 times made its weights large and its answers worse in this particular setup, but that isn't a carefully tuned comparison. Both versions are still tested on corruptions of sequences they already saw. This is one useful building block, not yet a system learning language or organizing itself.

## Are unsettled graphs causing the mistakes?

We looked back at the 32-sequence experiment. The full graph answered correctly about 57% of the time when it settled within 20 sweeps, and 47% when it did not. That might make settling look like the cause of success.

But on those very same groups of inputs, the feedback-free model also dropped from about 58% to 49%. That suggests the unsettled cases may simply include harder inputs for both models. To work out cause and effect, we'd need to change the graph's update dynamics while keeping the learned connections and inputs the same, then observe what changes.

## What happens when the graph doesn't settle?

We kept the trained connections and inputs fixed and changed how the output units updated. Under the original fixed order, every query that failed to settle within 20 sweeps had already entered a repeating cycle. Most of these alternated between two states. Waiting up to 100 sweeps didn't make them settle or improve the answer.

We also asked units to ignore weak signals. That made the modified rule stop more often, but many stopped states would still change under the original rule—and accuracy hardly changed. This is a useful distinction: **stopping** is not the same as reaching the right answer, or even the original model's idea of a stable state.

## Did symmetric connections help?

We made the feedback connections between output units symmetric: if A pushes B with some weight, B pushes A with the same weight. The graph updates one unit at a time, while the input stays fixed. Under those conditions a known energy argument gives the graph a path toward a stable state.

In the experiment, the symmetric version stopped within 20 sweeps for every tested query. That solved the observed looping problem in this setting. But stopping isn't the same as answering correctly: its accuracy stayed close to the original graph and the version with no feedback. This is a useful distinction between improving a system's **dynamics** and improving the task it performs.

## Can the graph remember a continuous stream?

Instead of showing the graph the same short stored examples, we fed it a continuous stream such as `1, 0, 1, 1, 0, …`. After each new bit, it had to report the bit from a fixed number of steps earlier. Each delay was trained separately, and the test used a different stream.

At a four-step delay, the graph with connections answered correctly about 66% of the time; identical units without those connections managed about 52%. A simple buffer that remembers the last four inputs got every answer right. So the connections helped this graph carry a memory trace, but the model is still much less effective than an ordinary memory buffer.

The graph's connections were random and fixed. Only a separate reader learned from examples. That makes this evidence for the usefulness of recurrent interactions on this task—not evidence that the graph learned to organize itself.

## Can the connections learn too?

Previously, the connections stayed random while the reader learned. We now let each connection keep a short trace of how active it has been, and use the reader's prediction error to nudge its weight. We compared this with frozen connections and with connections nudged using unrelated random answers. All three readers still trained on the correct answers.

The graph did learn in the narrow sense that its weights changed. On new streams, it improved slightly over frozen connections, but random nudges did better. Those random nudges also moved the weights farther, so we can't yet tell whether their apparent benefit comes from the signal or simply the size of the changes. The next useful test would make those changes comparable in size and inspect what they do to memory. This is why a control matters: an improvement over doing nothing is not automatically evidence that our proposed teaching signal is working.

## Does the direction of a connection change matter?

We took the graph whose connections learned from the task. Then, for each connection, we made another graph that moved the connection by exactly the same amount but chose the direction randomly. We froze both graphs and taught a fresh reader to use each one.

For remembering eight steps back, the graph with the learned directions did a little better: about 54.6% versus 54.0%, and it won on 17 of 20 paired runs. At four steps, the difference was much smaller. That is a hint that the direction of the updates carries some useful information, though the result is small and specific to this task. Also, our earlier graph trained with unrelated answers still did better, so we have not explained that surprising result.

## What if the random nudges aren't bigger?

We ran two graphs side by side. One used the real answer to nudge its connections; the other used a random answer. At every step, we resized the second graph's nudge so its total size matched the first graph's nudge. This needs information from both graphs, so it is a measuring tool, not a standalone learning method.

On the eight-step test, the real-answer graph scored 55.6%, while the matched random graph scored 55.0%, close to the fixed graph's 54.9%. The real-answer graph won in 18 of 20 paired runs. The original random-answer graph with unrestricted nudges scored 57.3%. This supports the idea that the size of changes matters, but doesn't prove size alone explains the difference: the graphs end up with different weights, and our matching procedure changes the character of the random signal too.

## Can stronger connections help the graph remember?

We kept the graph's wiring fixed and made its connections weaker or stronger before training the reader. At an eight-step delay, the original graph answered correctly 54.2% of the time; tripling the starting connection weights, with a cap on their size, raised that to 62.1%. That stronger fixed graph beat the learned-connection versions on all 20 test runs.

We also gave two copies of each graph different first bits, followed by the same inputs, and watched how long their internal states stayed different. The stronger graph kept a trace of that first bit longer. That fits the idea that stronger interactions can give the reader a longer memory, although it doesn't prove the whole explanation. In an earlier run, random feedback had increased connection magnitudes on average; checking that graph directly is a useful next step.

## Is the random-trained graph just stronger?

We took a graph after training it with random feedback and turned all its connections down until their average strength matched the original graph. We also took the original graph and turned its connections up to match the random-trained graph. Then we froze each graph and taught each a new reader.

On the eight-step test, the random-trained graph scored 58.0%. Turning its connections down reduced it to 55.4%. But that was still better than the original graph's 53.0%, and turning the original graph up only reached 54.2%. So strength matters, but the way the individual connections changed also seems to matter. We don't yet know which property of those changes helps. The random-feedback graph's reader did learn correct answers while the graph was trained, so the whole training process wasn't independent of the task.

## Does it matter which connection gets which weight?

We took a trained graph and shuffled its connection weights while keeping the wiring and all the same weight values. We also tried a gentler shuffle that only swaps weights among connections entering the same unit. Each graph then got a freshly trained reader.

For recalling a bit eight steps back, the trained graph scored 57.2%. The globally shuffled versions averaged 54.1%, and the gentler shuffles averaged 54.8%. The trained graph beat each kind of shuffle in 29 of 30 paired runs. This is a clearer result: on this small task, the particular placement of weights matters, not just the overall range of weights. It still doesn't tell us which pattern is useful, whether it transfers to language, or whether random feedback is the best learning method.

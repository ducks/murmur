# Alternative: sparse local associative graph

This is a standalone experiment in binary associative memory, **not** a sequence learner. It is a sparse, symmetric [Hopfield-style attractor network](https://doi.org/10.1073/pnas.79.8.2554) with Hebbian learning, not a novelty claim. Each of 128 units has persistent ±1 state. Roughly 25% of possible undirected edges are sampled once, with no self-edges. Every trained edge independently adds `state_i * state_j / 128` when a complete training pattern is presented. A cue initializes all states; units then update in fixed sequential order to the sign of their weighted neighbors, retaining their state on ties. At most 20 full sweeps run, stopping early if none changes.

The stored patterns are random binary vectors, externally supplied in full; held-out test examples are new independent 20%-corrupted versions of the **same stored patterns**. This tests pattern completion, not generalization to unseen patterns or sequential language understanding. The zero-weight untrained control has identical topology and preserves the corrupted cue. Its accuracy is the same as simply returning the cue. It is deliberately weak; no comparison to nearest-neighbor lookup or parameter-matched trained baselines has been done. Fixed symmetric weights and a standard serial update guarantee non-increasing Hopfield energy mathematically, checked with a test; they do not establish arbitrary stability of other dynamics.

From the repository root:

```sh
nix-shell shell.nix --run 'cd alternatives/local_graph && cargo test && cargo run --release -- results'
```

Output is `results/metrics.csv`, with ten graph/pattern seeds 101–110, 25 independently corrupted cues per stored pattern per seed, and up to 20 sweeps. Results on these exploratory seeds, means across seeds:

| Stored patterns | Exact recall | Bit accuracy | Untrained bit accuracy |
|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 0.802 |
| 4 | 0.889 | 0.998 | 0.802 |
| 8 | 0.0865 | 0.947 | 0.800 |
| 16 | 0.000 | 0.788 | 0.800 |

As more patterns share the same edges, interference increases; at 16 patterns even average bit accuracy fell below the untrained cue. This is not a calibrated capacity boundary. Parameters and seeds were exploratory, not preregistered, and trials are correlated because many cues share the same stored pattern and graph. Storage uses a dense matrix despite sparse nonzero topology; this emphasizes inspection over computational efficiency. This prototype does not implement growth, pruning, online sequence processing, or tests for continual retention.

## Sequence completion and a lookup baseline

This extension stores four-symbol sequences drawn from a four-symbol alphabet. Each `(position, symbol)` is represented by a fixed, seeded random 32-bit ±1 code; the four codes are concatenated into the graph's 128 units. Up to 32 sequences with distinct three-symbol prefixes are stored, so each training prefix has exactly one intended final symbol. The full encoded sequences are presented once to the same local Hebbian rule. This is a static positional encoding of a whole sequence: it does not feed tokens through the graph over time.

At evaluation the first three encoded symbols are supplied, with independently flipped bits at the stated probability. These 96 observed bits are held fixed during asynchronous updates. The missing 32-bit final block is initialized with random ±1 values independent of the target and updated for up to 20 sweeps. Its final symbol is decoded by nearest Hamming-distance code in the fourth position, breaking ties toward the lowest symbol index. The graph is compared with (1) a nearest-neighbor lookup that stores the same training sequences and finds the closest *observed noisy prefix* in the same codebook, breaking ties toward the first stored sequence; (2) an untrained graph using the same random missing block; and (3) the most common training-set final symbol, ties toward the lowest symbol. Both graph and lookup see exactly the same cue. The lookup does not inspect the missing target.

Ten graph/codebook/dataset seeds (201–210), 20 newly generated corruptions and missing-block initializations per stored sequence, and loads 2/4/8/16/32 were tested. The test inputs are new corruptions, but the underlying full sequences and prefixes are the training sequences; this measures memory retrieval, not prediction of unseen sequences, compositional generalization, or language understanding. Results are correlated within each seed and stored sequence. The alphabet, code width, density, seeds, and settings are exploratory and were not preregistered.

```sh
nix-shell --run 'cargo test --manifest-path alternatives/local_graph/Cargo.toml && cargo run --release --manifest-path alternatives/local_graph/Cargo.toml --bin sequence -- alternatives/local_graph/results/sequence'
```

Mean missing-symbol accuracy at 20% independent prefix-bit corruption:

| Stored sequences | Graph | Lookup | Untrained graph | Majority symbol |
|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 0.270 | 0.600 |
| 4 | 0.874 | 1.000 | 0.285 | 0.525 |
| 8 | 0.690 | 0.998 | 0.263 | 0.463 |
| 16 | 0.567 | 0.991 | 0.255 | 0.375 |
| 32 | 0.444 | 0.986 | 0.249 | 0.347 |

With clean prefixes, the graph scored 1.000, 0.890, 0.738, 0.620, and 0.475 at the same loads; lookup scored 1.000 at every load. Thus corruption is not the only failure mode: the learned graph also struggles when clean prefixes cue increasingly many overlapping stored patterns. This result is consistent with interference, but the study does not establish its exact cause. Graph and lookup budgets are not matched: the graph stores a dense 128×128 weight array and topology mask (with approximately 25% of undirected edges active), whereas lookup stores sequences and shares the codebook. No time or memory benchmarks were taken. `metrics.csv` contains all seeds and the 0%, 10%, and 20% corruption conditions; output files are overwritten on rerun.

## Why does completion degrade?

This exploratory follow-up uses new graph/dataset seeds 301–310, sequence loads 2/4/8/16/32, and clean or 20%-corrupted prefixes. The graph, codebook, stored sequences, noisy prefix and random initialization of the missing block are paired across variants for each seed and condition. Each sequence has ten cue initializations. The same trained sequences are used in evaluation, so these measurements concern retrieval of familiar memories. The tests are descriptive and not preregistered.

Controls:

- **Target only:** train a graph with the same topology but only the target sequence. This deliberately provides an oracle that already knows which memory to keep; it is a diagnostic, not a deployable competitor.
- **No output feedback:** remove edges between units in the missing final block while keeping edges from the clamped prefix. This changes the network and can both help and hurt; it isolates one type of recurrent interaction.
- **One sweep:** decode after a single asynchronous sweep instead of up to 20. This probes the effect of further settling but does not remove within-sweep sequential effects.
- **Orthogonal codes:** replace random symbol codes with four balanced, exactly orthogonal codes *within each position*, using a seeded sign mask per position. Cross-position codes need not be orthogonal; this also changes the code distribution. Different sequences can still share symbols and therefore whole code blocks.
- **Oracle field diagnostic:** initialize the full network to the actual correct pattern, which is not available at prediction time. For each output unit, compare the target-aligned field from its target-only graph with the corresponding field from the all-pattern graph. Count cases where the former is positive and the latter is nonpositive. Because Hebbian contributions add, these are cases in which the other stored patterns reverse or cancel a positive target-only field at the true state. This is not a causal error fraction during actual noisy inference.

```sh
nix-shell --run 'cargo test --manifest-path alternatives/local_graph/Cargo.toml && cargo run --release --manifest-path alternatives/local_graph/Cargo.toml --bin diagnose -- alternatives/local_graph/results/diagnostics'
```

Mean results across the ten seeds, **clean prefix**, random codes:

| Stored | Full graph | Target only | No output feedback | One sweep | Lookup | Output bits correct | Oracle field flips |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |
| 8 | 0.729 | 1.000 | 0.788 | 0.739 | 1.000 | 0.869 | 0.063 |
| 16 | 0.521 | 1.000 | 0.561 | 0.533 | 1.000 | 0.798 | 0.098 |
| 32 | 0.463 | 1.000 | 0.484 | 0.451 | 1.000 | 0.751 | 0.106 |

At 32 stored sequences, exact orthogonality within each position raised clean-prefix symbol accuracy from 0.463 to 0.476, still far below lookup (1.000). With 20% corruption, the corresponding figures were 0.429 and 0.448, versus 0.986 and 0.984 for lookup. At clean 32-item load, removing output feedback raised accuracy from 0.463 to 0.484 with random codes and from 0.476 to 0.553 with orthogonal codes. These are mean paired-condition observations on ten seeds, not significance estimates or proof that a particular edge group alone is responsible for all errors.

The target-only control shows the architecture can recover each particular sequence in isolation under these settings, and the field diagnostic shows direct opposing contributions from other stored patterns at the correct state. Together they support **cross-memory interference** as a major explanation for the many-memory failure. Partial output reconstruction also survives: at 32 clean random-coded sequences, 75.1% of output bits matched the target even though only 46.3% of decoded symbols were correct. Exact within-position orthogonality did not remove the problem, so accidental correlation between different random symbol codes within a position is not a sufficient explanation. Shared symbols across sequences, uncentered Hebbian correlations, sparse connectivity, the decoder, and attractor dynamics remain possible contributors. The results do not identify a unique cause or demonstrate that a particular fix will generalize. `metrics.csv` reports all individual seeds, loads, codebooks, and both noise levels.

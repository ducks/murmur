# Independent alternative: Bayesian change-point alarm

This standalone prototype uses Bayesian online change-point detection with a Dirichlet–multinomial transition table per candidate run length. It retains a posterior over possible time since the latest change, uses a fixed 0.01 hazard, and flags when posterior probability of a change before the latest symbol exceeds a threshold. Hypotheses beyond 256 run lengths are truncated and the rest renormalized. It learns the transition distributions from symbols without true context labels. This is an established BOCPD-style technique, not a novel Murmur graph or a growth mechanism; it only raises alarms. The comparison is an independently implemented CUSUM-style count-model detector against a uniform alternative. Its baseline counts keep updating, so it is not an ideal fixed-null likelihood ratio test.

Both observe identical seeded streams. Stationary streams last 5,000 steps with 10% or 25% transition noise. Switching streams alternate each 50, 100, or 200 steps, with 10% noise, and last 1,200 steps. Alarms are threshold crossings after 200 steps. A switch is detected if an alarm crossing occurs in its first 20 transitions; switches through step 200 are not scored. Consecutive above-threshold samples count as one crossing. Detector thresholds are selected after inspecting 20 development seeds (1001–1020): pick the smallest threshold from the predefined grid that produced at most 2/20 stationary-run alarms at each noise level. This picked 0.32 for BOCPD and 10 for CUSUM. This procedure is exploratory, not preregistered, and the finite dev sample does not calibrate a reliable true false-alarm bound. Evaluation used separate seeds 2001–2040. Each condition reuses the same seed, so conditions are paired. The hazard, cap, and threshold grid were not optimized or independently validated.

From the main project root, in the Nix shell:

```sh
nix-shell --run 'rustc --edition 2021 --test alternatives/change_detection/main.rs -o /tmp/murmur-change-test && /tmp/murmur-change-test && rustc --edition 2021 -O alternatives/change_detection/main.rs -o /tmp/murmur-change'
nix-shell --run 'for s in $(seq 1001 1020); do /tmp/murmur-change "$s" "/tmp/murmur-change-dev-$s.csv" > /dev/null; done'
nix-shell --run 'for s in $(seq 2001 2040); do /tmp/murmur-change "$s" "/tmp/murmur-change-test-$s.csv" > /dev/null; done'
python3 alternatives/change_detection/summarize.py /tmp/murmur-change-dev-*.csv
python3 alternatives/change_detection/summarize.py /tmp/murmur-change-test-*.csv
```

Observed held-out results at those thresholds:

| Detector | Any alarm, stable 10% | Any alarm, stable 25% | Detected 50-step switches | Detected 100-step switches | Detected 200-step switches |
|---|---:|---:|---:|---:|---:|
| BOCPD, 0.32 | 6/40 | 5/40 | 0/800 | 0/400 | 0/200 |
| CUSUM-style, 10 | 1/40 | 1/40 | 0/800 | 0/400 | 42/200 |

The Bayesian detector's high threshold was poorly calibrated and missed all scored switches here; stationary false alarms also exceeded the observed development proportion. The CUSUM detector detected some 200-step switches but none of the faster switches at its selected threshold. An alarm can only fire once while its score remains above threshold, and the CUSUM baseline may adapt its counts to a mixture before monitoring starts. Fixed 20-step detection windows, specific hazards, uncertainty introduced by truncation, and reuse of simple four-symbol transitions limit conclusions. Neither experiment reports recovery of the old rule, memory growth or speed/space comparisons. These results do not demonstrate any advantage to this BOCPD implementation and do not rule out other Bayesian detectors or parameter choices.

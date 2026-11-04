#!/usr/bin/env python3
"""Seed-paired probe comparisons and mean online learning windows."""
import csv
import random
import statistics
import sys
from collections import defaultdict

if len(sys.argv) != 4:
    raise SystemExit('usage: python3 toolkit/scripts/continual_rule_change_summary.py <probes.csv> <windows.csv> <training_tail_probes.csv>')
rows = defaultdict(dict)
with open(sys.argv[1], newline='') as f:
    for row in csv.DictReader(f):
        key = (row['checkpoint'], row['probe_rule'], int(row['seed']))
        if row['model'] in rows[key]:
            raise SystemExit(f'duplicate probe model: {key}')
        rows[key][row['model']] = float(row['accuracy'])
expected = {'fixed_graph', 'adaptive_graph', 'markov', 'oracle_markov'}
if not rows or any(set(row) != expected for row in rows.values()):
    raise SystemExit('missing or unexpected probe models')
for checkpoint, rule in (('A', 'A'), ('B', 'A'), ('B', 'B'), ('A_return', 'A'), ('A_return', 'B')):
    values = [r['adaptive_graph']-r['fixed_graph'] for (phase, target, _), r in sorted(rows.items()) if phase == checkpoint and target == rule]
    if not values:
        raise SystemExit(f'missing probe: {checkpoint} {rule}')
    rng = random.Random(611)
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    print(f'checkpoint={checkpoint} rule={rule} adaptive_minus_fixed n={len(values)} mean={statistics.mean(values):.6f} bootstrap95=[{samples[249]:.6f},{samples[9749]:.6f}] wins={sum(x>0 for x in values)} losses={sum(x<0 for x in values)} ties={sum(x==0 for x in values)}')
with open(sys.argv[2], newline='') as f:
    windows = list(csv.DictReader(f))
for phase in ('A', 'B', 'A_return'):
    starts = sorted({int(x['window_start']) for x in windows if x['phase'] == phase})
    if not starts:
        raise SystemExit(f'missing windows for {phase}')
    for start in (starts[0], starts[-1]):
        vals = []
        for model in ('fixed_graph', 'adaptive_graph', 'markov', 'oracle_markov'):
            values = [float(x['accuracy']) for x in windows if x['phase'] == phase and int(x['window_start']) == start and x['model'] == model]
            if not values:
                raise SystemExit(f'missing window {phase} {start} {model}')
            vals.append(f'{model}={statistics.mean(values):.4f}')
        print(f'online phase={phase} start={start} '+ ' '.join(vals))
with open(sys.argv[3], newline='') as f:
    tails = list(csv.DictReader(f))
for phase, rule in (('A', 'A'), ('B', 'B'), ('A_return', 'A')):
    for model in ('fixed_graph', 'adaptive_graph'):
        values = [float(x['accuracy']) for x in tails if x['checkpoint'] == phase and x['model'] == model]
        if not values:
            raise SystemExit(f'missing training-tail probe {phase} {model}')
        print(f'frozen phase={phase} model={model} training_tail={statistics.mean(values):.4f} independent={statistics.mean(r[model] for (p, t, _), r in rows.items() if p == phase and t == rule):.4f}')

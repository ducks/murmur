#!/usr/bin/env python3
"""Paired seed-level percentile bootstrap for the weight-placement runner."""
import csv
import random
import statistics
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    raise SystemExit('usage: python3 toolkit/scripts/weight_organization_summary.py <metrics.csv>')
rows = defaultdict(dict)
with open(sys.argv[1], newline='') as f:
    for row in csv.DictReader(f):
        key = (int(row['delay']), int(row['seed']))
        name = row['model']
        if name in rows[key]:
            raise SystemExit(f'duplicate metric: {key} {name}')
        rows[key][name] = float(row['accuracy'])
expected = {'initial', 'trained', 'normalized', *(f'global_{i}' for i in range(3)), *(f'destination_{i}' for i in range(3))}
if not rows or any(set(row) != expected for row in rows.values()):
    raise SystemExit('missing or unexpected models')
for delay in sorted({key[0] for key in rows}):
    group = [row for (d, _), row in sorted(rows.items()) if d == delay]
    for comparison in ('global', 'destination', 'normalized'):
        diffs = [row['trained'] - (row[comparison] if comparison == 'normalized' else statistics.mean(row[f'{comparison}_{i}'] for i in range(3))) for row in group]
        rng = random.Random(27011991 + delay)
        samples = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
        print(f'delay={delay} trained_minus={comparison} n={len(diffs)} mean={statistics.mean(diffs):.6f} bootstrap95=[{samples[249]:.6f},{samples[9749]:.6f}] wins={sum(x > 0 for x in diffs)} losses={sum(x < 0 for x in diffs)} ties={sum(x == 0 for x in diffs)}')

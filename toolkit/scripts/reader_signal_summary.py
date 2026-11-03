#!/usr/bin/env python3
"""Descriptive paired-seed bootstrap and edge-update alignment."""
import csv
import math
import random
import statistics
import sys
from collections import defaultdict

if len(sys.argv) != 3:
    raise SystemExit('usage: python3 toolkit/scripts/reader_signal_summary.py <metrics.csv> <weights.csv>')
rows = defaultdict(dict)
with open(sys.argv[1], newline='') as f:
    for row in csv.DictReader(f):
        key = (int(row['delay']), int(row['seed']))
        if row['model'] in rows[key]:
            raise SystemExit(f'duplicate model: {key}')
        rows[key][row['model']] = float(row['accuracy'])
expected = {'initial_frozen', 'combined_frozen', 'centered_frozen', 'reader_frozen', 'task_frozen', 'combined_online', 'reader_online', 'shift'}
if not rows or any(set(row) != expected for row in rows.values()):
    raise SystemExit('missing or unexpected models')
for delay in sorted({key[0] for key in rows}):
    group = [row for (d, _), row in sorted(rows.items()) if d == delay]
    for name in ('initial_frozen', 'combined_frozen', 'centered_frozen', 'task_frozen'):
        diffs = [row['reader_frozen'] - row[name] for row in group]
        rng = random.Random(557 + delay)
        samples = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
        print(f'delay={delay} reader_minus={name} n={len(diffs)} mean={statistics.mean(diffs):.6f} bootstrap95=[{samples[249]:.6f},{samples[9749]:.6f}] wins={sum(x > 0 for x in diffs)} losses={sum(x < 0 for x in diffs)} ties={sum(x == 0 for x in diffs)}')
acc = defaultdict(lambda: [0.0, 0.0, 0.0])
with open(sys.argv[2], newline='') as f:
    for row in csv.DictReader(f):
        key = (int(row['delay']), int(row['seed']))
        combined = float(row['combined']) - float(row['initial'])
        reader = float(row['reader']) - float(row['initial'])
        sums = acc[key]
        sums[0] += combined * reader
        sums[1] += combined * combined
        sums[2] += reader * reader
for delay in sorted({key[0] for key in acc}):
    values = [v[0] / math.sqrt(v[1] * v[2]) for (d, _), v in acc.items() if d == delay and v[1] * v[2] > 0]
    if not values:
        raise SystemExit(f'no nonzero edge changes for delay {delay}')
    print(f'delay={delay} edge_change_cosine n={len(values)} mean={statistics.mean(values):.6f} min={min(values):.6f} max={max(values):.6f}')

#!/usr/bin/env python3
"""Paired graph-seed bootstrap for centered-feedback metrics."""
import csv
import random
import statistics
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    raise SystemExit('usage: python3 toolkit/scripts/centered_feedback_summary.py <metrics.csv>')
rows = defaultdict(dict)
with open(sys.argv[1], newline='') as f:
    for row in csv.DictReader(f):
        key = (int(row['delay']), int(row['seed']))
        if row['model'] in rows[key]:
            raise SystemExit(f'duplicate model for {key}')
        rows[key][row['model']] = float(row['accuracy'])
expected = {'initial_frozen', 'pseudo_frozen', 'centered_frozen', 'task_frozen', 'pseudo_online', 'centered_online', 'shift'}
if not rows or any(set(row) != expected for row in rows.values()):
    raise SystemExit('missing or unexpected models')
for delay in sorted({key[0] for key in rows}):
    group = [row for (d, _), row in sorted(rows.items()) if d == delay]
    for name in ('centered_frozen', 'initial_frozen', 'task_frozen'):
        diffs = [row['pseudo_frozen'] - row[name] for row in group]
        rng = random.Random(331 + delay)
        samples = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
        print(f'delay={delay} pseudo_minus={name} n={len(diffs)} mean={statistics.mean(diffs):.6f} bootstrap95=[{samples[249]:.6f},{samples[9749]:.6f}] wins={sum(x > 0 for x in diffs)} losses={sum(x < 0 for x in diffs)} ties={sum(x == 0 for x in diffs)}')

#!/usr/bin/env python3
"""Paired seed-level bootstrap for reader-transfer metrics."""
import csv
import random
import statistics
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    raise SystemExit('usage: python3 toolkit/scripts/reader_transfer_summary.py <metrics.csv>')
rows = defaultdict(dict)
with open(sys.argv[1], newline='') as f:
    for row in csv.DictReader(f):
        key = (int(row['source_delay']), int(row['target_delay']), int(row['seed']))
        if row['model'] in rows[key]:
            raise SystemExit(f'duplicate model: {key}')
        rows[key][row['model']] = float(row['accuracy'])
expected = {'initial', 'true_pretrained', 'random_pretrained', 'strength_matched', 'fixed_2x', 'shift'}
if not rows or any(set(row) != expected for row in rows.values()):
    raise SystemExit('missing or unexpected models')
for source, target in sorted({(s, t) for s, t, _ in rows}):
    group = [row for (s, t, _), row in sorted(rows.items()) if s == source and t == target]
    for name in ('initial', 'random_pretrained', 'strength_matched', 'fixed_2x'):
        diffs = [row['true_pretrained'] - row[name] for row in group]
        rng = random.Random(470 + source * 100 + target)
        samples = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
        print(f'source={source} target={target} true_minus={name} n={len(diffs)} mean={statistics.mean(diffs):.6f} bootstrap95=[{samples[249]:.6f},{samples[9749]:.6f}] wins={sum(x > 0 for x in diffs)} losses={sum(x < 0 for x in diffs)} ties={sum(x == 0 for x in diffs)}')

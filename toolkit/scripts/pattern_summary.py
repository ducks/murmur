#!/usr/bin/env python3
"""Paired seed-level pattern metrics and descriptive bootstrap intervals."""
import csv
import random
import statistics
import sys
from collections import defaultdict

if len(sys.argv) != 3 or sys.argv[1] not in ('original', 'reweight'):
    raise SystemExit('usage: python3 toolkit/scripts/pattern_summary.py original|reweight <metrics.csv>')
mode = sys.argv[1]
rows = defaultdict(dict)
with open(sys.argv[2], newline='') as f:
    for row in csv.DictReader(f):
        seed = int(row['seed'])
        if row['model'] in rows[seed]:
            raise SystemExit(f'duplicate model for seed {seed}')
        rows[seed][row['model']] = row
required = ({'true_pretrained', 'initial', 'random_pretrained', 'strength_matched', 'fixed_2x', 'always_negative', 'exact_pattern'} if mode == 'original' else {'initial_one', 'pretrained_one', 'initial_repeat7', 'pretrained_repeat7', 'history_one', 'history_repeat7', 'always_negative', 'exact_pattern'})
if not rows or any(set(row) != required for row in rows.values()):
    raise SystemExit('missing or unexpected models')
model = 'true_pretrained' if mode == 'original' else 'pretrained_repeat7'
controls = ('initial', 'random_pretrained', 'strength_matched', 'fixed_2x', 'always_negative') if mode == 'original' else ('initial_repeat7', 'pretrained_one', 'history_repeat7')
for field in ('accuracy', 'recall', 'balanced_accuracy', 'f1'):
    for control in controls:
        vals = [(float(row[model][field]), float(row[control][field])) for _, row in sorted(rows.items()) if row[model][field] and row[control][field]]
        if not vals:
            continue
        diffs = [a-b for a, b in vals]
        rng = random.Random(6001 if mode == 'original' else 771)
        samples = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
        print(f'metric={field} {model}_minus={control} n={len(diffs)} mean={statistics.mean(diffs):.6f} bootstrap95=[{samples[249]:.6f},{samples[9749]:.6f}] wins={sum(v>0 for v in diffs)} losses={sum(v<0 for v in diffs)} ties={sum(v==0 for v in diffs)}')

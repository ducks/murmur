#!/usr/bin/env python3
"""Paired-seed summary for unlabeled hidden-context experiments."""
import csv
import random
import statistics
import sys
from collections import defaultdict
if len(sys.argv) != 4:
    raise SystemExit('usage: python3 toolkit/scripts/unlabeled_context_summary.py <probes.csv> <windows.csv> <slots.csv>')
with open(sys.argv[1], newline='') as f:
    probes = list(csv.DictReader(f))
with open(sys.argv[2], newline='') as f:
    windows = list(csv.DictReader(f))
with open(sys.argv[3], newline='') as f:
    slots = list(csv.DictReader(f))
for regime in ('aba', 'stationary', 'switching'):
    for noise in ('0.1', '0.25'):
        selected = [x for x in probes if x['regime'] == regime and x['noise'] == noise and x['checkpoint'] == '3' and x['rule'] == '1']
        mixture = {x['seed']: float(x['accuracy']) for x in selected if x['model'] == 'mixture'}
        single = {x['seed']: float(x['accuracy']) for x in selected if x['model'] == 'single'}
        if not mixture or mixture.keys() != single.keys():
            raise SystemExit(f'missing paired probes: {regime} {noise}')
        diffs = [mixture[k]-single[k] for k in sorted(mixture)]
        rng = random.Random(283)
        samples = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
        print(f'regime={regime} noise={noise} probe=B mixture_minus_single n={len(diffs)} mean={statistics.mean(diffs):.6f} bootstrap95=[{samples[249]:.6f},{samples[9749]:.6f}] wins={sum(x>0 for x in diffs)} losses={sum(x<0 for x in diffs)} ties={sum(x==0 for x in diffs)}')
        for model in ('single', 'mixture', 'known_filter', 'oracle'):
            rows = [x for x in windows if x['regime'] == regime and x['noise'] == noise and x['model'] == model]
            if not rows:
                raise SystemExit(f'missing windows: {regime} {noise} {model}')
            n = sum(int(x['trials']) for x in rows)
            print(f'online regime={regime} noise={noise} model={model} accuracy={sum(int(x["correct"]) for x in rows)/n:.6f} logloss={sum(float(x["mean_logloss"])*int(x["trials"]) for x in rows)/n:.6f}')
        for checkpoint in ('1', '2', '3'):
            groups = defaultdict(dict)
            for row in slots:
                if row['regime'] == regime and row['noise'] == noise and row['checkpoint'] == checkpoint:
                    groups[row['seed']][row['slot']] = row
            if not groups or any(set(pair) != {'0', '1'} for pair in groups.values()):
                raise SystemExit(f'missing slot data: {regime} {noise} {checkpoint}')
            distances = [sum(abs(float(pair['0'][k])-float(pair['1'][k])) for k in ('p1_if_0', 'p1_if_1'))/2 for pair in groups.values()]
            print(f'slots regime={regime} noise={noise} checkpoint={checkpoint} mean_absolute_row_difference={statistics.mean(distances):.6f}')

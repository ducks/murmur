#!/usr/bin/env python3
"""Summarize per-seed candidate-transfer metrics (no statistical inference)."""
import csv
import statistics
import sys
from collections import defaultdict

if len(sys.argv) != 2:
    raise SystemExit('usage: candidate_transfer_summary.py <metrics.csv>')
groups = defaultdict(list)
with open(sys.argv[1], newline='', encoding='utf-8') as source:
    for row in csv.DictReader(source):
        key = (row['family'], row['phase'], row['measurement'], row['model'])
        groups[key].append((float(row['accuracy']), float(row['logloss'])))
writer = csv.writer(sys.stdout, lineterminator='\n')
writer.writerow(['family', 'phase', 'measurement', 'model', 'seeds', 'mean_accuracy', 'mean_logloss'])
for (family, phase, measurement, model), values in sorted(groups.items()):
    writer.writerow([family, phase, measurement, model, len(values),
                     f'{statistics.mean(x[0] for x in values):.8f}',
                     f'{statistics.mean(x[1] for x in values):.8f}'])

#!/usr/bin/env python3
"""Generate a descriptive report from matched per-seed ablation metrics."""
import csv
import statistics
import sys
from collections import defaultdict

MODELS = ('single', 'mixture', 'posterior', 'predictive', 'posterior_neutral',
          'predictive_neutral', 'posterior_random', 'predictive_random', 'oracle')

def report(path):
    values = defaultdict(dict)
    with open(path, newline='', encoding='utf-8') as source:
        for row in csv.DictReader(source):
            key = (row['family'], row['phase'], row['measurement'], row['model'])
            if row['seed'] in values[key]:
                raise ValueError(f'duplicate seed in {key}')
            values[key][row['seed']] = float(row['accuracy'])
    families = sorted({key[0] for key in values})
    lines = ['# Matched prior and adaptation ablation', '',
             'Accuracy means over matched seeds. Context labels are given only to the oracle.', '',
             'Models: posterior and predictive use the original task-informed priors; neutral uses identical neutral slots; random uses seeded, task-neutral random starting slots.', '']
    for phase, measurement, title in [('3', 'probe_b', 'Frozen B recall after A–B–A'),
                                      ('2', 'online', 'Online accuracy during B'),
                                      ('2', 'window_0000', 'First 100 transitions after A → B'),
                                      ('3', 'window_0000', 'First 100 transitions after B → A')]:
        lines += [f'## {title}', '', '| Family | ' + ' | '.join(MODELS) + ' |',
                  '|---|' + '---:|' * len(MODELS)]
        for family in families:
            groups = [values[(family, phase, measurement, model)] for model in MODELS]
            if not groups[0] or any(set(g) != set(groups[0]) for g in groups):
                raise ValueError(f'missing or unmatched seeds for {family}/{phase}/{measurement}')
            means = [statistics.mean(g.values()) for g in groups]
            lines.append('| ' + family + ' | ' + ' | '.join(f'{x:.4f}' for x in means) + ' |')
        lines.append('')
    lines += ['## Limits', '', 'Exploratory, selected binary tasks; no significance, compute matching, blind holdout, or generalization guarantee. Predictions in a window are online and include learning. The oracle receives context labels.']
    return '\n'.join(lines) + '\n'

if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: candidate_ablation_report.py <metrics.csv>')
    print(report(sys.argv[1]), end='')

"""Independently check a saved three-arm paired-count log; not a simulation rerun."""
from __future__ import annotations
import argparse
import gzip
import json
import random
import re
from pathlib import Path

from . import runner

ROW = re.compile(r'^seed=(\d+) stream=(\d+) init=(\d+) probe=(\d+) switch_b=\[(\d+), (\d+), (\d+)\] switch_a=\[(\d+), (\d+), (\d+)\] recall_b=\[(\d+), (\d+), (\d+)\]$')
MEASURES = ('switch_to_b', 'switch_to_a', 'retention_b')
ARMS = ('consistency_filter', 'unguarded', 'mixture')


def check(log, count, starts, draws=10000, bootstrap_seed=42517):
    """Validate the saved rows and aggregate records; bootstrap the primary outcome independently."""
    if count < 2 or draws < 100 or len(starts) != 3:
        raise runner.PlanError('need at least two seeds, three starts, and 100 bootstrap draws')
    rows, records = {}, {}
    for line in log.splitlines():
        match = ROW.fullmatch(line.strip())
        if match:
            n = list(map(int, match.groups()))
            seed = n[0]
            if seed in rows or not 0 <= seed < count or n[1:4] != [x + seed for x in starts]:
                raise runner.PlanError('duplicate, out-of-range, or inconsistent seed roles')
            values = [n[4:7], n[7:10], n[10:13]]
            if any(not 0 <= value <= limit for group, limit in zip(values, (100, 100, 1000)) for value in group):
                raise runner.PlanError('per-seed correct count outside trial count')
            rows[seed] = values
        elif line.strip().startswith('seed='):
            raise runner.PlanError('malformed seed record')
        if 'MURMUR_FOLLOWUP ' in line:
            try:
                rec = json.loads(line.split('MURMUR_FOLLOWUP ', 1)[1], object_pairs_hook=runner.object_no_duplicates)
                runner.exact_keys(rec, {'measure', 'candidate', 'control', 'candidate_correct', 'control_correct', 'trials'}, 'follow-up')
                key = (rec['measure'], rec['candidate'], rec['control'])
                if key in records:
                    raise runner.PlanError('duplicate follow-up record')
                for field in ('candidate_correct', 'control_correct', 'trials'):
                    if not isinstance(rec[field], int) or isinstance(rec[field], bool):
                        raise runner.PlanError('follow-up counts must be integers')
                records[key] = rec
            except (ValueError, TypeError, KeyError) as exc:
                raise runner.PlanError(f'invalid follow-up: {exc}') from exc
    if len(rows) != count:
        raise runner.PlanError(f'expected {count} unique seed rows, found {len(rows)}')
    expected = {(f'high_noise_200_{m}', ARMS[a], ARMS[b]) for m in MEASURES for a, b in ((0, 1), (0, 2), (1, 2))}
    if records.keys() != expected:
        raise runner.PlanError('follow-up comparisons missing or unexpected')
    comparisons = {}
    for w, measure in enumerate(MEASURES):
        per_trial = (100, 100, 1000)[w]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            key = (f'high_noise_200_{measure}', ARMS[a], ARMS[b]); rec = records[key]
            x = sum(rows[i][w][a] for i in range(count)); y = sum(rows[i][w][b] for i in range(count))
            if (rec['candidate_correct'], rec['control_correct'], rec['trials']) != (x, y, count * per_trial):
                raise runner.PlanError(f'aggregate does not match seed counts: {key}')
            comparisons[':'.join(key)] = {'candidate_correct': x, 'control_correct': y, 'trials': count * per_trial,
                                           'mean_paired_difference': (x-y)/(count*per_trial)}
    diffs = [(rows[i][2][0] - rows[i][2][1]) / 1000 for i in range(count)]
    randomizer = random.Random(bootstrap_seed)
    boot = sorted(sum(randomizer.choices(diffs, k=count)) / count for _ in range(draws))
    return {'seeds': count, 'seed_role_starts': starts, 'comparisons': comparisons,
            'primary_independent_bootstrap': {'draws': draws, 'rng': 'Python random.Random', 'seed': bootstrap_seed,
                 'percentile_95': [boot[int(draws*0.025)], boot[int(draws*0.975)-1]]},
            'limitations': 'Validates internal consistency of reported counts only; does not rerun model, verify seed generation or freshness, authenticate logs, or prove pairing, protocol compliance, or generalization.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--starts', type=int, nargs=3, required=True, metavar=('STREAM', 'INIT', 'PROBE'))
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=42517)
    args = parser.parse_args()
    opener = gzip.open if args.log.suffix == '.gz' else open
    with opener(args.log, 'rt', encoding='utf-8') as source:
        result = check(source.read(), args.count, args.starts, args.draws, args.bootstrap_seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

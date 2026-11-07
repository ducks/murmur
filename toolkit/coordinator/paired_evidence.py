"""Validate agent-reported paired count records against a planned aggregate.

This checks arithmetic and structure only; it cannot establish data provenance,
actual pairing, or whether the intended experiment was performed.
"""
import hashlib
import json
import random
from pathlib import Path
from . import runner

MARKER = 'MURMUR_PAIRED '
FIELDS = {'measure', 'candidate', 'control', 'seed', 'candidate_correct', 'control_correct', 'trials'}


def validate(path, milestone, followups, max_seeds=10000, draws=2000):
    expected = (milestone['primary_measure'], milestone['candidate_label'], milestone['control_label'])
    aggregates = [x for x in runner.read_json(followups['full_records_path'])['records']
                  if (x['measure'], x['candidate'], x['control']) == expected]
    if len(aggregates) != 1:
        raise runner.PlanError('paired evidence requires exactly one primary aggregate record')
    rows = {}
    with Path(path).open(encoding='utf-8', errors='replace') as source:
        for line in source:
            if MARKER not in line:
                continue
            try:
                item = json.loads(line.split(MARKER, 1)[1], object_pairs_hook=runner.object_no_duplicates,
                                  parse_constant=lambda _: (_ for _ in ()).throw(ValueError('invalid number')))
                runner.exact_keys(item, FIELDS, 'paired record')
                if (item['measure'], item['candidate'], item['control']) != expected:
                    raise runner.PlanError('paired record does not match planned primary comparison')
                for key in ('seed', 'candidate_correct', 'control_correct', 'trials'):
                    if isinstance(item[key], bool) or not isinstance(item[key], int):
                        raise runner.PlanError(f'paired {key} must be an integer')
                seed = item['seed']; n = item['trials']
                if seed < 0 or seed in rows or n < 1 or any(not 0 <= item[k] <= n for k in ('candidate_correct', 'control_correct')):
                    raise runner.PlanError('invalid or duplicate paired seed/counts')
                if len(rows) >= max_seeds:
                    raise runner.PlanError('too many paired records')
                rows[seed] = item
            except (ValueError, TypeError, KeyError) as exc:
                raise runner.PlanError(f'invalid paired record: {exc}') from exc
    if len(rows) < 2:
        raise runner.PlanError('at least two paired seed records are required')
    if len({x['trials'] for x in rows.values()}) != 1:
        raise runner.PlanError('paired seeds must have equal trial counts')
    totals = {key: sum(x[key] for x in rows.values()) for key in ('candidate_correct', 'control_correct', 'trials')}
    if any(totals[key] != aggregates[0][key] for key in totals):
        raise runner.PlanError('paired seed totals do not match primary aggregate')
    ordered = [rows[k] for k in sorted(rows)]
    diffs = [(x['candidate_correct']-x['control_correct']) / x['trials'] for x in ordered]
    rng = random.Random(42517)
    bootstrap = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(draws))
    output = Path(path).parent / 'paired-records.json'
    runner.write_json(output, {'schema_version': 1, 'records': ordered})
    return {'status': 'arithmetic_checked', 'seeds': len(ordered), 'totals': totals,
            'mean_paired_difference': sum(diffs)/len(diffs),
            'bootstrap': {'draws': draws, 'seed': 42517, 'percentile_95': [bootstrap[int(draws*.025)], bootstrap[int(draws*.975)-1]]},
            'records_path': str(output), 'records_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'qualification': 'Checks reported counts and aggregate arithmetic only; no proof of actual pairing, seed freshness, protocol compliance, or model correctness.'}

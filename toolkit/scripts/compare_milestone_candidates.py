#!/usr/bin/env python3
"""Compare committed campaign candidates on identical milestone streams."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from toolkit.coordinator import campaign, runner


def call(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def read_metrics(path):
    rows = {}
    with path.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = tuple(row[k] for k in ('seed', 'family', 'phase', 'measurement', 'model'))
            if key in rows:
                raise ValueError(f'duplicate row: {key}')
            rows[key] = row
    return rows


def compare(rows1, rows2, seeds, families, draws, bootstrap_seed):
    # Every fixed control must agree on every seed and metric, not just its mean.
    for model in ('mixture', 'single', 'oracle'):
        a = {k: v for k, v in rows1.items() if k[-1] == model}
        b = {k: v for k, v in rows2.items() if k[-1] == model}
        if a != b:
            raise ValueError(f'{model} results differ across candidate runs')
    results = {}
    for family in families:
        for label, phase, measurement in (('retention_b', '3', 'probe_b'), ('switch_to_b', '2', 'window_0000'), ('switch_to_a', '3', 'window_0000')):
            values = []
            for seed in seeds:
                prefix = (str(seed), family, phase, measurement)
                first, second, baseline = (float(rows1[prefix + ('candidate',)]['accuracy']),
                                           float(rows2[prefix + ('candidate',)]['accuracy']),
                                           float(rows1[prefix + ('mixture',)]['accuracy']))
                values.append((first, second, baseline))
            diff = [b-a for a,b,_ in values]
            rng = random.Random(f'{bootstrap_seed}:{family}:{label}')
            bootstrap = sorted(statistics.mean(rng.choice(diff) for _ in diff) for _ in range(draws))
            results[f'{family}:{label}'] = {
                'mixture_mean': statistics.mean(x[2] for x in values),
                'idea_1_mean': statistics.mean(x[0] for x in values),
                'idea_2_mean': statistics.mean(x[1] for x in values),
                'idea_2_minus_idea_1': statistics.mean(diff),
                'bootstrap_95_percentile': [bootstrap[int(draws*0.025)], bootstrap[min(draws-1,int(draws*0.975))]],
                'wins_losses_ties': [sum(x>0 for x in diff), sum(x<0 for x in diff), sum(x==0 for x in diff)],
            }
    return results


def main():
    if len(sys.argv) != 3:
        raise SystemExit('usage: compare_milestone_candidates.py <protocol.json> <new-output-directory>')
    protocol_path = Path(sys.argv[1]).resolve()
    protocol = runner.read_json(protocol_path)
    if protocol.get('schema_version') != 1 or set(protocol['candidates']) != {'idea_1', 'idea_2'}:
        raise ValueError('invalid protocol')
    config_path = (ROOT / protocol['configuration']).resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError('config must be in repository')
    cfg = runner.config_parse(config_path.read_text())
    if len(cfg['seeds']) != 30 or protocol['split'] != 'development':
        raise ValueError('this protocol requires 30 development seeds')
    commits = protocol['candidates']
    # The runners and baselines must be identical; only the candidate is varied.
    common = ['toolkit/src/bin/milestone_context.rs', 'toolkit/src/unlabeled_context.rs',
              'toolkit/src/continual_rule_change.rs', 'toolkit/src/temporal_stream.rs', 'src/data.rs']
    call(['git', 'diff', '--quiet', commits['idea_1'], commits['idea_2'], '--', *common], cwd=ROOT)
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    temp = Path(tempfile.mkdtemp(prefix='murmur-matched-'))
    worktrees = []
    clean = True
    try:
        env = {**os.environ, 'CARGO_TARGET_DIR': str(temp / 'target')}
        for label, commit in commits.items():
            worktree = temp / label
            call(['git', 'worktree', 'add', '--detach', str(worktree), commit], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            worktrees.append(worktree)
            evidence = output / label
            with (output / f'{label}.log').open('wb') as log:
                call(['cargo', 'run', '--offline', '--quiet', '--release', '--manifest-path', str(worktree / 'toolkit/Cargo.toml'),
                      '--bin', 'milestone_context', '--', str(config_path), str(evidence), protocol['split']],
                     cwd=worktree, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        reports = {}
        for label in commits:
            plan = campaign.make_plan(1, {'question':'matched', 'hypothesis':'paired', 'falsification':'scorecard'}, 'milestone')
            plan['criteria']['max_accuracy_drop'] = protocol['max_accuracy_drop_vs_mixture']
            reports[label] = runner.analyze_milestone(plan, cfg, output / label, protocol['split'])
        first, second = (read_metrics(output / label / 'metrics.csv') for label in ('idea_1', 'idea_2'))
        paired = compare(first, second, cfg['seeds'], runner.MILESTONE_FAMILIES,
                         protocol['bootstrap']['draws'], protocol['bootstrap']['seed'])
        summary = {'protocol': protocol, 'config_sha256': runner.file_digest(config_path),
                   'source_revision': call(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                   'source_dirty': bool(call(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, capture_output=True, text=True).stdout),
                   'scorecards': reports, 'paired': paired,
                   'metrics_sha256': {label: runner.file_digest(output / label / 'metrics.csv') for label in commits}}
        runner.write_json(output / 'summary.json', summary)
        shutil.copyfile(protocol_path, output / 'protocol.json')
        shutil.copyfile(config_path, output / 'development.cfg')
    finally:
        for worktree in reversed(worktrees):
            result = subprocess.run(['git', 'worktree', 'remove', str(worktree)], cwd=ROOT, capture_output=True)
            if result.returncode:
                clean = False
                print(f'Could not remove worktree {worktree}: {result.stderr.decode(errors="replace")}; preserving {temp}', file=sys.stderr)
        if clean:
            shutil.rmtree(temp)
    print(f'Completed comparison: {output / "summary.json"}')

if __name__ == '__main__':
    main()

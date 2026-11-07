"""Replay an archived paired campaign test without a model call.

Candidate code is executable and is not sandboxed by this tool. Run only
campaigns whose code and environment are trusted. This verifies reproducibility,
not the truth of a scientific hypothesis or global seed freshness.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from . import campaign, paired_evidence, runner


def unpack(path):
    with gzip.open(path, 'rb') as source:
        return source.read()


def verify_archive(root):
    summary = runner.read_json(root / 'campaign-summary.json')
    item = summary['round']; commit = item['commit']
    if not isinstance(commit, str) or not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise runner.PlanError('archive requires a full candidate commit')
    if campaign.git(runner.ROOT, 'rev-parse', '--verify', f'{commit}^{{commit}}') != commit:
        raise runner.PlanError('candidate commit did not resolve exactly')
    for source, archive in (('toolkit/src/agent_candidate.rs', 'candidate.rs'), ('toolkit/tests/agent_generated.rs', 'test.rs')):
        original = subprocess.run(['git','show',f'{commit}:{source}'], cwd=runner.ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, check=True).stdout
        if original != (root / archive).read_bytes():
            raise runner.PlanError(f'archived {archive} differs from candidate commit')
    allocation_bytes = unpack(root / 'seed-allocation.json.gz')
    records_bytes = unpack(root / 'paired-records.json.gz')
    for data, expected in ((allocation_bytes,item['seed_allocation']['sha256']),
                           (records_bytes,item['paired_evidence']['records_sha256'])):
        if hashlib.sha256(data).hexdigest() != expected:
            raise runner.PlanError('archived allocation or paired records do not match their recorded hash')
    allocation = json.loads(allocation_bytes)
    expected_allocation = campaign.allocate_primary_seeds(summary['campaign_id'], item['index'], allocation['count'])
    if allocation != expected_allocation:
        raise runner.PlanError('archived seed allocation does not match deterministic allocation')
    original = json.loads(records_bytes)
    return item, allocation, original


def replay(archive, output, timeout=600, shared_cache=None):
    archive = Path(archive).resolve()
    item, allocation, original = verify_archive(archive)
    output = runner.output_location(output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise runner.PlanError(f'cannot create replay directory: {exc}') from exc
    worktree = output / 'worktree'; cache = Path(shared_cache) if shared_cache is not None else output / 'build-cache'; log = output / 'tests.log'
    result = {'schema_version':1, 'status':'running', 'commit':item['commit'], 'archive':str(archive),
              'qualification':'Same code and environment rerun; does not independently validate task design or global seed freshness.'}
    runner.write_json(output / 'result.json', result)
    try:
        campaign.git(runner.ROOT, 'worktree', 'add', '--detach', str(worktree), item['commit'])
        if shared_cache is None:
            cache.mkdir()
        elif not cache.is_dir() or cache.is_symlink():
            raise runner.PlanError('shared replay cache is missing or symlinked')
        with log.open('wb') as out:
            code = campaign.bounded_process(['cargo','test','--offline','--manifest-path','toolkit/Cargo.toml',
                         '--test','candidate_context','--test','agent_generated','--','--nocapture'],
                         cwd=worktree, stdout=out, stderr=subprocess.STDOUT, timeout=timeout,
                         env={**os.environ,'CARGO_TARGET_DIR':str(cache)})
        if code:
            raise runner.PlanError(f'replay tests failed ({code}); inspect tests.log')
        followups = campaign.followup_metrics(log)
        check = paired_evidence.validate(log, item['milestone'], followups,
                                         expected_seed_ids={r['seed_id'] for r in allocation['rows']})
        replayed = runner.read_json(Path(check['records_path']))
        if replayed != original:
            raise runner.PlanError('replayed paired records differ from archived records')
        if campaign.git(worktree, 'rev-parse', 'HEAD') != item['commit'] or campaign.changed_paths(worktree):
            raise runner.PlanError('replay changed its worktree; preserved for inspection')
        ignored = subprocess.run(['git','status','--porcelain=v1','--ignored=matching','--untracked-files=normal'],
                                 cwd=worktree,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True).stdout
        if ignored:
            raise runner.PlanError('replay created ignored or untracked files in its worktree; preserved for inspection')
        campaign.git(runner.ROOT, 'worktree', 'remove', str(worktree))
        if shared_cache is None:
            shutil.rmtree(cache)
        result.update(status='reproduced', paired_evidence=check, cleanup={'worktree':'removed','cache':'shared' if shared_cache is not None else 'removed'})
    except (runner.PlanError, OSError, subprocess.SubprocessError, ValueError) as exc:
        result.update(status='failed', error=str(exc), cleanup={'worktree':'retained_if_present','cache':'shared' if shared_cache is not None else 'retained_if_present'})
    runner.write_json(output / 'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive');p.add_argument('output');p.add_argument('--timeout',type=int,default=600)
    args = p.parse_args()
    try:
        result = replay(args.archive,args.output,args.timeout)
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0 if result['status'] == 'reproduced' else 1
    except (runner.PlanError,OSError,subprocess.SubprocessError,ValueError,KeyError) as exc:
        print(f'replay error: {exc}',file=sys.stderr);return 1


if __name__ == '__main__':
    raise SystemExit(main())

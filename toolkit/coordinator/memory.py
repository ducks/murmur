"""Index campaign outcomes and versioned research findings without an external service.

Campaign state remains the source of truth; indexing does not mutate experiments.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
from . import runner

DEFAULT_RUNS = runner.ROOT / 'toolkit/runs'
DEFAULT_FINDINGS = runner.ROOT / 'toolkit/research-memory/findings.json'


def load_findings(path=DEFAULT_FINDINGS):
    data = runner.read_json(path)
    runner.exact_keys(data, {'schema_version', 'findings'}, 'research findings')
    if data['schema_version'] != 1 or not isinstance(data['findings'], list):
        raise runner.PlanError('invalid research findings')
    ids = set()
    for item in data['findings']:
        runner.exact_keys(item, {'id', 'question', 'observation', 'hypothesis', 'next_test', 'limitations', 'evidence', 'related_findings', 'candidate_commits'}, 'research finding')
        for key in ('id', 'question', 'observation', 'hypothesis', 'next_test', 'limitations'):
            runner.text_field(item[key], key)
        if item['id'] in ids:
            raise runner.PlanError(f"duplicate research finding: {item['id']}")
        ids.add(item['id'])
        if not isinstance(item['evidence'], list) or not item['evidence']:
            raise runner.PlanError('research finding must link to evidence')
        if not isinstance(item['related_findings'], list) or not all(isinstance(x, str) for x in item['related_findings']):
            raise runner.PlanError('related findings must be a list of identifiers')
        if not isinstance(item['candidate_commits'], list) or not all(isinstance(x, str) and re.fullmatch(r'[0-9a-f]{40}', x) for x in item['candidate_commits']):
            raise runner.PlanError('candidate commits must be full Git object IDs')
        for value in item['evidence']:
            runner.text_field(value, 'evidence')
            p = Path(value)
            if p.is_absolute() or '..' in p.parts or not (runner.ROOT / p).resolve().is_relative_to(runner.ROOT):
                raise runner.PlanError(f'finding evidence must be a repository-relative path: {value}')
            if not (runner.ROOT / p).is_file():
                raise runner.PlanError(f'finding evidence is missing: {value}')
    for item in data['findings']:
        if any(x not in ids for x in item['related_findings']):
            raise runner.PlanError(f"unknown related finding in {item['id']}")
    return data['findings']


def checks(report):
    if not isinstance(report, dict):
        return None
    flags = report.get('checks')
    comparisons = report.get('comparisons')
    failed = []
    if isinstance(flags, dict):
        for name, passed in sorted(flags.items()):
            if passed is False:
                item = {'check': name}
                detail = comparisons.get(name) if isinstance(comparisons, dict) else None
                if isinstance(detail, dict):
                    item['mean_paired_difference'] = detail.get('mean_paired_difference')
                failed.append(item)
    return {'outcome': report.get('outcome'), 'failed_checks': failed,
            'seeds': report.get('seeds'), 'limitations': report.get('limitations', [])}


def campaign_records(directory):
    root = Path(directory).resolve()
    config = runner.read_json(root / 'campaign.json')
    state = runner.read_json(root / 'state.json')
    if config.get('schema_version') != 1 or state.get('schema_version') != 1 or not isinstance(config.get('campaign_id'), str) or not isinstance(state.get('completed'), list):
        raise runner.PlanError(f'invalid campaign metadata: {root}')
    records = []
    completed_indices = set()
    items = [(item, False) for item in state['completed']]
    for item in state['completed']:
        if not isinstance(item, dict) or not isinstance(item.get('index'), int) or item['index'] < 1:
            raise runner.PlanError(f'invalid completed round: {root}')
        if item['index'] in completed_indices:
            raise runner.PlanError(f'duplicate completed round: {root}')
        completed_indices.add(item['index'])
    for path in sorted(root.glob('idea-*/result.json')):
        item = runner.read_json(path)
        if not isinstance(item, dict) or not isinstance(item.get('index'), int) or item['index'] < 1 or path.parent.name != f"idea-{item['index']:03d}":
            raise runner.PlanError(f'invalid round result: {path}')
        if item['index'] not in completed_indices:
            items.append((item, True))
    for item, incomplete in items:
        round_dir = root / f"idea-{item['index']:03d}"
        idea = item.get('idea')
        # Failed agents may have written an idea before a later test failed.
        if idea is None and (round_dir / 'idea.json').is_file():
            idea = runner.read_json(round_dir / 'idea.json')
        followups = item.get('followups')
        recovered = round_dir / 'followup-records.json'
        if isinstance(followups, dict) and not followups.get('full_records_path') and recovered.is_file():
            full = runner.read_json(recovered)
            if not isinstance(full, dict) or not isinstance(full.get('records'), list):
                raise runner.PlanError(f'invalid recovered follow-up records: {recovered}')
            if not isinstance(full.get('aggregates', []), list):
                raise runner.PlanError(f'invalid recovered follow-up aggregates: {recovered}')
            if not isinstance(full.get('warnings', []), list):
                raise runner.PlanError(f'invalid recovered follow-up warnings: {recovered}')
            followups = {**followups, 'records': full['records'][:10], 'record_count': len(full['records']),
                         'records_shown': min(10, len(full['records'])), 'full_records_path': str(recovered),
                         'aggregates': full.get('aggregates', [])[:50], 'aggregate_count': len(full.get('aggregates', [])),
                         'original_warnings': followups.get('warnings', []), 'warnings': full.get('warnings', []),
                         'recovered_from': 'tests.log after the original round'}
        records.append({'id': f"{config['campaign_id']}:{item['index']:03d}",
                        'campaign_id': config['campaign_id'], 'round': item['index'],
                        'campaign_goal': config.get('goal'), 'protocol': config.get('protocol', 'legacy'),
                        'status': item.get('status'), 'incomplete': incomplete, 'candidate_commit': item.get('commit'),
                        'seed_candidate_commit': config.get('seed_candidate_commit'),
                        'idea': idea, 'milestone': item.get('milestone'), 'milestone_evidence': item.get('milestone_evidence'), 'milestone_review': item.get('milestone_review'), 'paired_evidence': item.get('paired_evidence'), 'seed_allocation': item.get('seed_allocation'), 'replay': item.get('replay'), 'provenance': item.get('provenance'), 'followups': followups, 'development': checks(item.get('development')),
                        'evaluation': checks(item.get('evaluation')), 'diagnosis': item.get('diagnosis'), 'error': item.get('error'), 'recovery_eligible': item.get('recovery_eligible'), 'recovery_scheduled': item.get('recovery_scheduled'),
                        'evidence_directory': str(round_dir),
                        'verification': 'indexed from campaign state; evidence hashes were not rechecked'})
    return records


def build_index(runs_root=DEFAULT_RUNS, findings_path=DEFAULT_FINDINGS):
    root = Path(runs_root).resolve()
    findings = load_findings(findings_path)
    records, warnings, seen = [], [], set()
    def order(path):
        state = path.parent / 'state.json'
        try:
            modified = state.stat().st_mtime_ns
        except OSError:
            modified = 0
        return (modified, str(path))
    for config_path in sorted(root.glob('*/campaign.json'), key=order):
        try:
            for record in campaign_records(config_path.parent):
                if record['id'] in seen:
                    raise runner.PlanError(f"duplicate campaign round id: {record['id']}")
                seen.add(record['id'])
                records.append(record)
        except (runner.PlanError, OSError, ValueError) as exc:
            warnings.append({'path': str(config_path.parent), 'error': str(exc)})
    return {'schema_version': 1, 'runs_root': str(root), 'findings': findings,
            'campaign_rounds': records, 'warnings': warnings,
            'limitations': ['Campaign evidence is indexed, not independently verified.',
                            'Only campaigns directly inside the runs root are discovered.',
                            'Campaign artifacts are local and may be ignored by Git; back them up.',
                            'Findings are curated notes, not automatically proven causal explanations.']}


def snapshot_for_agent(index, current_campaign, limit=20):
    records = [x for x in index['campaign_rounds'] if x['campaign_id'] != current_campaign]
    return {'schema_version': 1, 'findings': index['findings'], 'campaign_rounds': records[-limit:],
            'campaign_round_count': len(records), 'campaign_rounds_shown': min(len(records), limit),
            'warnings': index['warnings'], 'limitations': index['limitations']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', default=str(DEFAULT_RUNS))
    parser.add_argument('--output', help='save a new JSON snapshot; never overwrites an existing file')
    args = parser.parse_args(argv)
    try:
        index = build_index(args.runs_root)
        content = json.dumps(index, indent=2, sort_keys=True) + '\n'
        if args.output:
            with Path(args.output).open('x', encoding='utf-8') as f:
                f.write(content)
            print(f'Saved {len(index["campaign_rounds"])} rounds and {len(index["findings"])} findings to {args.output}')
        else:
            print(content, end='')
    except (runner.PlanError, OSError, ValueError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

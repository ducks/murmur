"""Freeze, execute and inspect bounded research plans without network services."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
ADAPTER = 'unlabeled_context'
MODELS = {'single', 'mixture', 'known_filter', 'oracle'}
OFFSETS = (0, 40000, 46000, 48000, 51000)
MASK64 = (1 << 64) - 1


class PlanError(Exception):
    pass


def object_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PlanError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=object_no_duplicates,
                          parse_constant=lambda x: (_ for _ in ()).throw(PlanError(f'invalid number: {x}')))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlanError(f'cannot read JSON {path}: {exc}') from exc


def exact_keys(obj, expected, name):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        raise PlanError(f'{name} must contain exactly: {", ".join(sorted(expected))}')


def text_field(value, name):
    if not isinstance(value, str) or not value.strip() or '\n' in value or '\r' in value:
        raise PlanError(f'{name} must be nonempty single-line text')


def finite_number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise PlanError(f'{name} must be a finite number in [{low}, {high}]')


def config_parse(text):
    fields = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            raise PlanError('invalid config line')
        key, value = (part.strip() for part in line.split('=', 1))
        if key in fields:
            raise PlanError(f'duplicate config key {key}')
        fields[key] = value
    exact_keys(fields, {'schema_version', 'seeds', 'delays', 'train_tokens', 'test_tokens', 'units', 'degree', 'learning_rate'}, 'config')
    try:
        if fields['schema_version'] != '1':
            raise PlanError('unsupported config schema')
        seeds = [int(x.strip()) for x in fields['seeds'].split(',')]
        delays = [int(x.strip()) for x in fields['delays'].split(',')]
        values = {key: int(fields[key]) for key in ('train_tokens', 'test_tokens', 'units', 'degree')}
        rate = float(fields['learning_rate'])
    except ValueError as exc:
        raise PlanError('invalid config number') from exc
    if not seeds or len(set(seeds)) != len(seeds) or any(x < 0 or x > MASK64 for x in seeds):
        raise PlanError('config seeds must be unique unsigned 64-bit integers')
    if delays != [1]:
        raise PlanError('this adapter requires delays=1')
    if any(v <= 0 for v in values.values()) or values['train_tokens'] <= 1 or values['test_tokens'] <= 1:
        raise PlanError('config dimensions must be positive and streams longer than delay')
    finite_number(rate, 'learning_rate', 1e-12, 1.0)
    role_seeds = [(s + offset) & MASK64 for s in seeds for offset in OFFSETS]
    if len(role_seeds) != len(set(role_seeds)):
        raise PlanError('seed roles overlap within a split')
    return {'seeds': seeds, 'role_seeds': set(role_seeds), 'delays': delays, **values, 'learning_rate': rate}


def config_path(value):
    text_field(value, 'config path')
    path = Path(value)
    if path.is_absolute() or '..' in path.parts:
        raise PlanError('config paths must be relative to the repository and cannot contain ..')
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise PlanError('config path resolves outside the repository')
    try:
        return resolved.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        raise PlanError(f'cannot read config {value}: {exc}') from exc


def validate_plan(plan, dev_text, evaluation_text):
    exact_keys(plan, {'schema_version', 'id', 'question', 'hypothesis', 'falsification', 'adapter', 'development_config', 'evaluation_config', 'comparison', 'criteria', 'budget'}, 'plan')
    if plan['schema_version'] != 1 or isinstance(plan['schema_version'], bool):
        raise PlanError('unsupported plan schema')
    for key in ('id', 'question', 'hypothesis', 'falsification'):
        text_field(plan[key], key)
    if plan['adapter'] != ADAPTER:
        raise PlanError(f'unsupported adapter; registered: {ADAPTER}')
    comparison = plan['comparison']
    exact_keys(comparison, {'candidate', 'baseline', 'required_models', 'metric', 'regime', 'noise', 'checkpoint', 'probe_rule'}, 'comparison')
    if comparison['candidate'] not in MODELS or comparison['baseline'] not in MODELS or comparison['candidate'] == comparison['baseline']:
        raise PlanError('candidate and baseline must be distinct registered models')
    models = comparison['required_models']
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models) or len(models) != len(set(models)) or set(models) != MODELS:
        raise PlanError('required_models must list all four adapter models exactly once')
    if comparison['metric'] != 'accuracy' or comparison['regime'] not in ('aba', 'stationary', 'switching') or comparison['noise'] not in ('0.1', '0.25') or comparison['checkpoint'] not in ('1', '2', '3') or comparison['probe_rule'] not in ('0', '1'):
        raise PlanError('unsupported comparison filter')
    criteria = plan['criteria']
    exact_keys(criteria, {'min_mean_difference', 'min_paired_win_fraction', 'max_stationary_accuracy_drop'}, 'criteria')
    finite_number(criteria['min_mean_difference'], 'min_mean_difference', -1.0, 1.0)
    finite_number(criteria['min_paired_win_fraction'], 'min_paired_win_fraction', 0.0, 1.0)
    finite_number(criteria['max_stationary_accuracy_drop'], 'max_stationary_accuracy_drop', 0.0, 1.0)
    budget = plan['budget']
    exact_keys(budget, {'max_seeds_per_split', 'max_logical_transitions_per_split', 'max_wall_seconds'}, 'budget')
    for key in budget:
        if isinstance(budget[key], bool) or not isinstance(budget[key], int) or budget[key] <= 0:
            raise PlanError(f'{key} must be a positive integer')
    dev, evaluation = config_parse(dev_text), config_parse(evaluation_text)
    if dev['role_seeds'] & evaluation['role_seeds']:
        raise PlanError('development and evaluation seed roles overlap')
    comparable = ('delays', 'train_tokens', 'test_tokens', 'units', 'degree', 'learning_rate')
    if any(dev[k] != evaluation[k] for k in comparable):
        raise PlanError('development and evaluation configs must differ only in seeds')
    for name, cfg in (('development', dev), ('evaluation', evaluation)):
        # Six regime/noise conditions, each with three training phases and six
        # frozen probes. This counts stream transitions once, not model steps.
        logical = len(cfg['seeds']) * 6 * (3 * cfg['train_tokens'] + 6 * cfg['test_tokens'])
        if len(cfg['seeds']) > budget['max_seeds_per_split'] or logical > budget['max_logical_transitions_per_split']:
            raise PlanError(f'{name} exceeds declared logical-transition or seed budget')
    return dev, evaluation


def command(*args, check=True):
    try:
        return subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PlanError(f'command failed: {args[0]}: {exc}') from exc


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_fingerprint():
    """Hash HEAD and every tracked or non-ignored untracked file, including edits."""
    head = command('git', 'rev-parse', 'HEAD').stdout.strip().decode('ascii')
    names = command('git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard').stdout.split(b'\0')
    hasher = hashlib.sha256()
    hasher.update(head.encode('ascii') + b'\0')
    for name in sorted(set(n for n in names if n)):
        path = ROOT / os.fsdecode(name)
        hasher.update(len(name).to_bytes(8, 'big') + name)
        if path.is_symlink():
            data = b'SYMLINK\0' + os.fsencode(os.readlink(path))
        elif path.is_file():
            data = b'FILE\0' + path.read_bytes()
        else:
            data = b'MISSING\0'
        mode = (path.stat().st_mode & 0o111) if path.is_file() and not path.is_symlink() else 0
        hasher.update(mode.to_bytes(2, 'big') + len(data).to_bytes(8, 'big') + data)
    return head, hasher.hexdigest()


def output_location(path):
    """Avoid causing a self-changing source fingerprint through run artifacts."""
    dest = Path(path).resolve()
    if dest.is_relative_to(ROOT):
        rel = dest.relative_to(ROOT)
        result = command('git', 'check-ignore', '-q', '--', str(rel), check=False)
        load_lock(directory)  # Fail closed if source or frozen plan changed during execution.
        if result.returncode != 0:
            raise PlanError('run directory inside repository must be git-ignored (e.g. toolkit/runs/...)')
    return dest


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def prepare(manifest_path, destination):
    plan = read_json(manifest_path)
    if not isinstance(plan, dict):
        raise PlanError('plan must be an object')
    dev_text = config_path(plan.get('development_config'))
    evaluation_text = config_path(plan.get('evaluation_config'))
    validate_plan(plan, dev_text, evaluation_text)
    destination = output_location(destination)
    head, fingerprint = source_fingerprint()
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PlanError(f'cannot create plan directory: {exc}') from exc
    try:
        write_json(destination / 'plan.json', plan)
        (destination / 'development.cfg').write_text(dev_text, encoding='utf-8')
        (destination / 'evaluation.cfg').write_text(evaluation_text, encoding='utf-8')
        files = {name: digest((destination / name).read_bytes()) for name in ('plan.json', 'development.cfg', 'evaluation.cfg')}
        version = lambda *args: command(*args, check=False).stdout.decode('utf-8', errors='replace').strip()
        write_json(destination / 'lock.json', {'schema_version': 1, 'head': head, 'source_sha256': fingerprint,
                   'files': files, 'git_status': command('git', 'status', '--porcelain=v1', '--untracked-files=all').stdout.decode('utf-8', errors='replace'), 'rustc': version('rustc', '--version'), 'cargo': version('cargo', '--version'),
                   'python': sys.version.split()[0], 'nix_shell': os.environ.get('IN_NIX_SHELL', 'not recorded'),
                   'nix_path': os.environ.get('NIX_PATH', 'not recorded'),
                   'nixpkgs_revision': 'unrecorded; shell.nix is not pinned',
                   'limitations': 'Local checksums are editable; seed files are visible and holdout secrecy is not enforced.'})
    except (OSError, PlanError) as exc:
        raise PlanError(f'could not finish preparation; inspect {destination}: {exc}') from exc
    return destination


def load_lock(directory, verify_source=True):
    directory = Path(directory).resolve()
    lock = read_json(directory / 'lock.json')
    if not isinstance(lock, dict) or lock.get('schema_version') != 1 or not isinstance(lock.get('files'), dict) or set(lock['files']) != {'plan.json', 'development.cfg', 'evaluation.cfg'}:
        raise PlanError('invalid lock file')
    for name, expected in lock['files'].items():
        try:
            actual = digest((directory / name).read_bytes())
        except OSError as exc:
            raise PlanError(f'missing frozen file {name}') from exc
        if actual != expected:
            raise PlanError(f'frozen file changed: {name}')
    plan = read_json(directory / 'plan.json')
    dev = (directory / 'development.cfg').read_text(encoding='utf-8')
    evaluation = (directory / 'evaluation.cfg').read_text(encoding='utf-8')
    configs = validate_plan(plan, dev, evaluation)
    if verify_source:
        head, current = source_fingerprint()
        if head != lock.get('head') or current != lock.get('source_sha256'):
            raise PlanError('repository source differs from frozen plan; prepare a new plan')
    return plan, dict(zip(('development', 'evaluation'), configs))


def load_csv(path, expected_columns):
    try:
        with Path(path).open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != expected_columns:
                raise PlanError(f'unexpected columns in {path}')
            rows = list(reader)
    except OSError as exc:
        raise PlanError(f'cannot read evidence {path}: {exc}') from exc
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise PlanError(f'malformed CSV rows in {path}')
    return rows


def parse_counts(row, location):
    try:
        correct, trials = int(row['correct']), int(row['trials'])
        accuracy = float(row['accuracy'])
    except (ValueError, TypeError, KeyError) as exc:
        raise PlanError(f'invalid counts in {location}') from exc
    if trials <= 0 or correct < 0 or correct > trials or not math.isfinite(accuracy) or abs(accuracy - correct / trials) > 1e-9:
        raise PlanError(f'inconsistent accuracy in {location}')
    return correct, trials


def analyze(plan, cfg, evidence):
    probe_fields = ['seed', 'regime', 'noise', 'checkpoint', 'rule', 'model', 'correct', 'trials', 'accuracy', 'mean_logloss']
    window_fields = ['seed', 'regime', 'noise', 'window_start', 'window_end', 'model', 'correct', 'trials', 'accuracy', 'mean_logloss']
    probe_rows = load_csv(evidence / 'probes.csv', probe_fields)
    window_rows = load_csv(evidence / 'windows.csv', window_fields)
    seeds = {str(s) for s in cfg['seeds']}
    lookup = {}
    for row in probe_rows:
        key = tuple(row[k] for k in ('seed', 'regime', 'noise', 'checkpoint', 'rule', 'model'))
        seed, regime, noise, checkpoint, rule, model = key
        if seed not in seeds or regime not in ('aba', 'stationary', 'switching') or noise not in ('0.1', '0.25') or checkpoint not in ('1', '2', '3') or rule not in ('0', '1') or model not in MODELS or key in lookup:
            raise PlanError(f'unexpected or duplicate probe: {key}')
        correct, trials = parse_counts(row, f'probe {key}')
        if trials != cfg['test_tokens']:
            raise PlanError(f'wrong probe trial count: {key}')
        try:
            logloss = float(row['mean_logloss'])
        except ValueError as exc:
            raise PlanError(f'invalid probe log loss: {key}') from exc
        if not math.isfinite(logloss) or logloss < 0:
            raise PlanError(f'invalid probe log loss: {key}')
        lookup[key] = correct / trials
    if len(lookup) != len(seeds) * 3 * 2 * 3 * 2 * len(MODELS):
        raise PlanError('probe evidence is incomplete')
    windows = {}
    total_steps = cfg['train_tokens'] * 3
    expected_starts = set(range(0, total_steps, 200))
    for row in window_rows:
        try:
            start, end = int(row['window_start']), int(row['window_end'])
        except ValueError as exc:
            raise PlanError('invalid window boundaries') from exc
        key = (row['seed'], row['regime'], row['noise'], str(start), row['model'])
        if row['seed'] not in seeds or row['regime'] not in ('aba', 'stationary', 'switching') or row['noise'] not in ('0.1', '0.25') or row['model'] not in MODELS or start not in expected_starts or end != min(start + 200, total_steps) or key in windows:
            raise PlanError(f'unexpected or duplicate window: {key}')
        correct, trials = parse_counts(row, f'window {key}')
        if trials != end - start:
            raise PlanError(f'wrong window trial count: {key}')
        try:
            logloss = float(row['mean_logloss'])
        except ValueError as exc:
            raise PlanError(f'invalid window log loss: {key}') from exc
        if not math.isfinite(logloss) or logloss < 0:
            raise PlanError(f'invalid window log loss: {key}')
        windows[key] = (correct, trials)
    if len(windows) != len(seeds) * 3 * 2 * len(expected_starts) * len(MODELS):
        raise PlanError('window evidence is incomplete')
    cmp = plan['comparison']
    diffs = []
    for seed in sorted(seeds, key=int):
        prefix = (seed, cmp['regime'], cmp['noise'], cmp['checkpoint'], cmp['probe_rule'])
        diffs.append(lookup[prefix + (cmp['candidate'],)] - lookup[prefix + (cmp['baseline'],)])
    mean = sum(diffs) / len(diffs)
    win_fraction = sum(x > 0 for x in diffs) / len(diffs)
    stationary = {}
    for noise in ('0.1', '0.25'):
        def stationary_accuracy(model):
            parts = [v for (s, regime, n, _, m), v in windows.items() if s in seeds and regime == 'stationary' and n == noise and m == model]
            return sum(x[0] for x in parts) / sum(x[1] for x in parts)
        stationary[noise] = stationary_accuracy(cmp['candidate']) - stationary_accuracy(cmp['baseline'])
    criteria = plan['criteria']
    checks = {'mean_difference': mean >= criteria['min_mean_difference'],
              'paired_win_fraction': win_fraction >= criteria['min_paired_win_fraction'],
              'stationary_accuracy': all(v >= -criteria['max_stationary_accuracy_drop'] for v in stationary.values())}
    return {'outcome': 'criteria_met' if all(checks.values()) else 'criteria_not_met',
            'checks': checks, 'candidate': cmp['candidate'], 'baseline': cmp['baseline'],
            'filter': {k: cmp[k] for k in ('regime', 'noise', 'checkpoint', 'probe_rule', 'metric')},
            'seeds': len(diffs), 'mean_paired_difference': mean, 'paired_win_fraction': win_fraction,
            'wins': sum(x > 0 for x in diffs), 'losses': sum(x < 0 for x in diffs),
            'ties': sum(x == 0 for x in diffs), 'stationary_online_difference': stationary,
            'limitations': ['Thresholds are operational criteria, not a statistical guarantee.',
                            'Oracle and known-rule controls have extra information.',
                            'Evaluation seeds are visible locally; holdout secrecy is not enforced.',
                            'The adapter does not measure peak memory or resource-matched compute.']}


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def stage_hashes(stage):
    return {str(path.relative_to(stage)): file_digest(path)
            for path in sorted(stage.rglob('*')) if path.is_file() and path.name != 'status.json'}


def stage_state(directory, split):
    stage = directory / split
    if not stage.exists():
        return 'pending'
    status = read_json(stage / 'status.json')
    if not isinstance(status, dict) or status.get('state') not in ('running', 'complete', 'failed'):
        raise PlanError(f'invalid {split} status')
    if status['state'] == 'complete':
        if not isinstance(status.get('artifacts_sha256'), dict) or stage_hashes(stage) != status['artifacts_sha256']:
            raise PlanError(f'{split} evidence changed after completion')
        report = read_json(stage / 'report.json')
        if report.get('outcome') != status.get('outcome'):
            raise PlanError(f'{split} status and report disagree')
    return status['state']


def run_stage(directory, split):
    directory = Path(directory).resolve()
    plan, configs = load_lock(directory)
    if split not in ('development', 'evaluation'):
        raise PlanError('split must be development or evaluation')
    if split == 'evaluation':
        if stage_state(directory, 'development') != 'complete':
            raise PlanError('complete development before evaluation')
        dev_report = read_json(directory / 'development' / 'report.json')
        if dev_report.get('outcome') != 'criteria_met':
            raise PlanError('development criteria were not met; prepare a revised plan rather than consuming evaluation')
    stage = directory / split
    try:
        stage.mkdir(exist_ok=False)
    except OSError as exc:
        raise PlanError(f'cannot create {split} stage (existing evidence is never overwritten): {exc}') from exc
    started = time.monotonic()
    status = {'state': 'running', 'split': split}
    write_json(stage / 'status.json', status)
    try:
        args = ['cargo', 'run', '--offline', '--quiet', '--release', '--manifest-path', str(ROOT / 'toolkit/Cargo.toml'), '--bin', ADAPTER, '--', str(directory / f'{split}.cfg'), str(stage / 'evidence')]
        with (stage / 'stdout.log').open('wb') as stdout, (stage / 'stderr.log').open('wb') as stderr:
            result = subprocess.run(args, cwd=ROOT, stdout=stdout, stderr=stderr, timeout=plan['budget']['max_wall_seconds'], check=False)
        load_lock(directory)  # Fail closed if source or frozen plan changed during execution.
        if result.returncode != 0:
            raise PlanError(f'adapter failed with exit code {result.returncode}; inspect stderr.log')
        if (stage / 'evidence' / 'status.txt').read_text(encoding='utf-8').strip() != 'complete':
            raise PlanError('adapter did not report complete')
        report = analyze(plan, configs[split], stage / 'evidence')
        report['split'] = split
        report['elapsed_seconds'] = time.monotonic() - started
        report['logical_transitions'] = len(configs[split]['seeds']) * 6 * (3 * configs[split]['train_tokens'] + 6 * configs[split]['test_tokens'])
        report['peak_memory'] = 'not measured'
        write_json(stage / 'report.json', report)
        (stage / 'report.md').write_text(
            f"# {plan['id']}: {split}\n\nQuestion: {plan['question']}\n\nHypothesis: {plan['hypothesis']}\n\n"
            f"Declared falsification: {plan['falsification']}\n\nOutcome: **{report['outcome']}**\n\n"
            f"{report['candidate']} minus {report['baseline']} mean paired accuracy: {report['mean_paired_difference']:.4f}; "
            f"wins/losses/ties: {report['wins']}/{report['losses']}/{report['ties']} across {report['seeds']} seeds.\n\n"
            f"Stationary online accuracy differences: {report['stationary_online_difference']}.\n\n"
            f"Checks: {report['checks']}.\n\n"
            "The oracle and known-rule controls have extra information. Evaluation seeds are visible locally; "
            "these criteria are not a statistical guarantee. See report.json and evidence/ for details.\n", encoding='utf-8')
        status.update(state='complete', outcome=report['outcome'], elapsed_seconds=report['elapsed_seconds'], artifacts_sha256=stage_hashes(stage))
    except (OSError, PlanError, subprocess.TimeoutExpired) as exc:
        status.update(state='failed', error=str(exc), elapsed_seconds=time.monotonic() - started)
        write_json(stage / 'status.json', status)
        raise PlanError(f'{split} failed: {exc}') from exc
    write_json(stage / 'status.json', status)
    return report


def next_action(directory):
    directory = Path(directory).resolve()
    load_lock(directory)
    dev = stage_state(directory, 'development')
    evaluation = stage_state(directory, 'evaluation')
    if dev == 'pending':
        return {'action': 'run_development', 'reason': 'The frozen plan has not been run.'}
    if dev != 'complete':
        return {'action': 'inspect_development', 'reason': f'Development status is {dev}; inspect the saved logs.'}
    outcome = read_json(directory / 'development' / 'report.json').get('outcome')
    if outcome != 'criteria_met':
        return {'action': 'revise_plan', 'reason': 'Development criteria were not met; record the negative result and prepare a new plan.'}
    if evaluation == 'pending':
        return {'action': 'run_evaluation', 'reason': 'Development criteria were met; evaluate the frozen plan on its separate seeds.'}
    if evaluation != 'complete':
        return {'action': 'inspect_evaluation', 'reason': f'Evaluation status is {evaluation}; inspect the saved logs.'}
    result = read_json(directory / 'evaluation' / 'report.json').get('outcome')
    return {'action': 'review_results', 'reason': f'Both stages finished; evaluation outcome: {result}.'}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='freeze a plan and two disjoint configurations')
    p.add_argument('manifest'); p.add_argument('directory')
    p = sub.add_parser('run', help='run one stage of the frozen plan')
    p.add_argument('directory'); p.add_argument('split', choices=('development', 'evaluation'))
    p = sub.add_parser('next', help='show the next bounded workflow action')
    p.add_argument('directory')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            print(f'Prepared: {prepare(args.manifest, args.directory)}')
        elif args.command == 'run':
            print(json.dumps(run_stage(args.directory, args.split), indent=2, sort_keys=True))
        else:
            print(json.dumps(next_action(args.directory), indent=2, sort_keys=True))
    except (PlanError, OSError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

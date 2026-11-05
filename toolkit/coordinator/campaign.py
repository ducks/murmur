"""Run bounded, agent-authored candidate studies in isolated Git worktrees.

The optional coding-agent backend is Codex CLI. Model generation may use its
configured remote service or an explicitly selected local OSS provider. The
candidate model and experiment runners use no model APIs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import signal
from pathlib import Path
import subprocess
import sys
import time
import uuid

from . import memory, runner

GOAL = 'Find a model that retains earlier rules and adapts to rule changes at least as well as the mixture baseline.'
ALLOWED = {'toolkit/src/agent_candidate.rs', 'toolkit/tests/agent_generated.rs'}
GENERATED = {'toolkit/configs/agent/dev.cfg', 'toolkit/configs/agent/evaluation.cfg', 'toolkit/plans/agent-candidate.json'}
SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {key: {'type': 'string', 'minLength': 1, 'maxLength': 1000}
                   for key in ('question', 'hypothesis', 'falsification', 'approach', 'limitations')},
    'required': ['question', 'hypothesis', 'falsification', 'approach', 'limitations'],
}
DIAGNOSIS_FIELDS = ('observation', 'hypothesis', 'alternative', 'followup_test', 'expected_evidence', 'limitations')
DIAGNOSIS_SCHEMA = {'type': 'object', 'additionalProperties': False,
                    'properties': {key: {'type': 'string', 'minLength': 1, 'maxLength': 1000} for key in DIAGNOSIS_FIELDS},
                    'required': list(DIAGNOSIS_FIELDS)}


def git(cwd, *args):
    try:
        result = subprocess.run(['git', *args], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise runner.PlanError(f'git operation failed: {args[0]}: {exc}') from exc
    return result.stdout.decode('utf-8', errors='strict').strip()


def seed_configs(index, seed_base=7_000_001):
    # All role offsets fit within 51,000; groups and rounds are 100,000+ apart.
    start = seed_base + index * 300_000
    def content(first, count):
        return ('schema_version=1\nseeds=' + ','.join(str(i) for i in range(first, first + count)) +
                '\ndelays=1\ntrain_tokens=1000\ntest_tokens=1000\nunits=128\ndegree=4\nlearning_rate=0.05\n')
    return content(start, 10), content(start + 100_000, 20)


def make_plan(index, idea, protocol='legacy'):
    plan = {
        'schema_version': 1, 'id': f'agent-idea-{index:03d}',
        'question': idea['question'], 'hypothesis': idea['hypothesis'], 'falsification': idea['falsification'],
        'adapter': 'candidate_context', 'development_config': 'toolkit/configs/agent/dev.cfg',
        'evaluation_config': 'toolkit/configs/agent/evaluation.cfg',
        'comparison': {'candidate': 'candidate', 'baseline': 'mixture',
                       'required_models': ['single', 'mixture', 'candidate', 'known_filter', 'oracle'],
                       'metric': 'accuracy', 'regime': 'aba', 'noise': '0.25', 'checkpoint': '3', 'probe_rule': '1'},
        'criteria': {'min_mean_difference': 0.05, 'min_paired_win_fraction': 0.6, 'max_stationary_accuracy_drop': 0.02, 'max_other_probe_accuracy_drop': 0.03},
        'budget': {'max_seeds_per_split': 20, 'max_logical_transitions_per_split': 1_200_000, 'max_wall_seconds': 180},
    }
    if protocol == 'milestone':
        plan.update(adapter='milestone_context',
                    comparison={'candidate': 'candidate', 'baseline': 'mixture',
                                'required_models': ['single', 'mixture', 'candidate', 'oracle'], 'metric': 'accuracy'},
                    criteria={'max_accuracy_drop': 0.02})
    elif protocol != 'legacy':
        raise runner.PlanError('unknown campaign protocol')
    return plan


def init(directory, goal, max_ideas, agent_seconds, test_seconds, local_provider=None, model=None, prior_campaign=None, protocol='milestone', diagnostic_seconds=120, diagnose_failures=True, max_reported_input_tokens=None):
    runner.text_field(goal, 'goal')
    if not (1 <= max_ideas <= 100 and 10 <= agent_seconds <= 7200 and 10 <= test_seconds <= 7200):
        raise runner.PlanError('limits outside supported ranges')
    if not 10 <= diagnostic_seconds <= 7200:
        raise runner.PlanError('diagnostic time outside supported range')
    if max_reported_input_tokens is not None and (isinstance(max_reported_input_tokens, bool) or not isinstance(max_reported_input_tokens, int) or max_reported_input_tokens < 1):
        raise runner.PlanError('reported input-token budget must be positive')
    if local_provider not in (None, 'ollama', 'lmstudio'):
        raise runner.PlanError('unsupported local provider')
    if protocol not in ('milestone', 'legacy'):
        raise runner.PlanError('unsupported campaign protocol')
    if model is not None:
        runner.text_field(model, 'model')
    if git(runner.ROOT, 'status', '--porcelain=v1', '--untracked-files=all'):
        raise runner.PlanError('commit or otherwise clean the starting worktree before creating a campaign')
    prior = prior_history(prior_campaign) if prior_campaign else []
    dest = runner.output_location(directory)
    try:
        dest.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise runner.PlanError(f'cannot create campaign directory: {exc}') from exc
    campaign_id = uuid.uuid4().hex[:12]
    seed_base = 10_000_000 + (int(campaign_id, 16) % 1_000_000_000_000) * 1_000_000
    config = {'schema_version': 1, 'campaign_id': campaign_id, 'seed_base': seed_base, 'goal': goal,
              'max_ideas': max_ideas, 'agent_seconds': agent_seconds, 'test_seconds': test_seconds,
              'local_provider': local_provider, 'model': model,
              'base_commit': git(runner.ROOT, 'rev-parse', 'HEAD'), 'prior_history': prior,
              'protocol': protocol,
              'diagnose_failures': diagnose_failures, 'diagnostic_seconds': diagnostic_seconds,
              'max_reported_input_tokens': max_reported_input_tokens,
              'scope': 'binary hidden-context sequence task; candidate Rust model only',
              'limits': 'LLM token/cost limits and blind holdouts are not enforced.'}
    runner.write_json(dest / 'campaign.json', config)
    runner.write_json(dest / 'state.json', {'schema_version': 1, 'completed': [], 'next_index': 1, 'base_commit': config['base_commit'], 'status': 'ready'})
    runner.write_json(dest / 'agent-output-schema.json', SCHEMA)
    runner.write_json(dest / 'diagnosis-output-schema.json', DIAGNOSIS_SCHEMA)
    return config


def probe_matrix(path):
    """Summarize all final frozen probes so narrow wins don't hide regressions."""
    if not Path(path).is_file():
        return {}
    groups = {}
    with Path(path).open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['checkpoint'] != '3':
                continue
            key = f"{row['regime']}:{row['noise']}:rule_{row['rule']}"
            groups.setdefault(key, {}).setdefault(row['model'], []).append(float(row['accuracy']))
    return {key: {name: sum(values) / len(values) for name, values in models.items()}
            for key, models in groups.items()}


def prior_history(path):
    """Import previous outcomes; later seeds remain fresh but no longer blind."""
    root = Path(path).resolve()
    state = runner.read_json(root / 'state.json')
    history = []
    for item in state.get('completed', []):
        entry = dict(item)
        stage = 'evaluation' if item.get('evaluation') else 'development'
        evidence = root / f"idea-{item['index']:03d}" / 'study' / stage / 'evidence' / 'probes.csv'
        entry['final_probe_matrix'] = probe_matrix(evidence)
        entry['previous_campaign'] = str(root)
        history.append(entry)
    return history


def prompt_for(config, history, research_memory=None):
    if config.get('protocol', 'legacy') == 'milestone':
        criteria_text = 'The supervisor checks final frozen B recall and the first 100 online predictions after each A→B and B→A switch, on each of four binary task families. The candidate must be within 2 percentage points of the unchanged mixture on each mean paired accuracy. Resource budgets apply to the entire stage; inline size and per-model training time are diagnostic only and heap memory is not measured.'
        evaluator_reference = ', toolkit/src/bin/milestone_context.rs'
    else:
        criteria_text = 'It will test candidate vs the unchanged original mixture on the noisiest ABA B retention, with stationary penalties, a maximum 3 percentage-point mean regression on the other final A/B probes, and fixed criteria.'
        evaluator_reference = ''
    return f'''You are one research agent in an automated, bounded model research campaign.
Overall goal: {config['goal']}

Propose and implement ONE genuinely motivated candidate model for the existing hidden-context binary sequence task. Read RESEARCH.md, EXPERIMENTS.md, toolkit/src/unlabeled_context.rs, toolkit/src/candidate_context.rs{evaluator_reference}, toolkit/src/agent_candidate.rs and toolkit/tests/candidate_context.rs. The existing mixture and other controls are fixed; seek an improvement over the mixture, not an attractive demonstration. A copy of the baseline can pass a non-inferiority threshold but is not a novel mechanism. Consider varied ideas, not only stronger thresholds. This is one limited task, not a claim of general language modeling or novelty.

Previous campaign results (may be empty):
{json.dumps(history, indent=2)}

Research memory (observations are not proven explanations; campaign evidence is not reverified here):
{json.dumps(research_memory or {}, indent=2)}

For an apparent failure, distinguish observations from possible explanations and propose a discriminating test in your hypothesis and falsification. The full index for this round is saved alongside the agent logs as research-memory.json; the prompt may include only recent campaign rounds.
If the preceding round contains a completed diagnosis, try to design a focused test that distinguishes its competing explanations, with an explicit control. If the proposed test cannot fit the candidate and test-file scope, state that limitation in your response and select another testable idea. A diagnosis is an unverified hypothesis, not an established cause.
Prior candidate commits are available locally: when building on a previous idea, inspect its candidate source with git show <commit>:toolkit/src/agent_candidate.rs and adapt it within the permitted files. Starting a new campaign does not automatically check out the prior candidate.
For any quantitative follow-up in agent_generated.rs, print one single-line record per aggregate measurement using this exact prefix and JSON shape: MURMUR_FOLLOWUP {{"measure":"name","candidate":"name","control":"name","candidate_correct":1,"control_correct":1,"trials":2}}. Counts must refer to the same paired observations. Prefer aggregate counts by task and measure, not one record per seed; keep any per-seed details in the test log. The supervisor preserves and checks the arithmetic but does not independently verify the experimental design. Do not emit a record for a test you did not run.

Edit ONLY toolkit/src/agent_candidate.rs and toolkit/tests/agent_generated.rs. The latter must contain at least one meaningful Rust test for your proposed mechanism. Do not alter evaluator, baseline, configs, docs, Git metadata, or any other file; do not commit. The supervisor rejects other changes. Preserve Candidate's Clone, new(seed, rate), reset(), predict(x)->f64 and observe(x,y,learn) interface. `predict` must happen without seeing y; output must be finite in [0,1]. In `observe(..., false)`, do not change learned parameters, though context inference state may update. reset clears transient inference state without erasing learned parameters. Candidate may use local Rust standard library and existing local crates only; no network, pretrained weights, external APIs, hidden future data, environment data, or provided true context. Model-code restrictions are reviewed/tested, not a complete security boundary.

The supervisor independently generates separate fresh deterministic development and evaluation configs, validates all controls and runs the evaluator. {criteria_text} You must not read, generate, inspect, or tune against evaluation data. You may run targeted tests and small development-only experiments, but do not alter their declared criteria or claim statistical significance. The coding-agent service may be remote; the candidate itself must be local.

Run: cargo test --offline --manifest-path toolkit/Cargo.toml --test candidate_context --test agent_generated
Then provide a JSON object matching the supplied schema: question, hypothesis, falsification, approach, limitations. Each field must be a nonempty single-line string. Explain why the idea could fail. Do not claim results you have not measured. Do not ask for input; stop on an actual blocker.
'''


def checked_agent_output(path):
    idea = runner.read_json(path)
    runner.exact_keys(idea, SCHEMA['required'], 'agent output')
    for key, value in idea.items():
        runner.text_field(value, key)
        if len(value) > 1000:
            raise runner.PlanError(f'agent output field {key} too long')
    return idea


def agent_command(config, worktree, response, schema):
    args = ['codex', '-a', 'never', 'exec', '--sandbox', 'workspace-write', '--ephemeral', '--json',
            '--cd', str(worktree), '--add-dir', str(response.parent.parent / 'build-cache'),
            '--output-schema', str(schema), '--output-last-message', str(response)]
    if config['local_provider']:
        args += ['--oss', '--local-provider', config['local_provider']]
    if config['model']:
        args += ['--model', config['model']]
    return args + ['-']


def bounded_process(args, *, cwd, stdout, stderr, timeout, input_bytes=None, env=None):
    """Stop the process group on timeout; preserve logs for inspection."""
    proc = subprocess.Popen(args, cwd=cwd, stdin=subprocess.PIPE if input_bytes is not None else None,
                            stdout=stdout, stderr=stderr, start_new_session=True, env=env)
    try:
        proc.communicate(input=input_bytes, timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
        raise
    return proc.returncode


def call_agent(config, worktree, round_dir, history):
    env = build_environment(round_dir.parent, config)
    args = agent_command(config, worktree, round_dir / 'idea.json', round_dir.parent / 'agent-output-schema.json')
    index = runner.read_json(round_dir / 'research-memory.json')
    (round_dir / 'prompt.txt').write_text(prompt_for(config, history, index['agent_context']), encoding='utf-8')
    try:
        with (round_dir / 'agent-events.jsonl').open('wb') as stdout, (round_dir / 'agent-stderr.log').open('wb') as stderr:
            result = bounded_process(args, cwd=worktree, input_bytes=(round_dir / 'prompt.txt').read_bytes(), stdout=stdout,
                                     stderr=stderr, timeout=config['agent_seconds'], env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise runner.PlanError(f'coding agent could not complete: {exc}') from exc
    if result != 0:
        raise runner.PlanError(f'coding agent exited {result}; inspect agent-stderr.log and agent-events.jsonl')
    return checked_agent_output(round_dir / 'idea.json')


def agent_usage(path):
    """Sum usage fields reported by completed CLI turns, if available."""
    if not Path(path).is_file():
        return None
    totals = {}
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get('type') != 'turn.completed' or not isinstance(event.get('usage'), dict):
                continue
            for key, value in event['usage'].items():
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    totals[key] = totals.get(key, 0) + value
    return totals or None


def followup_metrics(path, preview_limit=10, max_records=10000):
    """Read optional, agent-reported paired counts; never treat them as verified science."""
    records, warnings = [], []
    marker = 'MURMUR_FOLLOWUP '
    required = {'measure', 'candidate', 'control', 'candidate_correct', 'control_correct', 'trials'}
    with Path(path).open(encoding='utf-8', errors='replace') as source:
        for line in source:
            if marker not in line:
                continue
            if len(records) + len(warnings) >= max_records:
                warnings.append('additional follow-up records were omitted')
                break
            try:
                raw = line.split(marker, 1)[1].strip()
                item = json.loads(raw, object_pairs_hook=runner.object_no_duplicates,
                                  parse_constant=lambda _: (_ for _ in ()).throw(ValueError('invalid number')))
                runner.exact_keys(item, required, 'follow-up record')
                for key in ('measure', 'candidate', 'control'):
                    runner.text_field(item[key], key)
                    if len(item[key]) > 200:
                        raise runner.PlanError('follow-up label too long')
                for key in ('candidate_correct', 'control_correct', 'trials'):
                    if isinstance(item[key], bool) or not isinstance(item[key], int):
                        raise runner.PlanError('follow-up counts must be integers')
                if item['trials'] < 1 or any(not 0 <= item[k] <= item['trials'] for k in ('candidate_correct', 'control_correct')):
                    raise runner.PlanError('follow-up counts are inconsistent')
                item['candidate_accuracy'] = item['candidate_correct'] / item['trials']
                item['control_accuracy'] = item['control_correct'] / item['trials']
                records.append(item)
            except (ValueError, json.JSONDecodeError, runner.PlanError, TypeError) as exc:
                warnings.append(f'invalid agent-reported follow-up: {exc}')
    groups = {}
    for item in records:
        match = re.fullmatch(r'(.+)_seed_([0-9]+)', item['measure'])
        if not match:
            continue
        key = (match.group(1), item['candidate'], item['control'])
        groups.setdefault(key, []).append((match.group(2), item))
    aggregates = []
    for (measure, candidate, control), values in sorted(groups.items()):
        if len({seed for seed, _ in values}) != len(values):
            warnings.append(f'duplicate seed in agent-reported group: {measure}; group not aggregated')
            continue
        trials = sum(item['trials'] for _, item in values)
        candidate_correct = sum(item['candidate_correct'] for _, item in values)
        control_correct = sum(item['control_correct'] for _, item in values)
        aggregates.append({'measure': measure, 'candidate': candidate, 'control': control, 'seeds': len(values),
                           'trials': trials, 'candidate_accuracy': candidate_correct / trials, 'control_accuracy': control_correct / trials})
    full_path = Path(path).parent / 'followup-records.json'
    runner.write_json(full_path, {'records': records, 'aggregates': aggregates, 'warnings': warnings})
    return {'records': records[:preview_limit], 'record_count': len(records), 'records_shown': min(len(records), preview_limit),
            'aggregates': aggregates[:50], 'aggregate_count': len(aggregates),
            'full_records_path': str(full_path), 'warnings': warnings,
            'qualification': 'Agent-authored test output; arithmetic checked, experimental design not independently verified.'}


def diagnosis_prompt(result, round_dir):
    return f'''You are a research investigator analyzing one completed, unsuccessful candidate evaluation.
Candidate hypothesis and implementation summary:
{json.dumps(result.get('idea'), indent=2)}

Development and evaluation reports (evaluation may be null):
{json.dumps({'development': result.get('development'), 'evaluation': result.get('evaluation')}, indent=2)}

Agent-reported follow-up output (not independently validated):
{json.dumps(result.get('followups'), indent=2)}

The full local evidence and logs are in {round_dir}. You may inspect candidate source and existing evidence with read-only commands. Do not edit files, run new training, change Git state, or inspect an unrun evaluation split. Distinguish what the data show from competing explanations; do not claim a mechanism is proven by correlation. Propose ONE feasible follow-up that compares an intervention with an explicit control and states what observations would distinguish the hypotheses. The next agent can edit only toolkit/src/agent_candidate.rs and toolkit/tests/agent_generated.rs; use that scope where practical, or state clearly if an additional evaluator is required. The supervisor has not executed your proposed follow-up. Return only the requested structured JSON with concise single-line strings.
'''


def call_diagnoser(config, worktree, round_dir, result):
    response = round_dir / 'diagnosis.json'
    schema = round_dir.parent / 'diagnosis-output-schema.json'
    args = ['codex', '-a', 'never', 'exec', '--sandbox', 'read-only', '--ephemeral', '--json',
            '--cd', str(worktree), '--output-schema', str(schema), '--output-last-message', str(response)]
    if config.get('local_provider'):
        args += ['--oss', '--local-provider', config['local_provider']]
    if config.get('model'):
        args += ['--model', config['model']]
    args.append('-')
    prompt = diagnosis_prompt(result, round_dir)
    (round_dir / 'diagnosis-prompt.txt').write_text(prompt, encoding='utf-8')
    started = time.monotonic()
    try:
        with (round_dir / 'diagnosis-events.jsonl').open('wb') as stdout, (round_dir / 'diagnosis-stderr.log').open('wb') as stderr:
            code = bounded_process(args, cwd=worktree, input_bytes=prompt.encode('utf-8'), stdout=stdout,
                                   stderr=stderr, timeout=config['diagnostic_seconds'])
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise runner.PlanError(f'diagnosis could not complete: {exc}') from exc
    if code:
        raise runner.PlanError(f'diagnosis agent exited {code}; inspect diagnosis logs')
    diagnosis = runner.read_json(response)
    runner.exact_keys(diagnosis, DIAGNOSIS_FIELDS, 'diagnosis')
    for key, value in diagnosis.items():
        runner.text_field(value, key)
        if len(value) > 1000:
            raise runner.PlanError(f'diagnosis field {key} too long')
    return {'status': 'complete', 'analysis': diagnosis, 'elapsed_seconds': time.monotonic() - started,
            'usage': agent_usage(round_dir / 'diagnosis-events.jsonl'),
            'qualification': 'Agent-proposed explanations and follow-up are unverified.'}


def diagnose_failure(config, worktree, round_dir, result, diagnoser=None):
    if not config.get('diagnose_failures', False):
        return {'status': 'disabled'}
    reports = [result.get('development'), result.get('evaluation')]
    if not any(isinstance(report, dict) and report.get('outcome') == 'criteria_not_met' for report in reports):
        return {'status': 'not_needed'}
    try:
        return (diagnoser or call_diagnoser)(config, worktree, round_dir, result)
    except (runner.PlanError, OSError, subprocess.SubprocessError) as exc:
        return {'status': 'failed', 'error': str(exc), 'usage': agent_usage(round_dir / 'diagnosis-events.jsonl'),
                'qualification': 'No diagnosis was produced.'}


def reported_input_usage(completed):
    total = 0
    for item in completed:
        usage = item.get('agent_usage')
        if not isinstance(usage, dict) or not isinstance(usage.get('input_tokens'), int) or isinstance(usage['input_tokens'], bool) or usage['input_tokens'] < 0:
            return None
        total += usage['input_tokens']
        diagnosis = item.get('diagnosis') or {}
        if diagnosis.get('status') in ('complete', 'failed'):
            usage = diagnosis.get('usage')
            if not isinstance(usage, dict) or not isinstance(usage.get('input_tokens'), int) or isinstance(usage['input_tokens'], bool) or usage['input_tokens'] < 0:
                return None
            total += usage['input_tokens']
    return total


def changed_paths(worktree):
    # -z avoids ambiguity from spaces and unusual path names.
    paths = set()
    for args in (('diff', '--name-only', '-z', 'HEAD'), ('ls-files', '--others', '--exclude-standard', '-z')):
        raw = subprocess.run(['git', *args], cwd=worktree, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
        paths.update(os.fsdecode(x) for x in raw.split(b'\0') if x)
    return paths


def verify_agent_changes(worktree, base):
    if git(worktree, 'rev-parse', 'HEAD') != base:
        raise runner.PlanError('agent changed Git HEAD; preserve the worktree for inspection')
    changes = changed_paths(worktree)
    if not changes or 'toolkit/src/agent_candidate.rs' not in changes:
        raise runner.PlanError('agent did not change the candidate model')
    if changes - ALLOWED:
        raise runner.PlanError(f'agent changed files outside candidate scope: {sorted(changes - ALLOWED)}')
    for name in ALLOWED:
        path = worktree / name
        if not path.is_file() or path.is_symlink():
            raise runner.PlanError(f'candidate or test missing or symlinked: {name}')
    if '#[test]' not in (worktree / 'toolkit/tests/agent_generated.rs').read_text(encoding='utf-8'):
        raise runner.PlanError('candidate test file contains no Rust test')


def build_environment(directory, config):
    """Use a campaign-owned target directory, without changing the caller's environment."""
    cache = directory / 'build-cache'
    marker = cache / '.murmur-campaign'
    if not cache.exists() and not cache.is_symlink():
        cache.mkdir()
        marker.write_text(config['campaign_id'], encoding='utf-8')
    if cache.is_symlink() or not marker.is_file() or marker.is_symlink() or marker.read_text(encoding='utf-8') != config['campaign_id']:
        raise runner.PlanError(f'build cache is not owned by this campaign: {cache}')
    return {**os.environ, 'CARGO_TARGET_DIR': str(cache.resolve())}


def cleanup_round(directory, config, item):
    """Remove only a clean completed worktree whose branch and sealed evidence survive."""
    index = item['index']
    round_dir = directory / f'idea-{index:03d}'
    worktree = round_dir / 'worktree'
    result = runner.read_json(round_dir / 'result.json')
    expected_branch = f"toolkit/campaign-{config['campaign_id']}-{index:03d}"
    if (item.get('status') != 'complete' or result.get('status') != 'complete'
            or result.get('commit') != item.get('commit') or result.get('branch') != expected_branch):
        raise runner.PlanError('round metadata does not identify a completed candidate')
    if git(runner.ROOT, 'rev-parse', '--verify', f'refs/heads/{expected_branch}') != item['commit']:
        raise runner.PlanError('candidate branch no longer points to the recorded commit')
    study = round_dir / 'study'
    if runner.stage_state(study, 'development') != 'complete':
        raise runner.PlanError('development evidence is incomplete')
    evaluation = runner.stage_state(study, 'evaluation')
    if evaluation != ('complete' if result.get('evaluation') is not None else 'pending'):
        raise runner.PlanError('evaluation evidence does not match the recorded result')
    if not worktree.exists() and not worktree.is_symlink():
        return {'index': index, 'status': 'already_absent'}
    if worktree.is_symlink() or git(worktree, 'rev-parse', 'HEAD') != item['commit']:
        raise runner.PlanError('worktree is symlinked or its commit changed')
    if git(worktree, 'status', '--porcelain=v1', '--untracked-files=all'):
        raise runner.PlanError('worktree contains uncommitted or untracked files')
    raw = subprocess.run(['git', 'status', '--porcelain=v1', '-z', '--ignored=matching', '--untracked-files=normal'],
                         cwd=worktree, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout
    ignored = [os.fsdecode(entry[3:]) for entry in raw.split(b'\0') if entry]
    if any(path not in ('target/', 'toolkit/target/') for path in ignored):
        raise runner.PlanError(f'worktree contains other ignored files: {ignored}')
    git(runner.ROOT, 'worktree', 'remove', str(worktree))
    return {'index': index, 'status': 'removed'}


def cleanup_completed(directory, config, state, budget_finished=False):
    results = []
    for item in state.get('completed', []):
        if item.get('status') != 'complete':
            continue
        try:
            outcome = cleanup_round(directory, config, item)
        except (runner.PlanError, OSError, subprocess.SubprocessError) as exc:
            outcome = {'index': item.get('index'), 'status': 'retained', 'reason': str(exc)}
        results.append(outcome)
    cache_status = 'retained'
    cache = directory / 'build-cache'
    finished = len(state.get('completed', [])) >= config['max_ideas'] or (budget_finished and bool(state.get('completed')))
    if state.get('status') == 'ready' and finished and all(x.get('status') == 'complete' for x in state['completed']) and all(x['status'] != 'retained' for x in results):
        if not cache.exists() and not cache.is_symlink():
            cache_status = 'absent'
        else:
            try:
                build_environment(directory, config)  # verifies the ownership marker and symlinks
                shutil.rmtree(cache)
                cache_status = 'removed'
            except (runner.PlanError, OSError) as exc:
                cache_status = f'retained: {exc}'
    outcome = {'rounds': results, 'cache': cache_status}
    runner.write_json(directory / 'cleanup.json', outcome)
    return outcome


def cleanup(directory):
    directory = Path(directory).resolve()
    lock = directory / '.running.lock'
    try:
        lock.mkdir()
    except OSError as exc:
        raise runner.PlanError(f'campaign is locked or unavailable; inspect {lock}: {exc}') from exc
    try:
        return cleanup_completed(directory, runner.read_json(directory / 'campaign.json'), runner.read_json(directory / 'state.json'))
    finally:
        lock.rmdir()


def execute_round(directory, config, state, agent=call_agent):
    index = state['next_index']
    round_dir = directory / f'idea-{index:03d}'
    try:
        round_dir.mkdir(exist_ok=False)
    except OSError as exc:
        raise runner.PlanError(f'cannot create round; prior work is not overwritten: {exc}') from exc
    worktree = round_dir / 'worktree'
    branch = f"toolkit/campaign-{config['campaign_id']}-{index:03d}"
    base = state['base_commit']
    result = {'index': index, 'status': 'running', 'base_commit': base, 'branch': branch}
    runner.write_json(round_dir / 'result.json', result)
    try:
        git(runner.ROOT, 'worktree', 'add', '-b', branch, str(worktree), base)
        research_index = memory.build_index()
        research_index['agent_context'] = memory.snapshot_for_agent(research_index, config['campaign_id'])
        research_index['agent_context']['full_snapshot_path'] = str(round_dir / 'research-memory.json')
        runner.write_json(round_dir / 'research-memory.json', research_index)
        agent_started = time.monotonic()
        idea = agent(config, worktree, round_dir, config.get('prior_history', []) + state['completed'])
        result['agent_elapsed_seconds'] = time.monotonic() - agent_started
        result['agent_usage'] = agent_usage(round_dir / 'agent-events.jsonl')
        verify_agent_changes(worktree, base)
        dev, evaluation = seed_configs(index, config['seed_base'])
        plan = make_plan(index, idea, config.get('protocol', 'legacy'))
        runner.validate_plan(plan, dev, evaluation)
        (worktree / 'toolkit/configs/agent').mkdir(parents=True, exist_ok=True)
        (worktree / 'toolkit/plans').mkdir(parents=True, exist_ok=True)
        (worktree / 'toolkit/configs/agent/dev.cfg').write_text(dev, encoding='utf-8')
        (worktree / 'toolkit/configs/agent/evaluation.cfg').write_text(evaluation, encoding='utf-8')
        runner.write_json(worktree / 'toolkit/plans/agent-candidate.json', plan)
        with (round_dir / 'tests.log').open('wb') as out:
            tests = bounded_process(['cargo', 'test', '--offline', '--manifest-path', 'toolkit/Cargo.toml',
                                     '--test', 'candidate_context', '--test', 'agent_generated', '--', '--nocapture'],
                                    cwd=worktree, stdout=out, stderr=subprocess.STDOUT, timeout=config['test_seconds'],
                                    env=build_environment(directory, config))
        if tests:
            raise runner.PlanError(f'candidate tests failed ({tests}); see tests.log')
        result['followups'] = followup_metrics(round_dir / 'tests.log')
        unexpected = changed_paths(worktree) - ALLOWED - GENERATED
        if unexpected:
            raise runner.PlanError(f'unexpected changes after tests: {sorted(unexpected)}')
        git(worktree, 'add', '--', *sorted(ALLOWED | GENERATED))
        git(worktree, 'commit', '-m', f"Research agent candidate {index:03d}")
        commit = git(worktree, 'rev-parse', 'HEAD')
        study = round_dir / 'study'
        with (round_dir / 'coordinator.log').open('wb') as out:
            for args in (('prepare', 'toolkit/plans/agent-candidate.json', str(study)), ('cycle', str(study))):
                code = bounded_process([sys.executable, '-B', '-m', 'toolkit.coordinator.runner', *args],
                                       cwd=worktree, stdout=out, stderr=subprocess.STDOUT,
                                       timeout=config['test_seconds'] + 200, env=build_environment(directory, config))
                if code:
                    raise runner.PlanError(f'coordinator {args[0]} failed ({code}); inspect coordinator.log')
        development = runner.read_json(study / 'development/report.json')
        evaluation_path = study / 'evaluation/report.json'
        evaluation = runner.read_json(evaluation_path) if evaluation_path.exists() else None
        evidence_stage = 'evaluation' if evaluation is not None else 'development'
        matrix = probe_matrix(study / evidence_stage / 'evidence' / 'probes.csv')
        result.update(status='complete', commit=commit, idea=idea, development=development, evaluation=evaluation, final_probe_matrix=matrix)
        result['diagnosis'] = diagnose_failure(config, worktree, round_dir, result)
        if git(worktree, 'rev-parse', 'HEAD') != commit or changed_paths(worktree):
            raise runner.PlanError('worktree changed during diagnosis; preserve it for inspection')
        if runner.stage_state(study, 'development') != 'complete' or (evaluation is not None and runner.stage_state(study, 'evaluation') != 'complete'):
            raise runner.PlanError('study evidence changed during diagnosis')
        state['base_commit'] = commit
    except (runner.PlanError, OSError, subprocess.SubprocessError) as exc:
        result.update(status='failed', error=str(exc))
    runner.write_json(round_dir / 'result.json', result)
    state['completed'].append({k: result.get(k) for k in ('index', 'status', 'commit', 'idea', 'agent_elapsed_seconds', 'agent_usage', 'followups', 'development', 'evaluation', 'final_probe_matrix', 'diagnosis', 'error')})
    state['next_index'] += 1
    state['status'] = 'ready' if result['status'] == 'complete' else 'inspect_failure'
    runner.write_json(directory / 'state.json', state)
    if result['status'] == 'complete':
        cleanup_completed(directory, config, state)
    return result


def run(directory, max_new=None, agent=call_agent):
    directory = Path(directory).resolve()
    lock = directory / '.running.lock'
    try:
        lock.mkdir()
    except OSError as exc:
        raise runner.PlanError(f'campaign is locked or unavailable; inspect {lock}: {exc}') from exc
    try:
        return _run_locked(directory, max_new, agent)
    finally:
        lock.rmdir()


def _run_locked(directory, max_new=None, agent=call_agent):
    config = runner.read_json(directory / 'campaign.json')
    state = runner.read_json(directory / 'state.json')
    if config.get('schema_version') != 1 or state.get('schema_version') != 1:
        raise runner.PlanError('invalid campaign metadata')
    if state.get('status') != 'ready':
        return {'action': 'inspect_failure', 'completed': len(state.get('completed', []))}
    remaining = config['max_ideas'] - len(state['completed'])
    count = remaining if max_new is None else min(remaining, max_new)
    if count < 0:
        raise runner.PlanError('invalid campaign iteration limit')
    results = []
    budget_action = None
    for _ in range(count):
        budget = config.get('max_reported_input_tokens')
        used = reported_input_usage(state['completed']) if budget is not None else None
        if budget is not None and used is None:
            budget_action = 'inspect_usage'
            break
        if budget is not None and used >= budget:
            budget_action = 'reported_budget_reached'
            break
        result = execute_round(directory, config, state, agent)
        results.append({'index': result['index'], 'status': result['status'], 'commit': result.get('commit'),
                        'development': (result.get('development') or {}).get('outcome'),
                        'evaluation': (result.get('evaluation') or {}).get('outcome'),
                        'diagnosis': (result.get('diagnosis') or {}).get('status'), 'error': result.get('error')})
        if result['status'] != 'complete':
            break
    budget = config.get('max_reported_input_tokens')
    if budget is not None and budget_action is None and len(state['completed']) < config['max_ideas']:
        used = reported_input_usage(state['completed'])
        if used is None:
            budget_action = 'inspect_usage'
        elif used >= budget:
            budget_action = 'reported_budget_reached'
    action = 'inspect_failure' if state['status'] != 'ready' else ('review_results' if len(state['completed']) >= config['max_ideas'] else (budget_action or 'continue'))
    if action == 'reported_budget_reached':
        cleanup_completed(directory, config, state, budget_finished=True)
    return {'action': action, 'rounds': results, 'completed': len(state['completed']), 'limit': config['max_ideas'],
            'reported_input_tokens': reported_input_usage(state['completed']),
            'max_reported_input_tokens': config.get('max_reported_input_tokens')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('init'); p.add_argument('directory'); p.add_argument('--goal', default=GOAL)
    p.add_argument('--max-ideas', type=int, default=3); p.add_argument('--agent-seconds', type=int, default=600)
    p.add_argument('--test-seconds', type=int, default=180); p.add_argument('--local-provider', choices=('ollama', 'lmstudio'))
    p.add_argument('--model'); p.add_argument('--prior-campaign', help='include prior campaign outcomes in the agent context')
    p.add_argument('--protocol', choices=('milestone', 'legacy'), default='milestone')
    p.add_argument('--diagnostic-seconds', type=int, default=120)
    p.add_argument('--no-diagnosis', action='store_true', help='skip the separate failure-analysis agent')
    p.add_argument('--max-reported-input-tokens', type=int, help='stop between rounds when reported total reaches this threshold (not a hard cap)')
    p = sub.add_parser('run'); p.add_argument('directory'); p.add_argument('--ideas', type=int)
    p = sub.add_parser('cleanup'); p.add_argument('directory')
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            result = init(args.directory, args.goal, args.max_ideas, args.agent_seconds, args.test_seconds, args.local_provider, args.model, args.prior_campaign, args.protocol, args.diagnostic_seconds, not args.no_diagnosis, args.max_reported_input_tokens)
        elif args.command == 'cleanup':
            result = cleanup(args.directory)
        else:
            if args.ideas is not None and args.ideas < 1:
                raise runner.PlanError('--ideas must be positive')
            result = run(args.directory, args.ideas)
        print(json.dumps(result, indent=2, sort_keys=True))
    except (runner.PlanError, OSError, subprocess.SubprocessError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

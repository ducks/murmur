import csv
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from toolkit.coordinator import runner


def config(seeds, train=10, test=8):
    return ('schema_version=1\nseeds=' + ','.join(map(str, seeds)) +
            f'\ndelays=1\ntrain_tokens={train}\ntest_tokens={test}\nunits=8\ndegree=2\nlearning_rate=0.05\n')


def plan():
    return {
        'schema_version': 1, 'id': 'test', 'question': 'Does it work?', 'hypothesis': 'It improves.',
        'falsification': 'No improvement.', 'adapter': 'unlabeled_context',
        'development_config': 'dev.cfg', 'evaluation_config': 'eval.cfg',
        'comparison': {'candidate': 'mixture', 'baseline': 'single',
                       'required_models': ['single', 'mixture', 'known_filter', 'oracle'],
                       'metric': 'accuracy', 'regime': 'aba', 'noise': '0.25',
                       'checkpoint': '3', 'probe_rule': '1'},
        'criteria': {'min_mean_difference': 0.1, 'min_paired_win_fraction': 0.5, 'max_stationary_accuracy_drop': 0.1},
        'budget': {'max_seeds_per_split': 4, 'max_logical_transitions_per_split': 10000, 'max_wall_seconds': 10}}


def evidence(path, seeds, mixture=7, single=4):
    path.mkdir()
    (path / 'status.txt').write_text('complete\n')
    with (path / 'probes.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['seed','regime','noise','checkpoint','rule','model','correct','trials','accuracy','mean_logloss'])
        for seed in seeds:
            for regime in ('aba', 'stationary', 'switching'):
                for noise in ('0.1', '0.25'):
                    for checkpoint in ('1', '2', '3'):
                        for rule in ('0', '1'):
                            for model in sorted(runner.MODELS):
                                correct = mixture if model == 'mixture' else single
                                w.writerow([seed, regime, noise, checkpoint, rule, model, correct, 8, correct/8, 0.5])
    with (path / 'windows.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['seed','regime','noise','window_start','window_end','model','correct','trials','accuracy','mean_logloss'])
        for seed in seeds:
            for regime in ('aba', 'stationary', 'switching'):
                for noise in ('0.1', '0.25'):
                    for model in sorted(runner.MODELS):
                        correct = 15
                        w.writerow([seed, regime, noise, 0, 30, model, correct, 30, correct/30, 0.5])


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dev = config([1, 2]); self.evaluation = config([100001, 100002])
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text(json.dumps(plan()))
    def tearDown(self): self.temp.cleanup()
    def test_validate_rejects_cross_role_overlap_and_unmatched_configuration(self):
        runner.validate_plan(plan(), self.dev, self.evaluation)
        with self.assertRaisesRegex(runner.PlanError, 'seed roles overlap'):
            runner.validate_plan(plan(), self.dev, config([6001]))
        with self.assertRaisesRegex(runner.PlanError, 'differ only in seeds'):
            runner.validate_plan(plan(), self.dev, config([100001], train=11))
        bad = plan(); bad['comparison']['required_models'].remove('oracle')
        with self.assertRaisesRegex(runner.PlanError, 'all four'):
            runner.validate_plan(bad, self.dev, self.evaluation)
    def test_repo_output_must_be_ignored(self):
        self.assertEqual(runner.output_location(runner.ROOT / 'toolkit/runs/coordinator-test-location'),
                         runner.ROOT / 'toolkit/runs/coordinator-test-location')
        with self.assertRaisesRegex(runner.PlanError, 'must be git-ignored'):
            runner.output_location(runner.ROOT / 'not-ignored-coordinator-test')
    def test_budget_and_unknown_fields_rejected(self):
        p = plan(); p['budget']['max_logical_transitions_per_split'] = 1
        with self.assertRaisesRegex(runner.PlanError, 'exceeds'):
            runner.validate_plan(p, self.dev, self.evaluation)
        p = plan(); p['silent_option'] = True
        with self.assertRaisesRegex(runner.PlanError, 'exactly'):
            runner.validate_plan(p, self.dev, self.evaluation)
    def test_evidence_validation_detects_missing_and_corrupt_controls(self):
        cfg = runner.config_parse(self.dev); dest = self.root / 'evidence'; evidence(dest, [1, 2])
        report = runner.analyze(plan(), cfg, dest)
        self.assertEqual(report['outcome'], 'criteria_met')
        self.assertEqual(report['wins'], 2)
        content = (dest / 'probes.csv').read_text().splitlines()
        (dest / 'probes.csv').write_text('\n'.join(content[:-1]) + '\n')
        with self.assertRaisesRegex(runner.PlanError, 'incomplete'):
            runner.analyze(plan(), cfg, dest)
    def test_frozen_plan_refuses_tampering_and_source_changes(self):
        with patch.object(runner, 'config_path', side_effect=[self.dev, self.evaluation]), patch.object(runner, 'source_fingerprint', return_value=('head', 'original')):
            target = runner.prepare(self.manifest, self.root / 'run')
        with patch.object(runner, 'source_fingerprint', return_value=('head', 'changed')):
            with self.assertRaisesRegex(runner.PlanError, 'source differs'):
                runner.load_lock(target)
        (target / 'development.cfg').write_text(self.dev + '# altered\n')
        with self.assertRaisesRegex(runner.PlanError, 'frozen file changed'):
            runner.load_lock(target, verify_source=False)
    def test_full_workflow_and_no_overwrite(self):
        with patch.object(runner, 'config_path', side_effect=[self.dev, self.evaluation]), patch.object(runner, 'source_fingerprint', return_value=('head', 'same')):
            target = runner.prepare(self.manifest, self.root / 'run')
        def fake_process(args, **kwargs):
            stage = Path(args[-1]); split = stage.parent.name
            evidence(stage, [1, 2] if split == 'development' else [100001, 100002])
            return subprocess.CompletedProcess(args, 0)
        with patch.object(runner, 'source_fingerprint', return_value=('head', 'same')), patch.object(runner.subprocess, 'run', side_effect=fake_process):
            self.assertEqual(runner.next_action(target)['action'], 'run_development')
            with self.assertRaisesRegex(runner.PlanError, 'complete development'):
                runner.run_stage(target, 'evaluation')
            self.assertEqual(runner.run_stage(target, 'development')['outcome'], 'criteria_met')
            self.assertEqual(runner.next_action(target)['action'], 'run_evaluation')
            self.assertEqual(runner.run_stage(target, 'evaluation')['outcome'], 'criteria_met')
            self.assertEqual(runner.next_action(target)['action'], 'review_results')
            with self.assertRaisesRegex(runner.PlanError, 'never overwritten'):
                runner.run_stage(target, 'development')
            (target / 'development' / 'evidence' / 'probes.csv').write_text('tampered')
            with self.assertRaisesRegex(runner.PlanError, 'evidence changed'):
                runner.next_action(target)
    def test_failed_development_stops_evaluation(self):
        with patch.object(runner, 'config_path', side_effect=[self.dev, self.evaluation]), patch.object(runner, 'source_fingerprint', return_value=('head', 'same')):
            target = runner.prepare(self.manifest, self.root / 'run')
        def fake_process(args, **kwargs):
            evidence(Path(args[-1]), [1, 2], mixture=4, single=7)
            return subprocess.CompletedProcess(args, 0)
        with patch.object(runner, 'source_fingerprint', return_value=('head', 'same')), patch.object(runner.subprocess, 'run', side_effect=fake_process):
            self.assertEqual(runner.run_stage(target, 'development')['outcome'], 'criteria_not_met')
            self.assertEqual(runner.next_action(target)['action'], 'revise_plan')
            with self.assertRaisesRegex(runner.PlanError, 'not met'):
                runner.run_stage(target, 'evaluation')
            self.assertFalse((target / 'evaluation').exists())

if __name__ == '__main__': unittest.main()

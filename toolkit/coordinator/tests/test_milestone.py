import csv
from pathlib import Path
import tempfile
import unittest

from toolkit.coordinator import campaign, runner

class MilestoneTests(unittest.TestCase):
    def setUp(self):
        self.idea = {'question': 'q', 'hypothesis': 'h', 'falsification': 'f'}
        self.plan = campaign.make_plan(1, self.idea, 'milestone')
        self.dev, self.eval = campaign.seed_configs(1)
        self.cfg, _ = runner.validate_plan(self.plan, self.dev, self.eval)
        self.cfg = dict(self.cfg, seeds=[10, 11], train_tokens=100, test_tokens=100)

    def evidence(self, root, failing=None, omit=None):
        fields = ['seed','family','phase','measurement','model','correct','trials','accuracy','logloss']
        path = root / 'metrics.csv'
        with path.open('w', newline='') as f:
            out = csv.writer(f); out.writerow(fields)
            for seed in self.cfg['seeds']:
                for family in runner.MILESTONE_FAMILIES:
                    for phase in ('1','2','3'):
                        for metric in ('online','probe_a','probe_b','window_0000'):
                            for model in ('candidate','mixture','single','oracle'):
                                key = (str(seed), family, phase, metric, model)
                                if key == omit: continue
                                correct = 50
                                if model == 'candidate' and (family, phase, metric) == failing:
                                    correct = 47
                                out.writerow([*key, correct, 100, correct/100, 0.7])
        with (root / 'resources.csv').open('w', newline='') as f:
            out = csv.writer(f); out.writerow(['seed','family','model','inline_bytes','training_ns','training_steps'])
            for seed in self.cfg['seeds']:
                for family in runner.MILESTONE_FAMILIES:
                    for model in ('candidate','mixture','single','oracle'):
                        out.writerow([seed, family, model, 32, 1000, 300])
        return path

    def test_requires_every_family_retention_and_both_switches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.evidence(root)
            report = runner.analyze_milestone(self.plan, self.cfg, root)
            self.assertEqual(report['outcome'], 'criteria_met')
            self.assertEqual(len(report['checks']), 12)
            for family, phase, metric, label in [('constant_bias','3','probe_b','retention_b'),
                                                  ('asymmetric','2','window_0000','switch_to_b'),
                                                  ('unequal_noise','3','window_0000','switch_to_a')]:
                self.evidence(root, (family, phase, metric))
                report = runner.analyze_milestone(self.plan, self.cfg, root)
                self.assertEqual(report['outcome'], 'criteria_not_met')
                self.assertFalse(report['checks'][f'{family}:{label}'])

    def test_missing_rows_fail_closed_and_plan_checks_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.evidence(root, omit=('10','complementary','2','window_0000','candidate'))
            with self.assertRaisesRegex(runner.PlanError, 'incomplete'):
                runner.analyze_milestone(self.plan, self.cfg, root)
        self.plan['budget']['max_logical_transitions_per_split'] = 1
        with self.assertRaisesRegex(runner.PlanError, 'budget'):
            runner.validate_plan(self.plan, self.dev, self.eval)

    def test_prompt_explains_criteria_and_history(self):
        prompt = campaign.prompt_for({'goal':'retention', 'protocol':'milestone'},
                                     [{'development': {'outcome':'criteria_not_met', 'checks':{'foo':False}}}])
        self.assertIn('first 100', prompt)
        self.assertIn('foo', prompt)
        self.assertIn('2 percentage points', prompt)

if __name__ == '__main__':
    unittest.main()

import importlib.util
from pathlib import Path
import unittest

script = Path(__file__).resolve().parents[2] / 'scripts' / 'compare_milestone_candidates.py'
spec = importlib.util.spec_from_file_location('matched_comparison', script)
matched = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matched)

class MatchedComparisonTests(unittest.TestCase):
    def make_rows(self, candidate):
        rows = {}
        for seed in (1, 2):
            for phase, measurement in (('3','probe_b'), ('2','window_0000'), ('3','window_0000')):
                for model in ('candidate','mixture','single','oracle'):
                    key = (str(seed), 'family', phase, measurement, model)
                    value = candidate if model == 'candidate' else 0.5
                    rows[key] = {'accuracy': str(value), 'seed': str(seed), 'model': model}
        return rows

    def test_paired_difference_and_repeatable_bootstrap(self):
        a, b = self.make_rows(0.4), self.make_rows(0.6)
        first = matched.compare(a, b, [1, 2], ['family'], 100, 17)
        self.assertEqual(first, matched.compare(a, b, [1, 2], ['family'], 100, 17))
        for item in first.values():
            self.assertAlmostEqual(item['idea_2_minus_idea_1'], 0.2)
            self.assertEqual(item['wins_losses_ties'], [2, 0, 0])

    def test_mismatched_control_is_rejected(self):
        a, b = self.make_rows(0.4), self.make_rows(0.6)
        b[('1', 'family', '3', 'probe_b', 'mixture')]['accuracy'] = '0.3'
        with self.assertRaisesRegex(ValueError, 'mixture results differ'):
            matched.compare(a, b, [1, 2], ['family'], 100, 17)

if __name__ == '__main__': unittest.main()

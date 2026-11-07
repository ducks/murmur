import json
import unittest

from toolkit.coordinator import runner, verify_paired_log


def sample():
    rows = [
        'seed=0 stream=10 init=20 probe=30 switch_b=[10, 11, 12] switch_a=[20, 21, 22] recall_b=[300, 301, 302]',
        'seed=1 stream=11 init=21 probe=31 switch_b=[13, 14, 15] switch_a=[23, 24, 25] recall_b=[303, 304, 305]',
    ]
    totals = ((23,25,27),(43,45,47),(603,605,607))
    for w, measure in enumerate(verify_paired_log.MEASURES):
        for a,b in ((0,1),(0,2),(1,2)):
            rows.append('MURMUR_FOLLOWUP '+json.dumps({'measure':'high_noise_200_'+measure,
                'candidate':verify_paired_log.ARMS[a], 'control':verify_paired_log.ARMS[b],
                'candidate_correct':totals[w][a], 'control_correct':totals[w][b], 'trials':2000 if w==2 else 200}))
    return '\n'.join(rows)


class VerifyPairedLogTests(unittest.TestCase):
    def test_recomputes_aggregate_and_is_reproducible(self):
        result = verify_paired_log.check(sample(), 2, [10,20,30], draws=100)
        key = 'high_noise_200_retention_b:consistency_filter:unguarded'
        self.assertEqual(result['comparisons'][key]['mean_paired_difference'], -0.001)
        self.assertEqual(result, verify_paired_log.check(sample(), 2, [10,20,30], draws=100))

    def test_rejects_missing_duplicate_and_incorrect_seed_roles(self):
        lines=sample().splitlines()
        for text in ('\n'.join(lines[1:]), sample()+'\n'+lines[0], sample().replace('stream=11','stream=99')):
            with self.subTest(text=text[:30]), self.assertRaises(runner.PlanError):
                verify_paired_log.check(text, 2, [10,20,30], draws=100)

    def test_rejects_wrong_counts_and_missing_comparisons(self):
        for text in (sample().replace('"candidate_correct": 603', '"candidate_correct": 604', 1),
                     '\n'.join(sample().splitlines()[:-1]), sample().replace('recall_b=[300, 301, 302]','recall_b=[1001, 301, 302]')):
            with self.subTest(text=text[-40:]), self.assertRaises(runner.PlanError):
                verify_paired_log.check(text, 2, [10,20,30], draws=100)


if __name__ == '__main__':
    unittest.main()

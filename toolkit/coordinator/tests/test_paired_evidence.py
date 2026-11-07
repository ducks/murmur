import json
from pathlib import Path
import tempfile
import unittest
from toolkit.coordinator import campaign, paired_evidence, runner


class PairedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.path = self.root / 'tests.log'
        self.milestone = {'primary_measure':'recall','candidate_label':'model','control_label':'base'}
        self.base = {'measure':'recall','candidate':'model','control':'base'}
        self.rows = [dict(self.base, seed=11, candidate_correct=7, control_correct=6, trials=10),
                     dict(self.base, seed=12, candidate_correct=4, control_correct=6, trials=10)]
        runner.write_json(self.root/'followup-records.json', {'records':[dict(self.base, candidate_correct=11, control_correct=12, trials=20)]})
        self.followups = {'full_records_path':str(self.root/'followup-records.json')}

    def tearDown(self): self.tmp.cleanup()
    def write(self, rows): self.path.write_text(''.join('MURMUR_PAIRED '+json.dumps(x)+'\n' for x in rows))
    def check(self, rows):
        self.write(rows)
        return paired_evidence.validate(self.path, self.milestone, self.followups, draws=100)

    def test_valid_records_recompute_primary_and_preserve_evidence(self):
        result=self.check(self.rows)
        self.assertEqual(result['status'],'arithmetic_checked'); self.assertEqual(result['seeds'],2)
        self.assertAlmostEqual(result['mean_paired_difference'],-0.05)
        self.assertEqual(len(runner.read_json(Path(result['records_path']))['records']),2)
        self.assertIn('no proof',result['qualification'])
        self.assertEqual(result,self.check(list(reversed(self.rows))))

    def test_rejects_bad_or_missing_records(self):
        for rows in ([], self.rows[:1], self.rows+[self.rows[0]],
                     [self.rows[0],{**self.rows[1],'seed':True}],
                     [self.rows[0],{**self.rows[1],'control':'different'}],
                     [self.rows[0],{**self.rows[1],'trials':9}],
                     [self.rows[0],{**self.rows[1],'candidate_correct':5}],
                     [self.rows[0],{**self.rows[1],'control_correct':11}]):
            with self.subTest(rows=rows), self.assertRaises(runner.PlanError): self.check(rows)

    def test_duplicate_primary_aggregate_rejected(self):
        record=dict(self.base,candidate_correct=11,control_correct=12,trials=20)
        runner.write_json(self.root/'followup-records.json',{'records':[record,record]})
        with self.assertRaises(runner.PlanError):self.check(self.rows)

    def test_prompt_only_requires_contract_in_new_campaigns(self):
        self.assertIn('MURMUR_PAIRED', campaign.prompt_for({'goal':'g','paired_evidence_version':1}, []))
        self.assertNotIn('MURMUR_PAIRED', campaign.prompt_for({'goal':'g'}, []))


if __name__ == '__main__': unittest.main()

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

    def test_well_formed_non_primary_records_do_not_block_primary(self):
        extra={**self.rows[0], 'measure':'older_experiment'}
        result=self.check([extra,*self.rows])
        self.assertEqual(result['seeds'],2)
        self.assertEqual(result['other_records'],1)
        malformed={**extra,'trials':0}
        with self.assertRaises(runner.PlanError):self.check([malformed,*self.rows])

    def test_allocated_ids_must_match_exactly(self):
        self.write(self.rows)
        result=paired_evidence.validate(self.path,self.milestone,self.followups,draws=100,expected_seed_ids={11,12})
        self.assertEqual(result['seeds'],2)
        with self.assertRaisesRegex(runner.PlanError,'supervisor allocation'):
            paired_evidence.validate(self.path,self.milestone,self.followups,draws=100,expected_seed_ids={11,13})

    def test_supervisor_allocation_is_reproducible_and_role_disjoint(self):
        first=campaign.allocate_primary_seeds('123456abcdef',1)
        self.assertEqual(first,campaign.allocate_primary_seeds('123456abcdef',1))
        self.assertEqual(first['count'],100)
        self.assertNotEqual(first['rows'],campaign.allocate_primary_seeds('123456abcdef',2)['rows'])
        self.assertNotEqual(first['rows'],campaign.allocate_primary_seeds('123456abcdee',1)['rows'])
        roles=[row[k] for row in first['rows'] for k in ('stream','initialization','probe')]
        self.assertEqual(len(roles),len(set(roles)))
        self.assertEqual([r['seed_id'] for r in first['rows']],list(range(100)))
        with self.assertRaises(runner.PlanError):campaign.allocate_primary_seeds('bad',1)

    def test_seed_allocation_is_present_in_agent_prompt(self):
        allocation=campaign.allocate_primary_seeds('123456abcdef',1,count=2)
        prompt=campaign.prompt_for({'goal':'g','paired_evidence_version':1},[],seed_allocation=allocation)
        self.assertIn('supervisor froze',prompt)
        self.assertIn(str(allocation['rows'][0]['stream']),prompt)

    def test_duplicate_primary_aggregate_rejected(self):
        record=dict(self.base,candidate_correct=11,control_correct=12,trials=20)
        runner.write_json(self.root/'followup-records.json',{'records':[record,record]})
        with self.assertRaises(runner.PlanError):self.check(self.rows)

    def test_prompt_only_requires_contract_in_new_campaigns(self):
        self.assertIn('MURMUR_PAIRED', campaign.prompt_for({'goal':'g','paired_evidence_version':1}, []))
        self.assertNotIn('MURMUR_PAIRED', campaign.prompt_for({'goal':'g'}, []))


if __name__ == '__main__': unittest.main()

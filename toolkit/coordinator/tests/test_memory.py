import json
from pathlib import Path
import tempfile
import unittest
from toolkit.coordinator import memory, campaign, runner

class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.findings = self.root / 'findings.json'
        runner.write_json(self.findings, {'schema_version': 1, 'findings': [{
            'id':'finding-1','question':'why','observation':'observed','hypothesis':'possible cause',
            'next_test':'controlled test','limitations':'not causal','evidence':['EXPERIMENTS.md'],
            'related_findings':[],'candidate_commits':[]}]})
    def tearDown(self): self.tmp.cleanup()
    def write_campaign(self, name, campaign_id='abc', status='complete'):
        path=self.root/name;path.mkdir()
        runner.write_json(path/'campaign.json',{'schema_version':1,'campaign_id':campaign_id,'goal':'goal','protocol':'milestone'})
        runner.write_json(path/'state.json',{'schema_version':1,'completed':[{'index':1,'status':status,'commit':'123',
            'idea':{'question':'test idea'},'development':{'outcome':'criteria_not_met','checks':{'switch':False},
            'comparisons':{'switch':{'mean_paired_difference':-0.1}}},'evaluation':None,'error':None}]})
        return path
    def test_indexes_failure_evidence_and_curated_explanation_separately(self):
        p=self.write_campaign('campaign')
        result=memory.build_index(self.root,self.findings)
        self.assertEqual(result['warnings'],[])
        self.assertEqual(result['findings'][0]['observation'],'observed')
        self.assertEqual(result['findings'][0]['hypothesis'],'possible cause')
        record=result['campaign_rounds'][0]
        self.assertEqual(record['development']['failed_checks'],[{'check':'switch','mean_paired_difference':-0.1}])
        self.assertEqual(record['evidence_directory'],str(p/'idea-001'))
        self.assertIn('not rechecked',record['verification'])
        context=memory.snapshot_for_agent(result,'other')
        prompt=campaign.prompt_for({'goal':'goal','protocol':'milestone'},[],context)
        self.assertIn('possible cause',prompt);self.assertIn('controlled test',prompt);self.assertIn('switch',prompt)
    def test_bad_campaign_is_reported_and_other_campaign_remains(self):
        self.write_campaign('good')
        broken=self.root/'broken';broken.mkdir();(broken/'campaign.json').write_text('{bad')
        result=memory.build_index(self.root,self.findings)
        self.assertEqual(len(result['warnings']),1);self.assertEqual(len(result['campaign_rounds']),1)
    def test_context_limit_is_explicit_and_avoids_current_campaign(self):
        index={'findings':[],'campaign_rounds':[{'campaign_id':'current','id':'x'},
            {'campaign_id':'old','id':'y'},{'campaign_id':'old','id':'z'}], 'warnings':[],'limitations':[]}
        result=memory.snapshot_for_agent(index,'current',limit=1)
        self.assertEqual(result['campaign_round_count'],2);self.assertEqual(result['campaign_rounds_shown'],1)
        self.assertEqual(result['campaign_rounds'][0]['id'],'z')
    def test_missing_evidence_is_rejected(self):
        data=json.loads(self.findings.read_text());data['findings'][0]['evidence']=['missing-file']
        runner.write_json(self.findings,data)
        with self.assertRaisesRegex(runner.PlanError,'missing'):
            memory.load_findings(self.findings)
    def test_unknown_relationship_is_rejected(self):
        data=json.loads(self.findings.read_text());data['findings'][0]['related_findings']=['unknown']
        runner.write_json(self.findings,data)
        with self.assertRaisesRegex(runner.PlanError,'unknown related finding'):
            memory.load_findings(self.findings)
    def test_duplicate_campaign_ids_are_reported(self):
        self.write_campaign('first');self.write_campaign('second')
        result=memory.build_index(self.root,self.findings)
        self.assertEqual(len(result['campaign_rounds']),1);self.assertEqual(len(result['warnings']),1)
    def test_in_progress_round_is_preserved_and_labeled(self):
        p=self.write_campaign('campaign')
        round_dir=p/'idea-002';round_dir.mkdir()
        runner.write_json(round_dir/'result.json',{'index':2,'status':'running','branch':'branch'})
        runner.write_json(round_dir/'idea.json',{'question':'unfinished idea'})
        result=memory.build_index(self.root,self.findings)
        record=result['campaign_rounds'][1]
        self.assertTrue(record['incomplete']);self.assertEqual(record['status'],'running')
        self.assertEqual(record['idea']['question'],'unfinished idea')

if __name__=='__main__': unittest.main()

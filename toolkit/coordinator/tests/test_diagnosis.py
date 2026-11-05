import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from toolkit.coordinator import campaign, memory, runner

class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        self.round=self.root/'idea-001';self.round.mkdir();self.worktree=self.round/'worktree';self.worktree.mkdir()
        self.config={'diagnose_failures':True,'diagnostic_seconds':10,'local_provider':None,'model':None}
        self.analysis={k:k+' text' for k in campaign.DIAGNOSIS_FIELDS}
        self.failed={'idea':{'question':'q'},'development':{'outcome':'criteria_not_met','checks':{'x':False}},'evaluation':None}
        runner.write_json(self.root/'diagnosis-output-schema.json',campaign.DIAGNOSIS_SCHEMA)
    def tearDown(self):self.tmp.cleanup()
    def test_failed_results_are_diagnosed_and_success_is_not(self):
        calls=[]
        def fake(*args):calls.append(args);return {'status':'complete','analysis':self.analysis}
        self.assertEqual(campaign.diagnose_failure(self.config,self.worktree,self.round,self.failed,fake)['status'],'complete')
        passed={'development':{'outcome':'criteria_met'},'evaluation':{'outcome':'criteria_met'}}
        self.assertEqual(campaign.diagnose_failure(self.config,self.worktree,self.round,passed,fake)['status'],'not_needed')
        self.assertEqual(len(calls),1)
        self.assertEqual(campaign.diagnose_failure({'diagnose_failures':False},self.worktree,self.round,self.failed,fake)['status'],'disabled')
    def test_diagnoser_failure_is_visible_and_does_not_fabricate_result(self):
        def bad(*_):raise runner.PlanError('unavailable')
        result=campaign.diagnose_failure(self.config,self.worktree,self.round,self.failed,bad)
        self.assertEqual(result['status'],'failed');self.assertIn('unavailable',result['error']);self.assertNotIn('analysis',result)
    def test_separate_agent_is_read_only_and_validates_output(self):
        def process(args,**kwargs):
            self.assertEqual(args[args.index('--sandbox')+1],'read-only')
            self.assertIn('competing explanations',kwargs['input_bytes'].decode())
            runner.write_json(self.round/'diagnosis.json',self.analysis)
            return 0
        with patch.object(campaign,'bounded_process',side_effect=process):
            result=campaign.call_diagnoser(self.config,self.worktree,self.round,self.failed)
        self.assertEqual(result['analysis'],self.analysis)
        prompt=campaign.prompt_for({'goal':'goal','protocol':'milestone'},[{'diagnosis':result}])
        self.assertIn('followup_test text',prompt)
    def test_malformed_diagnosis_is_rejected(self):
        def process(*args,**kwargs):
            runner.write_json(self.round/'diagnosis.json',{'observation':'missing fields'});return 0
        with patch.object(campaign,'bounded_process',side_effect=process):
            with self.assertRaises(runner.PlanError):campaign.call_diagnoser(self.config,self.worktree,self.round,self.failed)
    def test_memory_exposes_diagnosis_with_unverified_label(self):
        root=self.root/'campaign';root.mkdir()
        runner.write_json(root/'campaign.json',{'schema_version':1,'campaign_id':'id'})
        runner.write_json(root/'state.json',{'schema_version':1,'completed':[{'index':1,'status':'complete','diagnosis':{'status':'complete','analysis':self.analysis,'qualification':'unverified'}}]})
        record=memory.campaign_records(root)[0]
        self.assertEqual(record['diagnosis']['analysis'],self.analysis)
        self.assertEqual(record['diagnosis']['qualification'],'unverified')
    def test_followup_records_validate_paired_counts_and_warn_on_invalid(self):
        path=self.root/'tests.log'
        good={'measure':'switch','candidate':'revised','control':'original','candidate_correct':60,'control_correct':50,'trials':100}
        bad=dict(good, candidate_correct=101)
        path.write_text('test ablation ... MURMUR_FOLLOWUP '+json.dumps(good)+'\nMURMUR_FOLLOWUP '+json.dumps(bad)+'\n')
        result=campaign.followup_metrics(path)
        self.assertEqual(len(result['records']),1)
        self.assertEqual(result['records'][0]['candidate_accuracy'],0.6)
        self.assertEqual(result['records'][0]['control_accuracy'],0.5)
        self.assertEqual(len(result['warnings']),1)
        self.assertIn('not independently verified',result['qualification'])

if __name__=='__main__':unittest.main()

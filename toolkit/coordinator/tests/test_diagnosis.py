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
    def test_success_review_requests_ablation_without_claiming_a_cause(self):
        passed={'development':{'outcome':'criteria_met'},'evaluation':{'outcome':'criteria_met'}}
        config={**self.config,'review_successes':True}
        def fake(*args):return {'status':'complete','analysis':self.analysis}
        self.assertEqual(campaign.diagnose_failure(config,self.worktree,self.round,passed,fake)['status'],'complete')
        prompt=campaign.diagnosis_prompt(passed,self.round)
        self.assertIn('passed the operational checks',prompt)
        self.assertIn('matched ablation',prompt)
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
    def test_followup_preview_is_bounded_but_full_records_are_preserved(self):
        path=self.root/'tests.log'
        record={'measure':'switch','candidate':'revised','control':'original','candidate_correct':1,'control_correct':0,'trials':1}
        path.write_text(('MURMUR_FOLLOWUP '+json.dumps(record)+'\n')*55)
        result=campaign.followup_metrics(path)
        self.assertEqual(result['record_count'],55);self.assertEqual(result['records_shown'],10)
        self.assertEqual(len(result['records']),10)
        self.assertEqual(len(runner.read_json(Path(result['full_records_path']))['records']),55)
    def test_seed_labeled_followups_are_aggregated_without_losing_records(self):
        path=self.root/'tests.log'
        records=[{'measure':f'switch_seed_{n}','candidate':'a','control':'b','candidate_correct':n,'control_correct':0,'trials':2} for n in (1,2)]
        path.write_text(''.join('MURMUR_FOLLOWUP '+json.dumps(x)+'\n' for x in records))
        result=campaign.followup_metrics(path)
        group=result['aggregates'][0]
        self.assertEqual(group['seeds'],2);self.assertEqual(group['trials'],4)
        self.assertEqual(group['candidate_accuracy'],0.75)
        self.assertEqual(len(runner.read_json(Path(result['full_records_path']))['records']),2)
    def test_reported_usage_includes_diagnostic_and_missing_usage_is_unknown(self):
        item={'agent_usage':{'input_tokens':100},'diagnosis':{'status':'complete','usage':{'input_tokens':30}}}
        self.assertEqual(campaign.reported_input_usage([item]),130)
        item['diagnosis']['usage']=None
        self.assertIsNone(campaign.reported_input_usage([item]))
    def test_reported_budget_stops_between_rounds(self):
        runner.write_json(self.root/'campaign.json',{'schema_version':1,'max_ideas':3,'max_reported_input_tokens':100})
        runner.write_json(self.root/'state.json',{'schema_version':1,'completed':[],'status':'ready'})
        def completed(directory,config,state,agent):
            state['completed'].append({'status':'complete','agent_usage':{'input_tokens':120},'diagnosis':{'status':'not_needed'}})
            return {'index':1,'status':'complete'}
        with patch.object(campaign,'execute_round',side_effect=completed) as execute, patch.object(campaign,'cleanup_completed') as cleanup:
            result=campaign.run(self.root)
        self.assertEqual(execute.call_count,1);self.assertEqual(result['action'],'reported_budget_reached')
        self.assertEqual(result['reported_input_tokens'],120)
        self.assertTrue(cleanup.call_args.kwargs['budget_finished'])
    def test_reported_budget_does_not_assume_missing_usage_is_zero(self):
        runner.write_json(self.root/'campaign.json',{'schema_version':1,'max_ideas':3,'max_reported_input_tokens':100})
        runner.write_json(self.root/'state.json',{'schema_version':1,'completed':[{'status':'complete','agent_usage':None}],'status':'ready'})
        with patch.object(campaign,'execute_round') as execute:
            result=campaign.run(self.root)
        execute.assert_not_called();self.assertEqual(result['action'],'inspect_usage')
        self.assertIsNone(result['reported_input_tokens'])
    def test_budget_finish_cleans_owned_cache_only_after_round_checks(self):
        config={'campaign_id':'test','max_ideas':3};state={'status':'ready','completed':[{'index':1,'status':'complete'}]}
        campaign.build_environment(self.root,config)
        with patch.object(campaign,'cleanup_round',return_value={'index':1,'status':'already_absent'}):
            self.assertEqual(campaign.cleanup_completed(self.root,config,state)['cache'],'retained')
            self.assertEqual(campaign.cleanup_completed(self.root,config,state,budget_finished=True)['cache'],'removed')
        self.assertFalse((self.root/'build-cache').exists())

if __name__=='__main__':unittest.main()

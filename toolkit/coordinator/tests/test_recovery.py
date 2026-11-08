import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from toolkit.coordinator import campaign, runner


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.config={'schema_version':1,'max_ideas':3,'max_recoveries':1}
        runner.write_json(self.root/'campaign.json',self.config)
        runner.write_json(self.root/'state.json',{'schema_version':1,'completed':[],'status':'ready'})
    def tearDown(self):self.tmp.cleanup()
    def run_fake(self, eligible, budget=None, limit=None):
        if budget is not None:
            self.config['max_reported_input_tokens']=budget
            runner.write_json(self.root/'campaign.json',self.config)
        def attempt(directory,config,state,agent):
            index=len(state['completed'])+1
            status='failed' if index==1 else 'complete'
            item={'index':index,'status':status,'agent_usage':{'input_tokens':50},'diagnosis':{'status':'not_needed'},'recovery_eligible':eligible if index==1 else None}
            state['completed'].append(item);state['status']='inspect_failure' if status=='failed' else 'ready'
            runner.write_json(directory/'state.json',state)
            return item
        with patch.object(campaign,'execute_round',side_effect=attempt), patch.object(campaign,'cleanup_completed'):
            return campaign.run(self.root,max_new=limit)
    def test_eligible_failure_gets_one_bounded_new_attempt(self):
        result=self.run_fake(True)
        self.assertEqual([x['status'] for x in result['rounds']],['failed','complete','complete'])
        state=runner.read_json(self.root/'state.json')
        self.assertEqual(state['recoveries_scheduled'],1)
        self.assertTrue(state['completed'][0]['recovery_scheduled'])
        self.assertEqual(result['action'],'review_results')
    def test_ineligible_failure_never_retries(self):
        result=self.run_fake(False)
        self.assertEqual(len(result['rounds']),1);self.assertEqual(result['action'],'inspect_failure')
    def test_budget_prevents_scheduled_retry(self):
        result=self.run_fake(True,budget=50)
        self.assertEqual(len(result['rounds']),1);self.assertEqual(result['action'],'reported_budget_reached')
    def test_retry_can_resume_on_next_invocation(self):
        result=self.run_fake(True,limit=1)
        self.assertEqual(result['action'],'continue')
        self.assertEqual(runner.read_json(self.root/'state.json')['status'],'ready')
    def test_max_recoveries_zero_disables_retry(self):
        self.config['max_recoveries']=0;runner.write_json(self.root/'campaign.json',self.config)
        self.assertEqual(self.run_fake(True)['action'],'inspect_failure')
    def test_retry_prompt_requires_inspection_and_last_good_code(self):
        prompt=campaign.prompt_for({'goal':'g'},[])
        self.assertIn('retained tests.log and worktree',prompt)
        self.assertIn('last accepted candidate',prompt)

if __name__=='__main__':unittest.main()

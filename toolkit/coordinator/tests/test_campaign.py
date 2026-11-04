import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from toolkit.coordinator import campaign, runner

class CampaignTests(unittest.TestCase):
    def test_seed_roles_are_disjoint_across_rounds(self):
        all_roles = set()
        for index in range(1, 5):
            dev, evaluation = campaign.seed_configs(index)
            p = campaign.make_plan(index, {'question':'q','hypothesis':'h','falsification':'f'})
            a, b = runner.validate_plan(p, dev, evaluation)
            self.assertTrue(all_roles.isdisjoint(a['role_seeds'] | b['role_seeds']))
            all_roles.update(a['role_seeds'] | b['role_seeds'])
            self.assertEqual(p['comparison']['baseline'], 'mixture')
            self.assertEqual(p['comparison']['candidate'], 'candidate')
    def test_usage_sums_completed_turns_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'events.jsonl'
            path.write_text('not-json\n'+json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':2}})+'\n'+json.dumps({'type':'turn.completed','usage':{'input_tokens':5,'output_tokens':3}})+'\n')
            self.assertEqual(campaign.agent_usage(path), {'input_tokens':15,'output_tokens':5})

    def test_scope_rejects_untrusted_evaluator_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with patch.object(campaign, 'git', return_value='base'), patch.object(campaign, 'changed_paths', return_value={'toolkit/src/candidate_context.rs'}):
                with self.assertRaisesRegex(runner.PlanError, 'did not change'):
                    campaign.verify_agent_changes(path, 'base')
            with patch.object(campaign, 'git', return_value='base'), patch.object(campaign, 'changed_paths', return_value={'toolkit/src/agent_candidate.rs', 'toolkit/src/candidate_context.rs'}):
                with self.assertRaisesRegex(runner.PlanError, 'outside candidate scope'):
                    campaign.verify_agent_changes(path, 'base')
    def test_agent_idea_schema_is_strict_and_single_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'idea.json'
            idea={'question':'q','hypothesis':'h','falsification':'f','approach':'a','limitations':'l'}
            path.write_text(json.dumps(idea))
            self.assertEqual(campaign.checked_agent_output(path),idea)
            idea['hypothesis']='two\nlines';path.write_text(json.dumps(idea))
            with self.assertRaises(runner.PlanError):campaign.checked_agent_output(path)
    def test_stops_on_failed_round_and_keeps_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);runner.write_json(path/'campaign.json',{'schema_version':1,'max_ideas':3});runner.write_json(path/'state.json',{'schema_version':1,'completed':[],'status':'ready'})
            def failed(directory,config,state,agent):
                state['completed'].append({'status':'failed'});state['status']='inspect_failure'
                runner.write_json(directory/'state.json',state)
                return {'index':1,'status':'failed','error':'bad candidate'}
            with patch.object(campaign,'execute_round',side_effect=failed):
                result=campaign.run(path,agent=lambda *_:None)
            self.assertEqual(result['action'],'inspect_failure');self.assertEqual(len(result['rounds']),1)
            self.assertEqual(campaign.run(path)['action'],'inspect_failure')
            self.assertFalse((path/'.running.lock').exists())
    def test_prior_history_keeps_full_probe_regressions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner.write_json(root/'state.json', {'completed': [{'index': 1, 'status': 'complete', 'evaluation': {'outcome': 'criteria_met'}}]})
            evidence = root/'idea-001'/'study'/'evaluation'/'evidence'
            evidence.mkdir(parents=True)
            (evidence/'probes.csv').write_text('seed,regime,noise,checkpoint,rule,model,accuracy\n1,aba,0.1,3,1,candidate,0.4\n1,aba,0.1,3,1,mixture,0.9\n')
            history = campaign.prior_history(root)
            self.assertEqual(history[0]['final_probe_matrix']['aba:0.1:rule_1']['candidate'], 0.4)
            self.assertEqual(history[0]['final_probe_matrix']['aba:0.1:rule_1']['mixture'], 0.9)

    def test_concurrent_campaign_run_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'.running.lock').mkdir()
            with self.assertRaisesRegex(runner.PlanError,'locked'):
                campaign.run(path)

if __name__=='__main__':unittest.main()

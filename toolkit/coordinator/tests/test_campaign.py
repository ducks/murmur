import json
import os
from pathlib import Path
import subprocess
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

    def test_cache_is_shared_owned_and_does_not_change_parent_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {'campaign_id': 'test'}
            previous = os.environ.get('CARGO_TARGET_DIR')
            first = campaign.build_environment(root, config)
            second = campaign.build_environment(root, config)
            self.assertEqual(first['CARGO_TARGET_DIR'], second['CARGO_TARGET_DIR'])
            self.assertEqual(os.environ.get('CARGO_TARGET_DIR'), previous)
            with self.assertRaisesRegex(runner.PlanError, 'not owned'):
                campaign.build_environment(root, {'campaign_id': 'other'})
            args = campaign.agent_command({'local_provider': None, 'model': None}, root / 'worktree', root / 'idea-001' / 'idea.json', root / 'schema.json')
            self.assertEqual(args[args.index('--add-dir') + 1], str(root / 'build-cache'))

    def test_cleanup_preserves_changes_and_only_removes_completed_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / 'repo'; repo.mkdir()
            def git(*args):
                return subprocess.run(['git', *args], cwd=repo, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()
            git('init'); git('config', 'user.name', 'Test'); git('config', 'user.email', 'test@example.com')
            (repo / '.gitignore').write_text('/target\n/private\n')
            (repo / 'source').write_text('model\n')
            git('add', '.'); git('commit', '-m', 'candidate')
            commit = git('rev-parse', 'HEAD')
            directory = root / 'campaign'; directory.mkdir()
            round_dir = directory / 'idea-001'; round_dir.mkdir()
            worktree = round_dir / 'worktree'
            branch = 'toolkit/campaign-test-001'
            git('worktree', 'add', '-b', branch, str(worktree))
            result = {'status': 'complete', 'commit': commit, 'branch': branch, 'evaluation': None}
            runner.write_json(round_dir / 'result.json', result)
            study = round_dir / 'study'; stage = study / 'development'; stage.mkdir(parents=True)
            runner.write_json(stage / 'report.json', {'outcome': 'criteria_not_met'})
            runner.write_json(stage / 'status.json', {'state': 'complete', 'outcome': 'criteria_not_met', 'artifacts_sha256': runner.stage_hashes(stage)})
            item = {'index': 1, 'status': 'complete', 'commit': commit}
            config = {'campaign_id': 'test', 'max_ideas': 1}
            state = {'status': 'ready', 'completed': [item]}
            campaign.build_environment(directory, config)
            with patch.object(runner, 'ROOT', repo):
                (stage / 'report.json').write_text('{}')
                self.assertEqual(campaign.cleanup_completed(directory, config, state)['rounds'][0]['status'], 'retained')
                runner.write_json(stage / 'report.json', {'outcome': 'criteria_not_met'})
                (worktree / 'private').write_text('keep')
                self.assertEqual(campaign.cleanup_completed(directory, config, state)['rounds'][0]['status'], 'retained')
                self.assertTrue((directory / 'build-cache').exists())
                (worktree / 'private').unlink()
                (worktree / 'source').write_text('unsaved')
                self.assertEqual(campaign.cleanup_completed(directory, config, state)['rounds'][0]['status'], 'retained')
                git('-C', str(worktree), 'checkout', '--', 'source')
                (worktree / 'target').mkdir()
                (worktree / 'target' / 'build').write_text('generated')
                outcome = campaign.cleanup_completed(directory, config, state)
            self.assertEqual(outcome['rounds'][0]['status'], 'removed')
            self.assertEqual(outcome['cache'], 'removed')
            self.assertFalse(worktree.exists())
            self.assertEqual(git('rev-parse', branch), commit)
            self.assertTrue((stage / 'report.json').exists())

if __name__=='__main__':unittest.main()

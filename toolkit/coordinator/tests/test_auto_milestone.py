import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from toolkit.coordinator import campaign, runner


class AutoMilestoneTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.round = self.root / 'idea-001'
        self.round.mkdir()
        self.worktree = self.round / 'worktree'
        self.worktree.mkdir()
        runner.write_json(self.root / 'milestone-output-schema.json', campaign.MILESTONE_SCHEMA)
        self.config = {'goal': 'learn', 'diagnostic_seconds': 10, 'local_provider': None, 'model': None, 'milestone_schema_version': 2}
        self.milestone = {field: field + ' text' for field in campaign.MILESTONE_V2_FIELDS}
        self.milestone['scope'] = 'test_only'

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_only_planner_freezes_complete_controlled_step(self):
        def process(args, **kwargs):
            self.assertEqual(args[args.index('--sandbox') + 1], 'read-only')
            self.assertIn('explicit control', kwargs['input_bytes'].decode())
            runner.write_json(self.round / 'milestone.json', self.milestone)
            return 0
        with patch.object(campaign, 'bounded_process', side_effect=process):
            milestone = campaign.plan_milestone(self.config, self.worktree, self.round, [], {'agent_context': {}})
        prompt = campaign.prompt_for(self.config, [], {}, milestone)
        self.assertIn('decision_rule text', prompt)
        self.assertIn('do not imply it was completed', prompt)

    def test_invalid_or_failed_planning_stops_before_coding(self):
        def invalid(*args, **kwargs):
            runner.write_json(self.round / 'milestone.json', {'question': 'incomplete'})
            return 0
        with patch.object(campaign, 'bounded_process', side_effect=invalid):
            with self.assertRaises(runner.PlanError):
                campaign.plan_milestone(self.config, self.worktree, self.round, [], {'agent_context': {}})
        with patch.object(campaign, 'bounded_process', return_value=1):
            with self.assertRaises(runner.PlanError):
                campaign.plan_milestone(self.config, self.worktree, self.round, [], {'agent_context': {}})

    def test_test_only_scope_blocks_candidate_changes(self):
        with patch.object(campaign, 'changed_paths', return_value={'toolkit/src/agent_candidate.rs'}):
            with self.assertRaises(runner.PlanError):
                campaign.enforce_milestone_scope(self.worktree, self.milestone)
            campaign.enforce_milestone_scope(self.worktree, {**self.milestone, 'scope': 'model_change'})
        with patch.object(campaign, 'changed_paths', return_value={'toolkit/tests/agent_generated.rs'}):
            campaign.enforce_milestone_scope(self.worktree, self.milestone)

    def test_primary_record_matches_all_labels_beyond_preview(self):
        path = self.round / 'tests.log'
        def record(measure, candidate, control):
            return {'measure': measure, 'candidate': candidate, 'control': control,
                    'candidate_correct': 1, 'control_correct': 0, 'trials': 2}
        rows = [record('unrelated', 'a', 'b')] * 11 + [record(self.milestone['primary_measure'],
                self.milestone['candidate_label'], self.milestone['control_label'])]
        path.write_text(''.join('MURMUR_FOLLOWUP ' + json.dumps(row) + '\n' for row in rows))
        followups = campaign.followup_metrics(path)
        evidence = campaign.assess_milestone_evidence(self.milestone, followups)
        self.assertEqual(evidence['status'], 'primary_record_reported_unverified')
        self.assertEqual(evidence['matching_records'], 1)
        self.assertEqual(campaign.assess_milestone_evidence({**self.milestone, 'control_label': 'other'}, followups)['status'], 'missing_primary_record')

    def test_v1_milestone_still_accepts_original_fields(self):
        old = {field: field + ' text' for field in campaign.MILESTONE_FIELDS}
        def process(*args, **kwargs):
            runner.write_json(self.round / 'milestone.json', old)
            return 0
        with patch.object(campaign, 'bounded_process', side_effect=process):
            result = campaign.plan_milestone({**self.config, 'milestone_schema_version': 1}, self.worktree, self.round, [], {'agent_context': {}})
        self.assertEqual(result, old)

    def test_planner_usage_is_included_and_missing_is_unknown(self):
        item = {'agent_usage': {'input_tokens': 4}, 'milestone': self.milestone,
                'milestone_usage': {'input_tokens': 3}, 'diagnosis': {'status': 'not_needed'}}
        self.assertEqual(campaign.reported_input_usage([item]), 7)
        item['milestone_usage'] = None
        self.assertIsNone(campaign.reported_input_usage([item]))

    def test_memory_preserves_milestone_and_its_unverified_status(self):
        from toolkit.coordinator import memory
        runner.write_json(self.root / 'campaign.json', {'schema_version': 1, 'campaign_id': 'id'})
        evidence = {'status': 'reported_unverified'}
        runner.write_json(self.root / 'state.json', {'schema_version': 1, 'completed': [
            {'index': 1, 'status': 'complete', 'milestone': self.milestone, 'milestone_evidence': evidence}]})
        result = memory.campaign_records(self.root)[0]
        self.assertEqual(result['milestone'], self.milestone)
        self.assertEqual(result['milestone_evidence'], evidence)


if __name__ == '__main__':
    unittest.main()

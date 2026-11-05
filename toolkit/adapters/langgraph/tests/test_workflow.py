import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from toolkit.coordinator import runner
from toolkit.adapters.langgraph import workflow

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.path = Path(self.tmp.name)
        runner.write_json(self.path/'campaign.json', {'schema_version':1,'max_ideas':2})
        runner.write_json(self.path/'state.json', {'schema_version':1,'status':'ready','completed':[]})
        self.index = {'campaign_rounds':[{'id':'x'}],'findings':[{'id':'f'}],'warnings':[]}
    def tearDown(self): self.tmp.cleanup()
    def test_preview_is_read_only_and_does_not_run_agent(self):
        with patch.object(workflow.memory,'build_index',return_value=self.index), patch.object(workflow.campaign,'run') as run:
            result=workflow.build_graph().invoke({'directory':str(self.path),'mode':'preview'})
        run.assert_not_called();self.assertEqual(result['action'],'continue');self.assertEqual(result['findings'],1)
    def test_execute_calls_one_bounded_iteration_and_reinspects(self):
        def fake_run(directory, max_new):
            self.assertEqual(max_new,1)
            runner.write_json(self.path/'state.json', {'schema_version':1,'status':'ready','completed':[{}]})
            return {'action':'continue','completed':1}
        with patch.object(workflow.memory,'build_index',return_value=self.index), patch.object(workflow.campaign,'run',side_effect=fake_run) as run:
            result=workflow.build_graph().invoke({'directory':str(self.path),'mode':'execute'})
        run.assert_called_once();self.assertEqual(result['completed'],1);self.assertEqual(result['campaign_result']['completed'],1)
    def test_failed_campaign_never_runs_agent(self):
        runner.write_json(self.path/'state.json', {'schema_version':1,'status':'inspect_failure','completed':[{}]})
        with patch.object(workflow.memory,'build_index',return_value=self.index), patch.object(workflow.campaign,'run') as run:
            result=workflow.build_graph().invoke({'directory':str(self.path),'mode':'execute'})
        run.assert_not_called();self.assertEqual(result['action'],'inspect_failure')

if __name__=='__main__': unittest.main()

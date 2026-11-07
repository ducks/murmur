import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from toolkit.coordinator import provenance, runner


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        summary=runner.read_json(runner.ROOT/'toolkit/experiment-results/automatic-replay-pilot-01/campaign-summary.json')
        self.config={'base_commit':summary['base_commit'],'seed_candidate_commit':summary['seed_candidate_commit']}
        self.start=summary['round']['base_commit'];self.candidate=summary['round']['commit']
    def tearDown(self):self.tmp.cleanup()
    def test_real_seed_transplant_is_distinct_from_agent_delta(self):
        p=provenance.record(self.config,1,self.start,self.candidate,self.root)
        self.assertEqual(p['seed_transplant']['source_commit'],self.config['seed_candidate_commit'])
        self.assertEqual(p['candidate_delta']['from_commit'],self.start)
        self.assertNotIn('toolkit/src/agent_candidate.rs',p['candidate_delta']['changed_paths'])
        self.assertIn('toolkit/tests/agent_generated.rs',p['candidate_delta']['agent_editable_paths'])
        self.assertIn('toolkit/configs/agent/dev.cfg',p['candidate_delta']['supervisor_generated_paths'])
        self.assertNotIn('toolkit/configs/agent/dev.cfg',p['candidate_delta']['agent_editable_paths'])
        for path,key in [('seed-transplant.patch','seed_transplant'),('candidate-changes.patch','candidate_delta')]:
            self.assertEqual(hashlib.sha256((self.root/path).read_bytes()).hexdigest(),p[key]['patch_sha256'])
        self.assertEqual(json.loads((self.root/'provenance.json').read_text()),p)
    def test_wrong_start_or_source_is_rejected(self):
        with self.assertRaises(runner.PlanError):provenance.record(self.config,1,self.config['base_commit'],self.candidate,self.root)
        wrong={**self.config,'seed_candidate_commit':self.config['base_commit']}
        with self.assertRaises(runner.PlanError):provenance.record(wrong,1,self.start,self.candidate,self.root)
        with self.assertRaises(runner.PlanError):provenance.exact_commit('short')
    def test_supervisor_generated_file_tampering_is_detected(self):
        from toolkit.coordinator import campaign
        path=self.root/'generated';path.write_text('expected')
        hashes={'generated':hashlib.sha256(path.read_bytes()).hexdigest()}
        campaign.verify_generated_files(self.root,hashes)
        path.write_text('changed')
        with self.assertRaises(runner.PlanError):campaign.verify_generated_files(self.root,hashes)
        path.unlink()
        with self.assertRaises(runner.PlanError):campaign.verify_generated_files(self.root,hashes)

    def test_reviewer_is_pointed_at_agent_delta(self):
        from toolkit.coordinator import campaign
        prompt=campaign.diagnosis_prompt({},self.root,True)
        self.assertIn('round_start_commit to candidate_commit',prompt)
        self.assertIn('unrelated campaign history',prompt)
        self.assertIn('not agent scope violations',prompt)

if __name__=='__main__':unittest.main()

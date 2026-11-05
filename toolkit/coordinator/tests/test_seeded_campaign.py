from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from toolkit.coordinator import campaign, runner

class SeededCampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();root=Path(self.tmp.name);self.repo=root/'repo';self.repo.mkdir()
        self.worktree=root/'worktree'
        self.git('init');self.git('config','user.name','Test');self.git('config','user.email','test@example.com')
        (self.repo/'toolkit/src').mkdir(parents=True);(self.repo/'toolkit/tests').mkdir(parents=True)
        (self.repo/'toolkit/src/agent_candidate.rs').write_text('seed candidate\n')
        (self.repo/'toolkit/tests/agent_generated.rs').write_text('#[test] fn seed_test() {}\n')
        (self.repo/'unrelated').write_text('keep baseline\n')
        self.git('add','.');self.git('commit','-m','candidate')
        self.seed=self.git('rev-parse','HEAD')
        (self.repo/'toolkit/src/agent_candidate.rs').write_text('current baseline\n')
        (self.repo/'unrelated').write_text('newer evaluator\n')
        self.git('add','.');self.git('commit','-m','current')
        self.base=self.git('rev-parse','HEAD')
        self.git('worktree','add','-b','seed-test',str(self.worktree),self.base)
    def tearDown(self):self.tmp.cleanup()
    def git(self,*args):
        return subprocess.run(['git',*args],cwd=self.repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,text=True).stdout.strip()
    def test_seed_preserves_current_evaluator_and_makes_auditable_commit(self):
        with patch.object(runner,'ROOT',self.repo):
            commit,hashes=campaign.resolve_seed_candidate(self.seed[:12])
            seeded=campaign.apply_seed_candidate(self.worktree,{'seed_candidate_commit':commit,'seed_source_sha256':hashes},self.base)
            self.assertNotEqual(seeded,self.base)
            self.assertEqual((self.worktree/'toolkit/src/agent_candidate.rs').read_text(),'seed candidate\n')
            self.assertEqual((self.worktree/'unrelated').read_text(),'newer evaluator\n')
            self.assertEqual(self.git('-C',str(self.worktree),'status','--porcelain'),'')
            (self.worktree/'toolkit/tests/agent_generated.rs').write_text('#[test] fn followup_test() {}\n')
            campaign.verify_agent_changes(self.worktree,seeded)  # test-only investigation is permitted
    def test_wrong_hash_rejected_before_writing(self):
        with patch.object(runner,'ROOT',self.repo):
            with self.assertRaisesRegex(runner.PlanError,'frozen hashes'):
                campaign.apply_seed_candidate(self.worktree,{'seed_candidate_commit':self.seed,'seed_source_sha256':{}},self.base)
        self.assertEqual((self.worktree/'toolkit/src/agent_candidate.rs').read_text(),'current baseline\n')
    def test_invalid_revision_is_rejected(self):
        with self.assertRaisesRegex(runner.PlanError,'hexadecimal'):
            campaign.resolve_seed_candidate('--bad-revision')

if __name__=='__main__':unittest.main()

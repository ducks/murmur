import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from toolkit.coordinator import campaign, replay, runner


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        source=runner.ROOT/'toolkit/experiment-results/seed-allocation-pilot-01'
        self.archive=self.root/'archive';shutil.copytree(source,self.archive)
    def tearDown(self):self.tmp.cleanup()
    def test_valid_archive_has_100_allocated_and_paired_rows(self):
        item, allocation, records=replay.verify_archive(self.archive)
        self.assertEqual(item['commit'],'653e854a969357c0a0a595d292d6ef0fc1c5882b')
        self.assertEqual(len(allocation['rows']),100);self.assertEqual(len(records['records']),100)
    def test_tampered_code_and_records_are_rejected(self):
        (self.archive/'candidate.rs').write_text('tampered')
        with self.assertRaises(runner.PlanError):replay.verify_archive(self.archive)
        shutil.copyfile(runner.ROOT/'toolkit/experiment-results/seed-allocation-pilot-01/candidate.rs',self.archive/'candidate.rs')
        with gzip.open(self.archive/'paired-records.json.gz','wb') as f:f.write(b'{}')
        with self.assertRaises(runner.PlanError):replay.verify_archive(self.archive)
    def test_failed_test_retains_evidence_and_does_not_claim_reproduction(self):
        output=self.root/'run'
        archive_data=replay.verify_archive(self.archive)
        with patch.object(replay,'verify_archive',return_value=archive_data), patch.object(campaign,'git',return_value=''), patch.object(campaign,'bounded_process',return_value=1):
            result=replay.replay(self.archive,output)
        self.assertEqual(result['status'],'failed');self.assertIn('tests failed',result['error'])
        self.assertTrue((output/'build-cache').exists());self.assertTrue((output/'tests.log').exists())
        self.assertEqual(json.loads((output/'result.json').read_text())['status'],'failed')

if __name__=='__main__':unittest.main()

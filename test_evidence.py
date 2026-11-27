import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evidence
from registry import digest

CID='a'*64
ART='b'*64
EV='c'*64
AD='d'*64

class Registry:
    def verify(self,cid):return True
    def contract(self,cid):return PLAN

PLAN={'contract_id':CID,'inputs':[{'role':'evaluator','sha256':EV,'path':'test.pyz'}]}
SOURCE={'schema':evidence.SOURCE,'contract':PLAN,'artifact_sha256':ART,'sessions':[{'sample':1}]}
REPORT={'evaluator_sha256':EV,'adapter_sha256':AD,'requests':8}

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=evidence.Store(Path(self.tmp.name)/'evidence.sqlite')
        self.reg=Registry()
        self.patches=[patch.object(evidence,'snapshot',return_value=SOURCE),patch.object(evidence,'native_report',return_value=REPORT),patch.object(evidence,'validate_source',return_value=[]),patch.object(evidence,'adapter_hash',return_value=AD)]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in self.patches:p.stop()
        self.store.close()
        self.tmp.cleanup()
    def test_idempotent(self):
        a=self.store.ingest(self.reg,CID,'unused')
        b=self.store.ingest(self.reg,CID,'unused')
        self.assertEqual(a,b)
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM blobs').fetchone()[0],2)
        self.assertEqual(len(self.store.find(artifact=ART)),1)
    def test_tamper(self):
        a=self.store.ingest(self.reg,CID,'unused')
        with self.store.db:self.store.db.execute('UPDATE blobs SET body=? WHERE id=?',(b'{}',a['source_id']))
        with self.assertRaisesRegex(ValueError,'blob hash'):self.store.get(a['source_id'])
    def test_replay_exact(self):
        a=self.store.ingest(self.reg,CID,'unused')
        result=self.store.replay(self.reg,CID)
        self.assertEqual(result['source_id'],a['source_id'])
        self.assertEqual(result['new_inference_requests'],0)
        with self.assertRaisesRegex(ValueError,'No exact'):self.store.replay(self.reg,'e'*64)
        with patch.object(evidence,'adapter_hash',return_value='f'*64):
            with self.assertRaisesRegex(ValueError,'No exact'):self.store.replay(self.reg,CID)
    def test_ambiguous(self):
        a=self.store.ingest(self.reg,CID,'unused')
        with patch.object(evidence,'snapshot',return_value={**SOURCE,'sessions':[{'sample':2}]}):
            b=self.store.ingest(self.reg,CID,'unused')
        self.assertNotEqual(a['source_id'],b['source_id'])
        with self.assertRaisesRegex(ValueError,'Multiple historical'):self.store.replay(self.reg,CID)
        self.assertEqual(self.store.replay(self.reg,CID,a['source_id'])['source_id'],a['source_id'])
    def test_report_disagreement(self):
        p=Path(self.tmp.name)/'report.json';p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Supplied report differs'):self.store.ingest(self.reg,CID,'unused',p)
        self.assertEqual(self.store.find(),[])
    def test_index_disagreement(self):
        a=self.store.ingest(self.reg,CID,'unused')
        with self.store.db:self.store.db.execute('UPDATE sources SET artifact_sha256=?',('f'*64,))
        with self.assertRaisesRegex(ValueError,'index mismatch'):self.store.find()
if __name__=='__main__':unittest.main()

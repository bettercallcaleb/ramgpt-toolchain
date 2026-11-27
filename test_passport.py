import copy
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evidence
import passport
import registry

class PassportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.reg=registry.Registry(self.root/'registry.sqlite')
        self.store=evidence.Store(self.root/'evidence.sqlite')
        self.model=self.root/'test.gguf'
        self.model.write_bytes(b'GGUF'+struct.pack('<IQQ',3,1,0)+b'fixture')
        self.aid=self.reg.register(self.model,'model')['sha256']
        self.doc=passport.build(self.reg,self.store,self.aid)
    def tearDown(self):
        self.store.close();self.reg.close();self.tmp.cleanup()
    def reseal(self,d):
        d['passport_id']=registry.digest(d['passport'])
        return d
    def test_deterministic_snapshot(self):
        self.assertEqual(self.doc,passport.build(self.reg,self.store,self.aid))
        self.assertIn('fidelity',self.doc['passport']['missing_axes'])
        self.assertFalse(self.doc['passport']['deployment']['authorized'])
        r=passport.verify_local(self.reg,self.store,self.doc,self.doc['passport_id'],self.model)
        self.assertTrue(r['local_snapshot_matches'])
        self.assertTrue(r['artifact_bytes_verified'])
        self.assertFalse(r['authenticity_verified'])
    def test_tampering(self):
        d=copy.deepcopy(self.doc);d['passport']['artifact']['bytes']+=1
        with self.assertRaises(ValueError):passport.verify(d)
        with self.assertRaises(ValueError):passport.verify(self.reseal(d),self.doc['passport_id'])
        with self.assertRaises(ValueError):passport.verify_local(self.reg,self.store,d)
    def test_reject_claim(self):
        d=copy.deepcopy(self.doc);d['passport']['deployment']['authorized']=True
        with self.assertRaises(ValueError):passport.verify(self.reseal(d))
    def test_private_paths(self):
        for p in ['/home/private/file','C:\\private\\file','~/private','GPU-00000000-0000-4000-8000-000000000001','file:///private']:
            with self.assertRaises(ValueError):passport.no_private({'field':p})
    def test_model_mismatch(self):
        self.model.write_bytes(self.model.read_bytes()+b'x')
        with self.assertRaises(ValueError):passport.verify_model(self.doc,model=self.model)
    def test_export_no_overwrite(self):
        out=self.root/'model.passport.json'
        passport.export(self.reg,self.store,self.aid,out)
        self.assertEqual(registry.read_json(out),self.doc)
        with self.assertRaises(FileExistsError):passport.export(self.reg,self.store,self.aid,out)
    def test_unknown_artifact(self):
        with self.assertRaises(ValueError):passport.build(self.reg,self.store,'a'*64)

if __name__=='__main__':unittest.main()

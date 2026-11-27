import copy
import hashlib
import struct
import tempfile
import unittest
from pathlib import Path
import evidence
import registry
import native_tools as nt
import native_bridge as nb
import model_passport as m

class ConsolidatedTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        self.reg=registry.Registry(self.p/'r.sqlite');self.store=evidence.Store(self.p/'e.sqlite')
    def tearDown(self):
        self.reg.close();self.store.close();self.tmp.cleanup()
    def test_gguf_native_summary_and_local_verify(self):
        p=self.p/'model.gguf';p.write_bytes(b'GGUF'+struct.pack('<IQQ',3,1,0)+b'fixture')
        a=self.reg.register(p,'model')
        nt.save(self.store,'conceptladder',a['sha256'],{'artifact_bytes':a['bytes'],'result':{'track':'ifeval','n':120,'strict_prompt':.8,'complete':True,'private_raw':'/home/private'}})
        packet=m.build(self.reg,self.store,a['sha256'])
        self.assertEqual(len(packet['passport']['native_evidence']),1)
        self.assertNotIn('native_ifeval',packet['passport']['missing_axes'])
        self.assertTrue(m.verify_model(packet,model=p,reg=self.reg,store=self.store)['artifact_bytes_verified'])
        self.assertNotIn('private_raw',str(packet))
        bad=copy.deepcopy(packet);bad['passport']['native_evidence'][0]['artifact_sha256']='a'*64;bad['passport_id']=registry.digest(bad['passport'])
        with self.assertRaisesRegex(ValueError,'Native artifact mismatch'):m.verify(bad)
    def test_exl3_manifest_and_tamper(self):
        p=self.p/'exl';p.mkdir()
        (p/'config.json').write_bytes(b'{}');(p/'model.safetensors').write_bytes(b'fixture')
        files={x.name:{'size':x.stat().st_size,'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in p.iterdir()}
        aid=registry.digest({'format':'exl3','files':files})
        nb.save(self.store,'exl3-first-mover',{'format':'exl3','artifact_id':aid,'repo_id':None,'source_revision':None},{'files':files,'artifact_bytes':sum(v['size'] for v in files.values()),'declared_bits':4})
        packet=m.build(self.reg,self.store,'exl3:'+aid)
        self.assertTrue(m.verify_model(packet,model=p,reg=self.reg,store=self.store)['artifact_bytes_verified'])
        (p/'README.md').write_text('new card')
        self.assertTrue(m.verify_model(packet,model=p)['artifact_bytes_verified'])
        (p/'model.safetensors').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'files differ'):m.verify_model(packet,model=p)
    def test_reject_deployment_and_hash(self):
        p=self.p/'model.gguf';p.write_bytes(b'GGUF'+struct.pack('<IQQ',3,1,0)+b'fixture')
        a=self.reg.register(p,'model');packet=m.build(self.reg,self.store,a['sha256'])
        bad=copy.deepcopy(packet);bad['passport']['deployment_authorized']=True;bad['passport_id']=registry.digest(bad['passport'])
        with self.assertRaises(ValueError):m.verify(bad)
        with self.assertRaises(ValueError):m.verify(packet,'a'*64)

if __name__=='__main__':unittest.main()

import copy
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from registry import Registry, fingerprint, read_json

class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.reg=Registry(self.root/'db.sqlite')
        self.model=self.root/'model.gguf'
        self.model.write_bytes(b'GGUF'+struct.pack('<IQQ',3,1,0)+b'fixture')
    def tearDown(self):
        self.reg.close()
        self.tmp.cleanup()
    def spec(self):
        inputs=[{'role':'model','path':str(self.model)}]
        for role in ('runtime','dataset','template','evaluator'):
            p=self.root/role;p.write_text(role)
            inputs.append({'role':role,'path':str(p)})
        return {'inputs':inputs,'execution':{'runtime_args':['-c','4096'],'gpu_uuids':[],'sampling':{'temperature':0},'reasoning_policy':'off'},'workload':{'protocol':'fixture-v1','seed':1,'budgets':[256]}}
    def test_identical_files(self):
        a=self.reg.register(self.model,'model')
        b=self.root/'renamed.gguf';shutil.copyfile(self.model,b)
        self.assertEqual(a['sha256'],self.reg.register(b,'model')['sha256'])
        self.assertEqual(len(self.reg.get(a['sha256'])['observations']),2)
    def test_same_filename_changed_bytes(self):
        a=self.reg.register(self.model,'model')
        self.model.write_bytes(self.model.read_bytes()+b'changed')
        self.assertNotEqual(a['sha256'],self.reg.register(self.model,'model')['sha256'])
    def test_freeze_and_verify(self):
        spec=self.spec();a=self.reg.freeze(spec)
        self.assertEqual(a,self.reg.freeze(spec))
        self.assertEqual(self.reg.verify(a['contract_id'])['status'],'inputs_verified')
        spec['workload']['seed']=2
        self.assertNotEqual(a['contract_id'],self.reg.freeze(spec)['contract_id'])
        (self.root/'template').write_text('changed')
        with self.assertRaises(ValueError):self.reg.verify(a['contract_id'])
    def test_invalid(self):
        bad=self.root/'bad.gguf';bad.write_bytes(b'not gguf')
        with self.assertRaises(ValueError):fingerprint(bad,'model')
        shard=self.root/'model-00001-of-00002.gguf';shutil.copyfile(self.model,shard)
        with self.assertRaises(ValueError):fingerprint(shard,'model')
        spec=self.spec();spec['inputs'].pop()
        with self.assertRaises(ValueError):self.reg.freeze(spec)
    def test_duplicate_json(self):
        p=self.root/'bad.json';p.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError):read_json(p)
    def test_contract_integrity(self):
        a=self.reg.freeze(self.spec())
        with self.reg.db:self.reg.db.execute('UPDATE contracts SET body=? WHERE id=?',('{}',a['contract_id']))
        with self.assertRaises(ValueError):self.reg.contract(a['contract_id'])

if __name__=='__main__':unittest.main()

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evidence
import native_bridge as b

GPU='GPU-00000000-0000-4000-8000-000000000001'

class Exl3BuildTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.s=evidence.Store(self.root/'e.sqlite')
        self.native=self.root/'fm.py'
        self.native.write_text('import hashlib\nCFG={}\ndef architecture_supported(arch):return arch=="Fixture"\ndef manifest_dir(root):return {str(p.relative_to(root)):{"size":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.rglob("*") if p.is_file() and p.name!="README.md"}\n')
        self.source=self.root/'source';self.source.mkdir()
        (self.source/'config.json').write_text(json.dumps({'architectures':['Fixture']}))
        (self.source/'model.safetensors').write_bytes(b'fixture-source')
        self.backend=self.root/'backend';(self.backend/'exllamav3').mkdir(parents=True)
        (self.backend/'exllamav3/__init__.py').write_text('pass')
        (self.backend/'convert.py').write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
for name,short in [('in_dir','i'),('out_dir','o'),('work_dir','w'),('bits','b'),('devices','d')]:p.add_argument('-'+short,'--'+name,required=True)
a=p.parse_args()
out=Path(a.out_dir)
(out/'config.json').write_text(json.dumps({'quantization_config':{'quant_method':'exl3'}}))
(out/'model.safetensors').write_bytes(b'fixture-converted')
''')
    def tearDown(self):
        self.s.close();self.tmp.cleanup()
    def build(self,execute=True):
        return b.exl3_build(self.s,self.source,self.root/'result',self.native,self.backend,sys.executable,GPU,4.,execute)
    def test_no_execution_without_flag(self):
        with self.assertRaisesRegex(ValueError,'Explicit --execute'):self.build(False)
        self.assertFalse((self.root/'result').exists())
    def test_official_entrypoint_handoff_fixture(self):
        with patch('runtime.lease',return_value=contextlib.nullcontext()),patch('runtime.gpu_state',return_value={GPU:{'used_mib':16}}),patch('runtime.compute_apps',return_value=[]),patch('shutil.disk_usage') as disk:
            disk.return_value.free=40*2**30
            result=self.build()
        self.assertEqual(result['subject']['format'],'exl3')
        self.assertIsNone(result['subject']['repo_id'])
        self.assertEqual(len(b.find(self.s,'exl3-first-mover')),2)
        self.assertFalse(result['deployment_authorized'])
        with self.assertRaisesRegex(ValueError,'fresh output'):self.build()

if __name__=='__main__':unittest.main()

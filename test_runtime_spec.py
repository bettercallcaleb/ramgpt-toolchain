import json
import struct
import tempfile
import unittest
from pathlib import Path
from registry import fingerprint
import runtime_spec as s

GPU='GPU-00000000-0000-4000-8000-000000000001'

class RuntimeSpecTests(unittest.TestCase):
    def test_requires_explicit_execution(self):
        with self.assertRaisesRegex(ValueError,'Explicit --execute'):
            s.capture('missing','missing','missing',GPU,'missing')
    def test_default_runtime_options(self):
        e=s.execution('/tmp/model.gguf',GPU,18429,4096,8,999)
        self.assertEqual(e['reasoning_policy'],'off')
        self.assertEqual(e['sampling']['temperature'],0)
        self.assertIn('--no-warmup',e['runtime_args'])
        self.assertEqual(e['runtime_args'][e['runtime_args'].index('--host')+1],'127.0.0.1')
        with self.assertRaises(ValueError):s.execution('model','invalid',18429,4096,8,999)
        with self.assertRaises(ValueError):s.execution('model',GPU,80,4096,8,999)
    def test_capture_binding_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);model=p/'model.gguf';model.write_bytes(b'GGUF'+struct.pack('<IQQ',3,1,0)+b'fixture')
            template=p/'template.json';template.write_text(json.dumps({'prompt':'sample'}))
            refs=[fingerprint(model,'model'),fingerprint(template,'template')]
            spec={'inputs':[{k:r[k] for k in ('role','path')} for r in refs],'execution':{'template_capture':{'inputs':sorted([{k:r[k] for k in ('role','sha256','bytes')} for r in refs],key=lambda r:r['role'])}}}
            s.check_binding(spec)
            template.write_text(json.dumps({'prompt':'changed'}))
            with self.assertRaisesRegex(ValueError,'Captured spec inputs changed'):s.check_binding(spec)
            s.check_binding({'execution':{}})

if __name__=='__main__':unittest.main()

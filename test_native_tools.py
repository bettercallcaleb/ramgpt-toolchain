import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evidence
import native_tools as n

class NativeToolTests(unittest.TestCase):
    def test_receipts_join_exact_artifact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=evidence.Store(Path(tmp)/'e.sqlite')
            try:
                a=n.save(s,'gguf-healer','a'*64,{'result':{'verified_payloads':2}})
                b=n.save(s,'conceptladder','a'*64,{'result':{'n':60}})
                self.assertEqual(a,n.save(s,'gguf-healer','a'*64,{'result':{'verified_payloads':2}}))
                self.assertEqual({r['receipt_id'] for r in n.find(s,'a'*64)},{a['receipt_id'],b['receipt_id']})
                self.assertEqual(n.find(s,'b'*64),[])
                self.assertFalse(a['deployment_authorized'])
            finally:s.close()
    def test_index_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=evidence.Store(Path(tmp)/'e.sqlite')
            try:
                r=n.save(s,'conceptladder','a'*64,{'result':{}})
                with s.db:s.db.execute('UPDATE native_tool_receipts SET artifact_sha256=? WHERE id=?',('b'*64,r['receipt_id']))
                with self.assertRaisesRegex(ValueError,'index mismatch'):n.find(s,'b'*64)
            finally:s.close()
    def test_native_tree_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);f=p/'mod.py';f.write_text('x=1')
            a=n.tree(p);f.write_text('x=2')
            self.assertNotEqual(a,n.tree(p))
    def test_invalid_growth_rejected_before_inputs(self):
        for v in (-1,float('nan'),float('inf'),2):
            with self.assertRaises(ValueError):n.healer_import(None,None,'missing','missing','missing',v)
    def test_compose_needs_explicit_execution(self):
        with self.assertRaisesRegex(ValueError,'Explicit --execute'):
            n.healer_compose(None,None,'missing','missing','missing','missing','missing',['tensor'],.05)
    @unittest.skipUnless(os.getenv('HEALER_TEST_SCRIPT') and os.getenv('GGUF_TEST_ROOT'),'Native Healer fixture dependencies required')
    def test_native_compose_fixture(self):
        import sys
        import numpy as np
        import registry
        root=Path(os.environ['GGUF_TEST_ROOT'])
        sys.path.insert(0,str(root))
        try:
            from gguf import GGUFWriter
            with tempfile.TemporaryDirectory() as tmp:
                p=Path(tmp)
                for name,dtype in [('base.gguf',np.float16),('donor.gguf',np.float32)]:
                    w=GGUFWriter(p/name,'llama')
                    for t in ('output.weight','other.weight'):
                        w.add_tensor(t,np.arange(128,dtype=dtype).reshape(8,16))
                    w.write_header_to_file();w.write_kv_data_to_file();w.write_tensors_to_file();w.close()
                reg=registry.Registry(p/'reg.sqlite');store=evidence.Store(p/'ev.sqlite')
                try:
                    r=n.healer_compose(reg,store,p/'base.gguf',p/'donor.gguf',p/'new.gguf',os.environ['HEALER_TEST_SCRIPT'],root,['output.weight'],.5,True)
                    self.assertEqual(r['result']['verified_payloads'],2)
                    self.assertEqual(n.find(store,r['artifact_sha256']),[r])
                    with self.assertRaises(ValueError):
                        n.healer_compose(reg,store,p/'base.gguf',p/'donor.gguf',p/'new.gguf',os.environ['HEALER_TEST_SCRIPT'],root,['output.weight'],.5,True)
                finally:reg.close();store.close()
        finally:sys.path.remove(str(root))
    def test_worker_isolated_and_json(self):
        self.assertEqual(n.worker('import json,sys;print(json.dumps(sys.argv[1:]))',['x']),['x'])
        with self.assertRaisesRegex(ValueError,'Native tool failed'):n.worker('raise ValueError("bad")',[])

if __name__=='__main__':unittest.main()

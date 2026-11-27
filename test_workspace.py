import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import evidence
import registry
import workspace
import native_tools
import native_bridge

class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        self.reg=registry.Registry(self.p/'reg.sqlite');self.store=evidence.Store(self.p/'ev.sqlite')
    def tearDown(self):
        self.reg.close();self.store.close();self.tmp.cleanup()
    def test_config_paths_and_override(self):
        p=self.p/'tools.json'
        (self.p/'q4fit.py').write_text('pass')
        p.write_text(json.dumps({'tools':{'q4fit':{'native':'q4fit.py','gguf_root':'.','quantizer':'missing'}}}))
        cfg=workspace.load(p)
        self.assertEqual(cfg['q4fit']['native'],str(self.p/'q4fit.py'))
        args=SimpleNamespace(action='build',native='explicit',quantizer=None,gguf_root=None)
        workspace.apply(args,cfg)
        self.assertEqual(args.native,'explicit')
        self.assertEqual(args.gguf_root,str(self.p))
        self.assertFalse(workspace.catalog(cfg)['tools'][0]['configured'])
        self.assertEqual(len(workspace.catalog(cfg)['tools']),9)
    def test_q4fit_evidence_separate_from_builder(self):
        cfg={'q4fit':{'native':'q4fit.py','evidence_script':'q4evidence.py'}}
        args=SimpleNamespace(action='weight-import',kind='q4fit',native=None)
        workspace.apply(args,cfg)
        self.assertEqual(args.native,'q4evidence.py')
        args.native=None
        with self.assertRaisesRegex(ValueError,'q4fit.evidence_script'):workspace.apply(args,{'q4fit':{'native':'q4fit.py'}})
    def test_unknown_tool_or_field(self):
        p=self.p/'tools.json'
        for tools in ({'unknown':{}},{'q4fit':{'publish':True}}):
            p.write_text(json.dumps({'tools':tools}))
            with self.assertRaises(ValueError):workspace.load(p)
    def test_gguf_lineage_and_receipts(self):
        p=self.p/'model.gguf';p.write_bytes(b'GGUF'+struct.pack('<IQQ',3,1,0)+b'x')
        aid=self.reg.register(p,'model')['sha256']
        native_tools.save(self.store,'q4fit',aid,{'artifact_bytes':p.stat().st_size})
        native_tools.save(self.store,'gguf-healer','b'*64,{'inputs':{'base':{'sha256':aid,'bytes':p.stat().st_size},'output':{'sha256':'b'*64,'bytes':10}}})
        r=workspace.model_info(self.reg,self.store,aid)
        self.assertEqual(r['native_receipts'][0]['tool'],'q4fit')
        self.assertEqual(r['derived_artifacts'][0]['sha256'],'b'*64)
        self.assertEqual(workspace.model_info(self.reg,self.store,str(p)),r)
        self.assertEqual(len(workspace.models(self.reg,self.store)['models']),2)
    def test_exl3_identity_separate(self):
        aid='a'*64
        native_bridge.save(self.store,'exl3-first-mover',{'format':'exl3','artifact_id':aid,'repo_id':'fixture/model','source_revision':'b'*40},{'artifact_bytes':100,'declared_bits':4})
        self.assertEqual(workspace.model_info(self.reg,self.store,'exl3:'+aid)['bytes'],100)
        with self.assertRaises(ValueError):workspace.model_info(self.reg,self.store,aid)
    def test_build_explicit_execution(self):
        with self.assertRaisesRegex(ValueError,'Explicit --execute'):
            native_tools.q4fit_build(None,None,'a','b','c','d','e','Q4_K',[],8)

if __name__=='__main__':unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evidence
import native_bridge as b

class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        self.s=evidence.Store(self.p/'ev.sqlite')
    def tearDown(self):
        self.s.close();self.tmp.cleanup()
    def test_report_lookup_and_idempotence(self):
        r=b.save(self.s,'hereticbench',{'track':'retention'},{'result':{'n':2}})
        self.assertEqual(r,b.save(self.s,'hereticbench',{'track':'retention'},{'result':{'n':2}}))
        self.assertEqual(b.find(self.s,'hereticbench'),[r])
        self.assertEqual(b.find(self.s,'exl3-first-mover'),[])
        self.assertNotIn('artifact_sha256',r)
    def test_native_preflight_gate(self):
        native=self.p/'fm.py';native.write_text('pass')
        for gate,support,expected in [('GREEN',True,True),('BLOCKED',True,False),('GREEN',None,False),('GREEN',False,False)]:
            r={'repo_id':'owner/model','source_revision':'a'*40,'targets':[4.0],'sweet_spot':gate,'exllamav3_arch_supported':support}
            with patch('native_tools.worker',return_value=r):
                receipt=b.exl3_preflight(self.s,'owner/model',native,self.p/'missing')
            self.assertEqual(receipt['conversion_preflight_passed'],expected)
            self.assertFalse(receipt['deployment_authorized'])
    def test_native_config_overrides_rejected(self):
        native=self.p/'fm.py';native.write_text('pass');(self.p/'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'config overrides'):b.exl3_preflight(self.s,'owner/model',native,self.p/'backend')
    def test_exl3_manifest_identity_and_tamper(self):
        native=self.p/'fm.py'
        native.write_text('import hashlib\ndef manifest_dir(root):\n return {str(p.relative_to(root)):{"size":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in root.rglob("*") if p.is_file() and p.name!="README.md"}\n')
        folder=self.p/'artifact';folder.mkdir()
        (folder/'config.json').write_text(json.dumps({'quantization_config':{'quant_method':'exl3','bits':4}}))
        (folder/'model.safetensors').write_bytes(b'fixture')
        import runpy
        m=self.p/'manifest.json'
        m.write_text(json.dumps({'repo_id':'fixture/model','source_revision':'a'*40,'bits':4,'files':runpy.run_path(str(native))['manifest_dir'](folder)}))
        r=b.exl3_import(self.s,folder,m,native)
        self.assertEqual(r['subject']['format'],'exl3')
        self.assertEqual(r['file_count'],2)
        self.assertNotIn('artifact_sha256',r)
        (folder/'model.safetensors').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'manifest mismatch'):b.exl3_import(self.s,folder,m,native)
    def test_native_heretic_replay_contract(self):
        native=self.p/'native';package=native/'hereticbench';package.mkdir(parents=True)
        (package/'__init__.py').write_text('')
        (package/'common.py').write_text('import json\ndef load_jsonl(path):return [json.loads(x) for x in open(path) if x.strip()]\n')
        (package/'grading.py').write_text('def grade_dual(content,grading):\n v=float(content==grading["expected"]);return v,v,{}\n')
        d={'id':'t1','track':'retention','category':'test','messages':[{'role':'user','content':'test'}],'grading':{'expected':'ok'}}
        ds=self.p/'data.jsonl';ds.write_text(json.dumps(d)+'\n')
        from registry import fingerprint
        sha=fingerprint(ds,'configuration')['sha256']
        a=self.p/'a.jsonl';z=self.p/'b.jsonl'
        def row(label,content,score):
            return {**d,'label':label,'status':'ok','dataset_sha256':sha,'content':content,'semantic_score':score,'format_score':score}
        a.write_text(json.dumps(row('base','ok',1))+'\n')
        z.write_text(json.dumps(row('variant','bad',0))+'\n')
        r=b.heretic_import(self.s,native,ds,a,z,'retention')
        self.assertEqual(r['result']['native_grade_parity'],2)
        self.assertEqual(r['result']['tracks']['retention']['base_only'],1)
        z.write_text(json.dumps(row('variant','bad',1))+'\n')
        with self.assertRaisesRegex(ValueError,'grade parity failure'):b.heretic_import(self.s,native,ds,a,z,'retention')
        z.write_text('\n')
        with self.assertRaisesRegex(ValueError,'Incomplete paired'):b.heretic_import(self.s,native,ds,a,z,'retention')

if __name__=='__main__':unittest.main()

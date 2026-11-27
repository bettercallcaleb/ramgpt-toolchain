import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import evidence
import frontdoor as f
import registry

class FrontdoorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.p=Path(self.tmp.name)
        self.reg=registry.Registry(self.p/'r.sqlite');self.store=evidence.Store(self.p/'e.sqlite')
        self.gg=self.p/'model.gguf';self.gg.write_bytes(b'fixture')
        self.ex=self.p/'exl3';self.ex.mkdir()
        self.cfg={'q4fit':{'native':'q4fit.py','quantizer':'llama-quantize','gguf_root':'gguf'},'exl3-first-mover':{'native':'fm.py','backend':'exllama','python':'python'},'budgetfit':{'evaluator':'budgetfit.pyz'}}
    def tearDown(self):
        self.reg.close();self.store.close();self.tmp.cleanup()
    def build_args(self,model,**extra):
        return SimpleNamespace(**dict({'model':str(model),'format':None,'native':None,'quantizer':None,'gguf_root':None,'backend':None,'python':None,'gpu_uuid':None,'bits':None,'default':None,'set':[],'imatrix':None,'threads':None,'out':'new','execute':True},**extra))
    def run_args(self,value,**extra):
        return SimpleNamespace(**dict({'id':str(value),'format':None,'spec':None,'dataset':None,'seed':None,'variants':None,'budgets':None,'strategies':None,'reserve':None,'kv':None,'context':None,'max_requests':None,'max_completion_tokens':None,'native':None,'backend':None,'python':None,'evaluator':None,'gpu_uuid':None,'out':None,'execute':False,'max_new':None,'archive':False,'evidence_db':'db'},**extra))
    def test_build_dispatch(self):
        with patch.object(f.native_tools,'q4fit_build',return_value={'gguf':True}) as g,patch.object(f.native_bridge,'exl3_build',return_value={'exl3':True}) as x:
            self.assertEqual(f.build(self.reg,self.store,self.build_args(self.gg),self.cfg),{'gguf':True})
            self.assertEqual(g.call_args.args[7],'Q4_K')
            self.assertEqual(f.build(self.reg,self.store,self.build_args(self.ex,gpu_uuid='GPU-test'),self.cfg),{'exl3':True})
            self.assertEqual(x.call_args.args[-2],4.0)
            with self.assertRaises(ValueError):f.build(self.reg,self.store,self.build_args(self.gg,bits=4),self.cfg)
            with self.assertRaises(ValueError):f.build(self.reg,self.store,self.build_args(self.ex,default='Q4_K'),self.cfg)
    def test_gguf_plan_then_execute(self):
        spec=self.p/'spec.json';spec.write_text(json.dumps({'inputs':[{'role':'model','path':str(self.gg)}]}))
        args=self.run_args(self.gg,spec=str(spec),seed=821,out='new',budgets=[256],max_requests=9,max_completion_tokens=2112)
        with patch.object(f.measurement_plan,'prepare',return_value={'plan_id':'a'*64}) as prepare,patch.object(f.measurement_plan,'run',return_value={'status':'measurement_complete'}) as run:
            self.assertEqual(f.run(self.reg,self.store,args,self.cfg),{'plan_id':'a'*64})
            run.assert_not_called()
            self.assertEqual(prepare.call_args.args[9],['f16'])
            args.execute=True
            self.assertEqual(f.run(self.reg,self.store,args,self.cfg)['result']['status'],'measurement_complete')
            run.assert_called_once()
        spec.write_text(json.dumps({'inputs':[{'role':'model','path':str(self.ex)}]}))
        with self.assertRaisesRegex(ValueError,'does not match'):f.run(self.reg,self.store,args,self.cfg)
    def test_exl3_run_dispatch(self):
        args=self.run_args(self.ex,dataset='tasks.json',gpu_uuid='GPU-test',out='new',budgets=[256],max_requests=8,max_completion_tokens=2048,execute=True)
        with patch.object(f.cross_format,'run',return_value={'requests':8}) as x:
            self.assertEqual(f.run(self.reg,self.store,args,self.cfg),{'requests':8})
            self.assertEqual(x.call_args.args[3],'budgetfit.pyz')
        args.spec='spec.json'
        with self.assertRaises(ValueError):f.run(self.reg,self.store,args,self.cfg)
    def test_frozen_id_and_archive(self):
        args=self.run_args('a'*64,out='run',archive=True)
        with patch.object(self.reg,'contract',return_value={'workload':{'evaluator':'BudgetFit 0.6.0'}}),patch.object(f.inference,'run',return_value={'status':'complete'}),patch.object(self.store,'ingest',return_value={'evaluation_id':'b'*64}):
            self.assertEqual(f.run(self.reg,self.store,args,self.cfg)['archive']['evaluation_id'],'b'*64)
            args.budgets=[256]
            with self.assertRaisesRegex(ValueError,'frozen run'):f.run(self.reg,self.store,args,self.cfg)

if __name__=='__main__':unittest.main()

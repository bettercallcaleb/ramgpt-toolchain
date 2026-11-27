import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evidence
import inference
import measurement_plan as m
import registry
import runtime
import evaluation

class CostTests(unittest.TestCase):
    def test_includes_smoke(self):
        c=m.cost([{'max_tokens':256},{'max_tokens':1024}],2)
        self.assertEqual(c['total_completion_requests'],6)
        self.assertEqual(c['total_completion_token_cap'],2688)
        m.check_cost(c,6,2688)
        with self.assertRaises(ValueError):m.check_cost(c,5,2688)
        with self.assertRaises(ValueError):m.check_cost(c,6,2687)

    def test_invalid_matrix(self):
        args=([256],['allocated'],64,['f16'],123,1,9,3000)
        m.limits(*args)
        for i,v in [(0,[256,256]),(2,256),(3,['f16','f16']),(4,-1),(5,33),(6,0),(7,0)]:
            bad=list(args);bad[i]=v
            with self.assertRaises(ValueError):m.limits(*bad)

    def test_no_execute_without_confirmation(self):
        with self.assertRaisesRegex(ValueError,'Explicit --execute'):m.run(None,None,'a'*64)

    def test_direct_run_is_blocked(self):
        class Reg:
            def contract(self,cid):return {'workload':{'measurement_cohort_key':'key'}}
        with self.assertRaisesRegex(ValueError,'measurement-run'):inference.run(Reg(),'a'*64,'unused')

@unittest.skipUnless(os.getenv('BUDGETFIT_TEST_ARCHIVE'),'Pinned native archive required')
class MeasurementIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.reg=registry.Registry(self.root/'reg.sqlite')
        self.store=evidence.Store(self.root/'ev.sqlite')
        model=self.root/'fixture.gguf'
        model.write_bytes(b'GGUF'+struct.pack('<IQQ',3,1,0)+b'fixture')
        server=self.root/'llama-server'
        shutil.copy2('/bin/true',server)
        template=self.root/'template.json';template.write_text('{}')
        archive=str(Path(os.environ['BUDGETFIT_TEST_ARCHIVE']).resolve())
        args=['-m',str(model),'--alias','model','--host','127.0.0.1','--port','18089','-c','4096','-np','1','--jinja','--reasoning','off','--no-webui','--temp','0']
        spec={'inputs':[{'role':k,'path':str(v)} for k,v in [('model',model),('runtime',server),('template',template),('evaluator',archive)]],'execution':{'runtime_args':args,'gpu_uuids':['GPU-00000000-0000-4000-8000-000000000001'],'sampling':{'temperature':0},'reasoning_policy':'off'},'workload':{}}
        self.spec=self.root/'spec.json'
        evaluation.write_new(self.spec,spec)
        self.calls=[]

    def tearDown(self):
        self.store.close();self.reg.close();self.tmp.cleanup()

    def prepare(self,out='plan',requests=18,tokens=4224):
        return m.prepare(self.reg,self.store,self.spec,self.root/out,719,1,[256],['allocated'],64,['f16','q8_0'],requests,tokens)

    def fake_session(self,reg,cid,out,timeout=90,work=None):
        out.mkdir()
        ident={'model':'model','system_fingerprint':'synthetic'}
        def send(body,timeout):
            self.calls.append(body)
            return {**ident,'choices':[{'message':{'content':'FINAL: 42'},'finish_reason':'stop'}],'usage':{'prompt_tokens':16,'completion_tokens':4,'total_tokens':20}},.01
        result=work(send,ident)
        r={'contract_id':cid,'status':'runtime_session_passed','response_identity':ident,'workload_result':result,**{k:True for k in ('input_precheck','input_postcheck','smoke_pass','template_probe_match','cleanup_complete')}}
        runtime.atomic(out/'runtime_receipt.json',r)
        return r

    def test_frozen_prepare_execute_archive_idempotent(self):
        d=self.prepare()
        self.assertEqual(d['cost']['total_completion_requests'],18)
        self.assertEqual(d['cost']['total_completion_token_cap'],4224)
        self.assertEqual(d['status'],'frozen_not_executed')
        self.assertEqual(m.load(self.reg,self.store,d['plan_id'])['cost'],d['cost'])
        with self.assertRaisesRegex(ValueError,'already planned'):self.prepare('duplicate')
        with patch.object(runtime,'smoke',side_effect=self.fake_session):
            result=m.run(self.reg,self.store,d['plan_id'],True)
            self.assertEqual(result['new_inference_requests'],16)
            self.assertEqual(result['new_smoke_requests'],2)
            again=m.run(self.reg,self.store,d['plan_id'],True)
        self.assertEqual(again['new_inference_requests'],0)
        self.assertEqual(again['new_smoke_requests'],0)
        self.assertEqual(again['arms'],result['arms'])
        self.assertEqual(len(self.calls),16)
        self.assertEqual(len(self.store.find()),2)

    def test_over_budget_no_output(self):
        with self.assertRaisesRegex(ValueError,'Request cap exceeded'):self.prepare(requests=17)
        self.assertFalse((self.root/'plan').exists())

    def test_tampered_plan_rejected(self):
        d=self.prepare()
        p=self.root/'plan'/'measurement-plan.json'
        b=registry.read_json(p);b['max_requests']+=1;runtime.atomic(p,b)
        with self.assertRaisesRegex(ValueError,'plan file mismatch'):m.load(self.reg,self.store,d['plan_id'])

    def test_started_missing_directory_never_retries(self):
        d=self.prepare()
        with self.store.db:self.store.db.execute('INSERT INTO measurement_started VALUES (?,?)',(d['arms'][0]['contract_id'],d['plan_id']))
        with patch.object(inference,'run') as run:
            with self.assertRaisesRegex(ValueError,'missing bound run directory'):m.run(self.reg,self.store,d['plan_id'],True)
            run.assert_not_called()

if __name__=='__main__':unittest.main()

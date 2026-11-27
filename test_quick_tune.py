import os
import unittest
from unittest.mock import patch
import adaptive
import evaluation
import quick_tune as q
import runtime
import runtime_spec
from registry import fingerprint,read_json
import test_workflow as tw

GPU='GPU-00000000-0000-4000-8000-000000000001'

class GPUChoiceTests(unittest.TestCase):
    def test_unique_idle_or_explicit(self):
        with patch.object(runtime,'gpu_state',return_value={'a':{'used_mib':10},'b':{'used_mib':900}}),patch.object(runtime,'compute_apps',return_value=[]):
            self.assertEqual(q.choose_gpu(None),'a')
            self.assertEqual(q.choose_gpu('b'),'b')
            with self.assertRaisesRegex(ValueError,'Unknown GPU'):q.choose_gpu('c')
        with patch.object(runtime,'gpu_state',return_value={'a':{'used_mib':10},'b':{'used_mib':10}}),patch.object(runtime,'compute_apps',return_value=[]):
            with self.assertRaisesRegex(ValueError,'exactly one'):q.choose_gpu(None)

@unittest.skipUnless(os.getenv('BUDGETFIT_TEST_ARCHIVE'),'Pinned native archive required')
class QuickTuneTests(unittest.TestCase):
    setUp=tw.WorkflowTests.setUp
    tearDown=tw.WorkflowTests.tearDown
    def config(self):
        p=read_json(self.spec)
        inputs={r['role']:r['path'] for r in p['inputs']}
        return {'llama-autotune':{'runtime':inputs['runtime']},'budgetfit':{'evaluator':inputs['evaluator']}}
    def capture(self,model,binary,archive,gpu,folder,port,context,threads,ngl,execute):
        folder.mkdir()
        t=folder/'template.json';evaluation.write_new(t,{'prompt':'test'})
        ex=runtime_spec.execution(model,gpu,port,context,threads,ngl)
        evaluation.write_new(folder/'spec.json',{'inputs':[{'role':r,'path':str(p)} for r,p in [('model',model),('runtime',binary),('evaluator',archive),('template',t)]],'execution':ex,'workload':{}})
    def tune(self,execute=False,max_requests=36,context=4096):
        return q.run(self.reg,self.store,self.root/'fixture.gguf',self.root/'quick',self.config(),GPU,context=context,budgets=[256],seed=719,validation_seed=720,max_requests=max_requests,max_tokens=8448,execute=execute)
    def test_preview_cap_and_capture_once(self):
        with patch.object(q.install_native,'runtime_check',side_effect=lambda b:fingerprint(b,'runtime')),patch.object(q,'choose_gpu',return_value=GPU) as choose,patch.object(runtime_spec,'capture',side_effect=self.capture) as capture,patch.object(q.auto,'run',return_value={'status':'no_holdout'}) as run:
            with self.assertRaisesRegex(ValueError,'Request cap exceeded'):self.tune(max_requests=35)
            choose.assert_not_called()
            p=self.tune()
            self.assertEqual(p['gpu_model_loads'],0)
            self.assertFalse((self.root/'quick').exists())
            capture.assert_not_called()
            run.assert_not_called()
            self.assertEqual(self.tune(True)['template_capture_loads'],1)
            self.assertEqual(self.tune(True)['template_capture_loads'],0)
            capture.assert_called_once()
            with self.assertRaisesRegex(ValueError,'parameters changed'):self.tune(True,context=8192)
    def test_interrupted_capture_no_retry(self):
        def fail(*args):
            args[4].mkdir()
            raise ValueError('capture failed')
        with patch.object(q.install_native,'runtime_check',side_effect=lambda b:fingerprint(b,'runtime')),patch.object(q,'choose_gpu',return_value=GPU),patch.object(runtime_spec,'capture',side_effect=fail) as capture:
            with self.assertRaisesRegex(ValueError,'capture failed'):self.tune(True)
            with self.assertRaisesRegex(ValueError,'Incomplete runtime capture'):self.tune(True)
            capture.assert_called_once()

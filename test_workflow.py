import os
import unittest
from pathlib import Path
from unittest.mock import patch
import inference
import registry
import runtime
import workflow
import passport
import test_measurement_plan as tm
import test_joint as tj

@unittest.skipUnless(os.getenv('BUDGETFIT_TEST_ARCHIVE'),'Pinned native archive required')
class WorkflowTests(unittest.TestCase):
    def setUp(self):
        tm.MeasurementIntegration.setUp(self)
        self.fail_holdout=False

    def tearDown(self):
        tm.MeasurementIntegration.tearDown(self)

    prepare=tm.MeasurementIntegration.prepare

    def fake_session(self,reg,cid,out,timeout=90,work=None):
        start=registry.now()
        plan=reg.contract(cid);opts=runtime.validate(plan)[1]
        kv=opts['--cache-type-k'];holdout='joint_decision_id' in plan['workload']
        peak=600 if kv=='f16' else 400
        if holdout and self.fail_holdout and kv=='q8_0':peak=700
        r=tm.MeasurementIntegration.fake_session(self,reg,cid,out,timeout,work)
        samples=[{'elapsed_s':t,'used_bytes':u,'total_bytes':1000,'reserved_bytes':10,'free_bytes':990-u} for t,u in [(0,10),(.125,peak),(.25,10)]]
        lib=next(x for x in plan['inputs'] if x['role']=='runtime_dependency')
        r.update(started_at=start,finished_at=registry.now(),startup_s=.1,gpu_before={'uuid':plan['execution']['gpu_uuids'][0],'name':'Synthetic GPU','driver':'test','total_mib':24*1024},memory_trace={'method':'NVML device-wide sampled high-water mark; lower bound on instantaneous peak','samples':samples,'total_bytes':1000,'sampled_peak_bytes':peak,'minimum_free_bytes':990-peak,'max_sample_gap_s':.125},mapped_libraries=[{'path':lib['path'],'sha256':lib['sha256'],'frozen':True}])
        runtime.atomic(out/'runtime_receipt.json',r)
        return r

    def select(self,d,policy=None,requests=18,tokens=4224):
        p={**tj.policy(),'min_accuracy':0} if policy is None else policy
        return workflow.select(self.reg,self.store,d['plan_id'],p,'f16',[256,'allocated',64],720,requests,tokens)

    def test_complete_workflow_and_idempotence(self):
        d=self.prepare()
        with patch.object(runtime,'smoke',side_effect=self.fake_session):
            with self.assertRaisesRegex(ValueError,'archive every calibration'):self.select(d)
            tm.m.run(self.reg,self.store,d['plan_id'],True)
            with self.assertRaisesRegex(ValueError,'Request cap exceeded'):self.select(d,requests=17)
            r=self.select(d)
            self.assertEqual(r['status'],'holdout_frozen')
            self.assertEqual(r['cost']['total_completion_requests'],18)
            self.assertEqual(r,self.select(d))
            with self.assertRaisesRegex(ValueError,'already frozen'):self.select(d,tokens=5000)
            h=self.store.get(r['holdout_id'])
            with self.assertRaisesRegex(ValueError,'workflow-finish'):inference.run(self.reg,h['arms'][0]['contract_id'],h['arms'][0]['run_path'])
            with self.assertRaisesRegex(ValueError,'Explicit --execute'):workflow.finish(self.reg,self.store,d['plan_id'])
            result=workflow.finish(self.reg,self.store,d['plan_id'],True)
            again=workflow.finish(self.reg,self.store,d['plan_id'],True)
        self.assertEqual(result['status'],'heldout_pilot_passed')
        self.assertEqual(result['new_inference_requests'],16)
        self.assertEqual(result['new_smoke_requests'],2)
        self.assertEqual(again['new_inference_requests'],0)
        self.assertEqual(again['new_smoke_requests'],0)
        self.assertEqual(len(self.calls),32)
        self.assertEqual(result['passport_id'],again['passport_id'])
        self.assertFalse(result['deployment_authorized'])
        packet=registry.read_json(Path(d['root'])/'model.passport.json')
        self.assertEqual(passport.verify(packet)['passport_id'],result['passport_id'])

    def test_failed_holdout_is_retained(self):
        d=self.prepare()
        self.fail_holdout=True
        with patch.object(runtime,'smoke',side_effect=self.fake_session):
            tm.m.run(self.reg,self.store,d['plan_id'],True)
            self.select(d)
            r=workflow.finish(self.reg,self.store,d['plan_id'],True)
            again=workflow.finish(self.reg,self.store,d['plan_id'],True)
        self.assertEqual(r['status'],'heldout_failed')
        self.assertIn('objective_not_improved',r['failures'])
        self.assertEqual(again['new_inference_requests'],0)
        p=registry.read_json(Path(d['root'])/'model.passport.json')['passport']
        self.assertEqual(p['optimization_decisions'][0]['validation']['status'],'heldout_failed')

    def test_interrupted_holdout_blocks_retry(self):
        d=self.prepare()
        with patch.object(runtime,'smoke',side_effect=self.fake_session):
            tm.m.run(self.reg,self.store,d['plan_id'],True)
            r=self.select(d)
        h=self.store.get(r['holdout_id'])
        with self.store.db:self.store.db.execute('INSERT INTO workflow_started VALUES (?,?)',(h['arms'][0]['contract_id'],d['plan_id']))
        with patch.object(inference,'run') as run:
            with self.assertRaisesRegex(ValueError,'Previously started holdout missing'):workflow.finish(self.reg,self.store,d['plan_id'],True)
            run.assert_not_called()

    def test_no_eligible_candidate_skips_holdout(self):
        d=self.prepare()
        with patch.object(runtime,'smoke',side_effect=self.fake_session):
            tm.m.run(self.reg,self.store,d['plan_id'],True)
            r=self.select(d,{**tj.policy(),'min_accuracy':1})
            self.assertEqual(r['status'],'no_holdout')
            self.assertIsNone(r['holdout_id'])
            result=workflow.finish(self.reg,self.store,d['plan_id'],True)
        self.assertEqual(result['status'],'no_holdout')
        self.assertEqual(result['new_inference_requests'],0)
        self.assertEqual(len(self.calls),16)

if __name__=='__main__':unittest.main()

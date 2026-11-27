import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import optimize,evidence
from optimize import select,validate_metrics,context,config

POLICY={'objective':'tokens','min_accuracy':.5,'accuracy_tolerance':.02,'max_validation_drop':.1,'min_savings':.05,'max_truncation':.25,'min_tasks':8,'reference':[1024,'allocated',64]}

def p(cap,success,tokens,latency=1,n=100,truncation=0):
    return {'budget':cap,'strategy':'allocated','reserve':64,'n':n,'attempted':n,'complete':True,'joint_success':success,'accuracy':success/n,'semantic_accuracy':success/n,'truncation_rate':truncation,'mean_completion_tokens':tokens,'mean_latency_s':latency}

def report(*ps):
    return {'profiles':[{'budgetfit':v} for v in ps]}

class OptimizeTests(unittest.TestCase):
    def test_eligible(self):
        r=select(report(p(1024,85,200,2),p(256,84,100,1)),POLICY)
        self.assertEqual(config(r['candidate']),(256,'allocated',64))
        self.assertEqual(len(r['pareto_frontier']),2)
    def test_reject_accuracy(self):
        r=select(report(p(1024,90,200),p(256,87,100)),POLICY)
        self.assertIsNone(r['candidate'])
        self.assertIn('accuracy_floor',r['screening'][1]['reasons'])
    def test_reject_no_savings_and_truncation(self):
        r=select(report(p(1024,85,200),p(256,85,195)),POLICY)
        self.assertIsNone(r['candidate'])
        r=select(report(p(1024,85,200),p(256,85,100,truncation=.26)),POLICY)
        self.assertIsNone(r['candidate'])
    def test_best_accuracy_floor(self):
        r=select(report(p(1024,80,200),p(256,80,100),p(512,90,190)),POLICY)
        self.assertEqual(r['candidate']['budget'],512)
    def test_holdout(self):
        c=p(256,85,100)
        self.assertEqual(validate_metrics(p(256,84,90),p(1024,85,200),c,POLICY)['status'],'heldout_pilot_passed')
        r=validate_metrics(p(256,86,100),p(1024,89,200),c,POLICY)
        self.assertEqual(r['status'],'heldout_failed')
        self.assertIn('reference_accuracy_drop',r['failures'])
        self.assertIn('calibration_accuracy_drop',validate_metrics(p(256,70,100),p(1024,70,200),c,POLICY)['failures'])
    def test_invalid_profile(self):
        x=p(256,85,100);x['complete']=False
        with self.assertRaises(ValueError):select(report(p(1024,85,200),x),POLICY)
        x=p(256,85,100);x['mean_completion_tokens']=float('nan')
        with self.assertRaises(ValueError):select(report(p(1024,85,200),x),POLICY)
        x=p(256,85,100);x['accuracy']=.99
        with self.assertRaises(ValueError):select(report(p(1024,85,200),x),POLICY)
    def test_context_gpu(self):
        gpu={'uuid':'gpu','name':'test','driver':'1','total_mib':100}
        s={'contract':{'inputs':[{'role':'model','sha256':'a'*64,'bytes':1}], 'execution':{}},'response_identity':{},'sessions':[{'receipt':{'gpu_before':gpu}}]}
        r={'adapter_sha256':'b','evaluator_sha256':'c'}
        a=context(s,r)
        s['sessions'].append({'receipt':{'gpu_before':{**gpu,'driver':'2'}}})
        with self.assertRaises(ValueError):context(s,r)

class GateTests(unittest.TestCase):
    def test_frozen_gate_and_single_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=evidence.Store(Path(tmp)/'db.sqlite')
            gpu={'uuid':'gpu','name':'test','driver':'1','total_mib':100}
            def pair(seed,started,finished,tag):
                manifest={'seed':seed,'version':'v','suite':'core','variants':1,'ladder_variants':1,'tasks':[{'prompt':f'{tag}-{i}'} for i in range(8)]}
                source={'contract':{'inputs':[{'role':'model','sha256':'a'*64,'bytes':1}],'execution':{}},'response_identity':{'model':'m'},'sessions':[{'receipt':{'gpu_before':gpu,'started_at':started,'finished_at':finished}}],'manifest':manifest}
                result=report(p(256,7,100,n=8),p(1024,7,200,n=8))
                result.update(adapter_sha256='b'*64,evaluator_sha256='c'*64,rows=[{'id':str(i),'budget':cap,'strategy':'allocated','reserve':64,'grade':{'joint_success':i<7}} for cap in (256,1024) for i in range(8)])
                with store.db:
                    sid=store.put_blob(source);eid=store.put_blob({'source_id':sid,'report':result})
                    store.db.execute('INSERT INTO sources VALUES (?,?,?,?)',(sid,tag,'a'*64,started))
                    store.db.execute('INSERT INTO evaluations VALUES (?,?,?,?,?)',(eid,sid,'c'*64,'b'*64,started))
                return eid,(source,result)
            cal=pair(1,'2020-01-01','2020-01-02','cal')
            hold=pair(2,'2020-01-04','2020-01-05','hold')
            badseed=pair(3,'2020-01-04','2020-01-05','badseed')
            early=pair(2,'2020-01-01','2020-01-02','early')
            other=pair(2,'2020-01-04','2020-01-05','other')
            mapping=dict([cal,hold,badseed,early,other])
            try:
                with patch.object(optimize,'load_evaluation',side_effect=lambda _,eid:mapping[eid]),patch.object(optimize,'now',return_value='2020-01-03'):
                    d=optimize.freeze(store,cal[0],POLICY)
                    self.assertEqual(d['validation_seed'],2)
                    with self.assertRaisesRegex(ValueError,'Wrong frozen validation seed'):optimize.validate(store,d['decision_id'],badseed[0])
                    with self.assertRaisesRegex(ValueError,'predates'):optimize.validate(store,d['decision_id'],early[0])
                    v=optimize.validate(store,d['decision_id'],hold[0])
                    self.assertEqual(v['status'],'heldout_pilot_passed')
                    self.assertFalse(v['deployment_authorized'])
                    self.assertEqual(v,optimize.validate(store,d['decision_id'],hold[0]))
                    with self.assertRaisesRegex(ValueError,'already validated'):optimize.validate(store,d['decision_id'],other[0])
            finally:store.close()

if __name__=='__main__':unittest.main()

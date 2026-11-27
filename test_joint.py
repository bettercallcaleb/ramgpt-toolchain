import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evidence
import joint
from registry import digest

def policy():
    return {'objective':'memory','min_accuracy':.5,'max_latency_s':4,'max_sampled_peak_bytes':1000,'min_free_bytes':100,'max_mean_tokens':512,'max_truncation':0,'min_tasks':8,'max_validation_drop':.125,'accuracy_tolerance':.02,'fidelity_report_id':None,'max_kl':None,'max_artifact_bytes':1000,'max_sample_gap_s':.25}

def row(a='a',peak=500,acc=.75,latency=2,tokens=128):
    return {'candidate_id':a*64,'evaluation_id':a*64,'artifact_sha256':'f'*64,'artifact_bytes':100,'runtime':{},'fixed':[1024,'allocated',64],'metrics':{'n':8,'accuracy':acc,'mean_latency_s':latency,'mean_completion_tokens':tokens,'truncation_rate':0},'memory':{'sampled_peak_bytes':peak,'minimum_free_bytes':200,'max_sample_gap_s':.1,'mapped_libraries':[]},'fidelity':None}

class JointTests(unittest.TestCase):
    def test_policy_missing_and_nan(self):
        p=policy();joint.policy(p)
        for k,v in [('min_tasks',7),('max_latency_s',float('nan')),('max_kl',.02),('objective','weighted')]:
            with self.assertRaises(ValueError):joint.policy({**p,k:v})
    def test_missing_fidelity_and_memory(self):
        r=row();p=policy()
        self.assertEqual(joint.screen(r,p),[])
        p.update(fidelity_report_id='c'*64,max_kl=.01)
        r['memory']=None
        self.assertEqual(joint.screen(r,p),['missing_verified_memory','missing_matched_fidelity'])
    def test_hard_limits(self):
        r=row(peak=1001,acc=.25,latency=5,tokens=513)
        self.assertEqual(joint.screen(r,policy()),['accuracy_floor','latency_limit','token_limit','sampled_vram_limit'])
    def test_pareto(self):
        a=row('a',500,.75,2,128);b=row('b',400,.75,3,128);c=row('c',600,.5,3,150)
        self.assertEqual(joint.frontier([a,b,c],policy()),[a,b])
        self.assertEqual(min([a,b],key=lambda r:joint.rank_key(r,policy())),b)
    def test_holdout_gate(self):
        a=row('a',500,.75);b=row('b',400,.75)
        self.assertEqual(joint.validation_metrics(b,a,b,policy()),[])
        b['metrics']['accuracy']=.5
        self.assertIn('holdout_reference_accuracy_drop',joint.validation_metrics(b,a,row(),policy()))
        self.assertIn('holdout_calibration_accuracy_drop',joint.validation_metrics(b,a,row(),policy()))
        self.assertIn('objective_not_improved',joint.validation_metrics(row(peak=500),a,row(),policy()))
    def test_context_mismatch_and_router_exclusion(self):
        a=row('a');b=row('b',peak=400);c=row('c')
        fixture={a['evaluation_id']:({'manifest':{'seed':1}},[a],{'ctx':1},None),b['evaluation_id']:({'manifest':{'seed':2}},[b],{'ctx':1},None),c['evaluation_id']:({},[],None,'unsupported_adaptive_profile')}
        with patch('joint.load',side_effect=lambda store,eid,p,f:fixture[eid]):
            r=joint.select(None,list(fixture),a['evaluation_id'],a['fixed'],policy())
        self.assertEqual(r['selected']['candidate_id'],a['candidate_id'])
        self.assertEqual([x['reason'] for x in r['excluded_evaluations']],['incompatible_task_or_execution_context','unsupported_adaptive_profile'])
    def test_bound_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=evidence.Store(Path(tmp)/'test.sqlite')
            try:
                joint.init(s)
                path=str(Path(tmp)/'run');cid='a'*64
                with s.db:
                    did=s.put_blob({'test':1})
                    h={'decision_id':did,'arms':[{'contract_id':cid,'arm':'candidate','run_path':path}]}
                    hid=s.put_blob(h)
                    s.db.execute('INSERT INTO joint_decisions(id,cohort_key,holdout_id) VALUES (?,?,?)',(did,'cohort',hid))
                    s.db.execute('INSERT INTO joint_bindings VALUES (?,?,?,?)',(cid,did,'candidate',path))
                plan={'contract_id':cid,'workload':{'joint_decision_id':did,'joint_arm':'candidate'}}
                self.assertEqual(joint.check_binding(s,plan,path),h)
                with self.assertRaises(ValueError):joint.check_binding(s,plan,path+'-retry')
            finally:s.close()
    def test_memory_validation(self):
        samples=[{'used_bytes':u,'reserved_bytes':10,'free_bytes':990-u,'total_bytes':1000,'elapsed_s':t} for u,t in [(10,0),(600,.1),(10,.2)]]
        t={'method':'NVML device-wide sampled high-water mark; lower bound on instantaneous peak','samples':samples,'total_bytes':1000,'max_sample_gap_s':.1,'sampled_peak_bytes':600,'minimum_free_bytes':390}
        lib={'path':'/lib/test.so','sha256':'a'*64,'frozen':True}
        s={'contract':{'inputs':[{'role':'runtime_dependency','path':lib['path'],'sha256':lib['sha256']}]},'sessions':[{'receipt':{'memory_trace':t,'mapped_libraries':[lib]}}]}
        self.assertEqual(joint.memory(s)['sampled_peak_bytes'],600)
        bad=copy.deepcopy(s);bad['sessions'][0]['receipt']['memory_trace']['sampled_peak_bytes']=599
        with self.assertRaises(ValueError):joint.memory(bad)
        bad=copy.deepcopy(s);bad['sessions'][0]['receipt']['mapped_libraries'][0]['frozen']=False
        self.assertIsNone(joint.memory(bad))
if __name__=='__main__':unittest.main()

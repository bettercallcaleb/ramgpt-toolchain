import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evidence
import native_bridge as nb
import native_tools as nt
import planner
import portfolio
import registry
import workspace
from test_planner import candidate,policy

class PortfolioTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();p=Path(self.tmp.name)
        self.reg=registry.Registry(p/'registry.sqlite');self.store=evidence.Store(p/'evidence.sqlite')
    def tearDown(self):
        self.store.close();self.reg.close();self.tmp.cleanup()
    def test_separate_cohorts_and_limits(self):
        a='a'*64;b='b'*64;c='c'*64
        models=[{'format':'gguf','sha256':aid,'bytes':100,'sources':[]} for aid in (a,b,c)]
        def plan(store,m,p):
            r=candidate(m['sha256'][0]);r.update(artifact_sha256=m['sha256'],artifact_bytes=100,rejections=[])
            return {'validation_history':[],'next_steps':[],'cohorts':[{'cohort_id':'one' if m['sha256']!=c else 'two','context_sha256':'d'*64,'manifest_sha256':'e'*64,'hardware':{'name':'fixture'},'suite':'core','seed':1,'screening':[r]}]}
        with patch.object(workspace,'models',return_value={'models':models}),patch.object(planner,'build_artifact',side_effect=plan):
            result=portfolio.select(self.store,self.reg,policy())
            self.assertEqual(len(result['cohorts']),2)
            self.assertEqual(sum(m['eligible_profiles'] for m in result['models']),3)
            self.assertFalse(result['deployment_authorized'])
            self.assertEqual(portfolio.select(self.store,self.reg,policy(),max_model_bytes=99)['status'],'no_eligible_candidate')
            self.assertEqual(portfolio.select(self.store,self.reg,policy(),manifest='f'*64)['cohorts'],[])
            with self.assertRaises(ValueError):portfolio.select(self.store,self.reg,policy(),[a,a])
    def test_exl3_legacy_and_observed_memory_not_certified(self):
        aid='a'*64
        nb.save(self.store,'exl3-first-mover',{'format':'exl3','artifact_id':aid,'repo_id':None,'source_revision':None},{'artifact_bytes':100,'declared_bits':4})
        m={'n':8,'attempted':8,'complete':True,'joint_success':6,'accuracy':.75,'semantic_accuracy':.75,'truncation_rate':0,'mean_completion_tokens':20,'mean_latency_s':.1,'budget':256,'strategy':'allocated','reserve':64}
        samples=[{'elapsed_s':0,'used_bytes':10,'free_bytes':90,'reserved_bytes':0,'total_bytes':100},{'elapsed_s':.1,'used_bytes':20,'free_bytes':80,'reserved_bytes':0,'total_bytes':100}]
        trace={'method':'NVML device-wide sampled high-water mark; lower bound on instantaneous peak','samples':samples,'total_bytes':100,'sampled_peak_bytes':20,'minimum_free_bytes':80,'max_sample_gap_s':.1}
        body={'kind':'shared-native-evaluation','artifact_bytes':100,'context':4096,'gpu_name':'fixture','manifest_sha256':'b'*64,'profiles':[{'budgetfit':m}]}
        nb.save(self.store,'exl3-first-mover',{'format':'exl3','artifact_id':aid},body)
        self.assertIn('missing_verified_memory',portfolio.select(self.store,self.reg,policy())['cohorts'][0]['screening'][0]['rejections'])
        nb.save(self.store,'exl3-first-mover',{'format':'exl3','artifact_id':aid},{**body,'memory_trace':trace})
        r=portfolio.select(self.store,self.reg,policy())
        self.assertTrue(all('exl3_runtime_provenance_not_frozen' in x['rejections'] for g in r['cohorts'] for x in g['screening']))
        self.assertEqual(r['status'],'no_eligible_candidate')
        with self.assertRaisesRegex(ValueError,'memory summary'):portfolio.exl_memory({'memory_trace':{**trace,'sampled_peak_bytes':0}})
    def test_native_sync_is_idempotent(self):
        p=Path(self.tmp.name)/'other.sqlite';source=evidence.Store(p)
        try:
            a=nt.save(source,'q4fit','a'*64,{'artifact_bytes':123})
            b=nb.save(source,'hereticbench',{'track':'epistemic'},{'result':{'n':10}})
        finally:source.close()
        first=workspace.sync_native(self.store,p);second=workspace.sync_native(self.store,p)
        self.assertEqual((first['native_receipts_added'],first['native_reports_added']),(1,1))
        self.assertEqual((second['native_receipts_added'],second['native_reports_added']),(0,0))
        self.assertEqual(nt.find(self.store),[a]);self.assertEqual(nb.find(self.store),[b])
        with self.assertRaisesRegex(ValueError,'must differ'):workspace.sync_native(self.store,Path(self.tmp.name)/'evidence.sqlite')

if __name__=='__main__':unittest.main()

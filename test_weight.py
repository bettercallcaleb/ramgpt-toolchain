import copy
import tempfile
import unittest
from pathlib import Path
import evidence
import weight

def record(h,kl,b):
    return {'artifact_sha256':h*64,'artifact_bytes':b,'reference_sha256':'f'*64,'reference_precision':'BF16','kind':'fidelity','scope':{'benchmark_id':'fixture'},'metrics':{'mean_kl_bf16_to_artifact':kl}}

def report():
    return {'schema':weight.SCHEMA,'producer':'quantbench','status':'historical_compact_and_native_bootstrap_replayed','records':[record('a',.01,20),record('b',.02,10),record('c',.03,30)],'source':{},'limitations':[]}

class WeightTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=evidence.Store(Path(self.tmp.name)/'e.sqlite')
    def tearDown(self):
        self.store.close();self.tmp.cleanup()
    def test_idempotent_exact_join(self):
        d=report();a=weight.ingest(self.store,d);b=weight.ingest(self.store,d)
        self.assertEqual(a,b)
        self.assertEqual(len(weight.find(self.store,reference='f'*64)),3)
        self.assertEqual(len(weight.find(self.store,artifact='a'*64)),1)
        v=weight.artifact_view(self.store,'a'*64)
        self.assertEqual(v['missing_axes'],['knowledge_depth','quantization_recipe','managed_task_accuracy'])
        self.assertEqual(weight.artifact_view(self.store,'d'*64)['weight_evidence'],[])
    def test_constraints(self):
        a=weight.ingest(self.store,report())
        r=weight.select_fidelity(self.store,a['report_id'],.025,30)
        self.assertEqual([x['artifact_sha256'] for x in r['pareto']],['a'*64,'b'*64])
        self.assertEqual(weight.select_fidelity(self.store,a['report_id'],.005,30)['status'],'no_eligible_candidate')
        with self.assertRaises(ValueError):weight.select_fidelity(self.store,a['report_id'],float('nan'),30)
    def test_incompatible_reference(self):
        d=report();d['records'][1]['reference_sha256']='e'*64
        a=weight.ingest(self.store,d)
        with self.assertRaises(ValueError):weight.select_fidelity(self.store,a['report_id'],.1,100)
    def test_index_tampering(self):
        a=weight.ingest(self.store,report())
        with self.store.db:self.store.db.execute("UPDATE weight_records SET kind='other' WHERE report_id=? AND ordinal=0",(a['report_id'],))
        with self.assertRaises(ValueError):weight.find(self.store)
    def test_release_hash(self):
        p=Path(self.tmp.name);f=p/'source';f.write_text('original')
        m={'files':{'source':weight.file_record(f)}}
        self.assertEqual(weight.manifest_files(p,m),m['files'])
        f.write_text('modified')
        with self.assertRaises(ValueError):weight.manifest_files(p,m)
    def test_compact_metrics(self):
        import numpy as np
        p=Path(self.tmp.name)/'compact.npz'
        np.savez(p,kl=[.1,.2,.3,.4],segment_id=[0,0,1,1],top1_flip=np.array([True,False,False,True]),bf16_margin=[1.,2.,3.,4.])
        g={'total_positions':4,'segments':2,'positions_per_segment':2}
        r=weight.compact_metrics(p,g,[1,2,4])
        self.assertAlmostEqual(r['mean_kl_bf16_to_artifact'],.25)
        self.assertEqual(r['top1_flips'],2)
        self.assertEqual(r['hcdfr']['4']['flips'],1)
        wrong=copy.deepcopy(r);wrong['top1_flips']=1
        with self.assertRaises(ValueError):weight.same_metrics(r,wrong)
        with self.assertRaises(ValueError):weight.compact_metrics(p,{**g,'segments':3},[1])
    def test_conflicting_size(self):
        d=report();weight.ingest(self.store,d)
        d=copy.deepcopy(d);d['records'][0]['artifact_bytes']=21;weight.ingest(self.store,d)
        with self.assertRaises(ValueError):weight.artifact_view(self.store,'a'*64)

if __name__=='__main__':unittest.main()

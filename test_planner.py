import copy
import struct
import tempfile
import unittest
from pathlib import Path
import evidence
import planner
import registry
from test_joint import row

def policy(**changes):
    p=planner.policy('24GiB','balanced',.5,8,'1GiB',None,.25,None,None)
    p.update(changes)
    return p

def candidate(a='a',peak=10*2**30,accuracy=.75,latency=2,tokens=128):
    r=row(a,peak,accuracy,latency,tokens)
    r['memory']['minimum_free_bytes']=2*2**30
    r['runtime']={'-c':'4096'}
    return r

class PlannerTests(unittest.TestCase):
    def test_units_and_validation(self):
        self.assertEqual(planner.bytes_arg('24GB'),24*10**9)
        self.assertEqual(planner.bytes_arg('24GiB'),24*2**30)
        for v in ('24','0GB','-1GiB','1.2B','nanGB'):
            with self.assertRaises(ValueError):planner.bytes_arg(v)
        with self.assertRaises(ValueError):planner.policy('1GiB','balanced',.85,8,'1GiB',None,.25,None,None)
        with self.assertRaises(ValueError):planner.policy('24GiB','balanced',.85,8,'1GiB',None,.25,None,.1)

    def test_balanced_keeps_tradeoffs(self):
        a=candidate('a',latency=1);b=candidate('b',peak=9*2**30,latency=2)
        r=planner.select_group([a,b],policy(),{})
        self.assertEqual(r['pareto'],[a['candidate_id'],b['candidate_id']])
        self.assertIsNone(r['exploratory_pick'])
        r=planner.select_group([a,b],policy(objective='memory'),{})
        self.assertEqual(r['exploratory_pick'],b['candidate_id'])

    def test_failed_holdout_never_reselected(self):
        a=candidate('a');b=candidate('b',peak=9*2**30)
        r=planner.select_group([a,b],policy(),{b['candidate_id']:['d'*64]})
        self.assertEqual(r['pareto'],[a['candidate_id']])
        self.assertIn('previous_heldout_failure',b['rejections'])
        self.assertEqual(b['failed_validation_ids'],['d'*64])

    def test_duplicate_replicates_not_cherry_picked(self):
        a=candidate();b=copy.deepcopy(a);b['metrics']['accuracy']=1
        r=planner.select_group([a,b],policy(),{})
        self.assertEqual(r['status'],'no_eligible_candidate')
        self.assertTrue(all('replicate_resolution_required' in x['rejections'] for x in r['screening']))

    def test_memory_constraints(self):
        a=candidate(peak=24*2**30)
        a['memory']['minimum_free_bytes']=0
        a['memory']['max_sample_gap_s']=.5
        self.assertEqual(planner.reasons(a,policy(context_tokens=8192),{}),['context_mismatch','vram_budget','observed_headroom','telemetry_gap'])
        a['memory']=None
        self.assertIn('missing_verified_memory',planner.reasons(a,policy(),{}))

    def test_fidelity_and_accuracy(self):
        a=candidate(accuracy=.75)
        self.assertEqual(planner.reasons(a,policy(min_accuracy=.85,max_kl=.01),{}),['accuracy_floor','missing_matched_fidelity'])

    def test_new_model_plan(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);m=p/'test.gguf';m.write_bytes(b'GGUF'+struct.pack('<IQQ',3,1,0)+b'fixture')
            reg=registry.Registry(p/'reg.sqlite');store=evidence.Store(p/'ev.sqlite')
            try:
                d=planner.build(reg,store,m,policy())
                self.assertEqual(d['status'],'measurement_required')
                self.assertEqual(d['new_inference_requests'],0)
                self.assertFalse(d['deployment_authorized'])
                self.assertEqual(d,planner.build(reg,store,m,policy()))
                self.assertIn('measure_fixed_profiles_on_target_runtime',d['next_steps'])
            finally:reg.close();store.close()

if __name__=='__main__':unittest.main()

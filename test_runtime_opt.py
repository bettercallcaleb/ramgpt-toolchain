import copy
import unittest
from unittest.mock import patch
import runtime_opt
import telemetry

class RuntimeOptimizationTests(unittest.TestCase):
    def test_timings(self):
        raw={'usage':{'prompt_tokens':100,'completion_tokens':20},'timings':{'prompt_n':100,'predicted_n':20,'cache_n':0,'prompt_ms':200,'predicted_ms':400}}
        report=telemetry.summarize([{'raw':raw,'elapsed_s':.7}])
        self.assertEqual(report['prefill_tokens_per_s'],500)
        self.assertEqual(report['decode_tokens_per_s'],50)
        self.assertEqual(report['mean_http_latency_s'],.7)
        for k,v in [('predicted_n',21),('cache_n',1),('prompt_ms',0),('predicted_ms',float('inf'))]:
            bad=copy.deepcopy(raw);bad['timings'][k]=v
            with self.assertRaises(ValueError):telemetry.timings(bad)

    def test_native_args(self):
        base=['-m','model','-c','4096','-np','1','-ngl','999','--host','127.0.0.1','--jinja','-fitt','2048']
        native=['llama-server','--model','model','--gpu-layers','999','-ctk','q8_0','-ctv','q8_0','--ctx-size','4096','--parallel','1','--device','CUDA0','--fit','off','--no-warmup']
        args=runtime_opt.args_for(base,native)
        self.assertEqual(args[:3],['--host','127.0.0.1','--jinja'])
        self.assertNotIn('-fitt',args)
        self.assertIn('--cache-type-k',args)
        with self.assertRaises(ValueError):runtime_opt.args_for(base,native+['--host','0.0.0.0'])
        with self.assertRaises(ValueError):runtime_opt.args_for(base,native+['-ctk','f16'])

    def test_memory_report(self):
        t=object.__new__(telemetry.MemoryTrace)
        t.interval=.1;t.errors=[]
        t.samples=[{'elapsed_s':i*.1,'used_bytes':u,'total_bytes':1000,'free_bytes':1000-u,'reserved_bytes':0} for i,u in enumerate([10,600,40])]
        r=t.report()
        self.assertEqual(r['sampled_peak_bytes'],600)
        self.assertEqual(r['minimum_free_bytes'],400)
        self.assertEqual(r['baseline_bytes'],10)
        t.errors=['failed']
        with self.assertRaises(ValueError):t.report()

    def test_dependencies_unresolved(self):
        with patch('runtime_opt.subprocess.run') as p:
            p.return_value.stdout='libfoo.so => not found'
            with self.assertRaises(ValueError):runtime_opt.dependencies('llama-server')

if __name__=='__main__':unittest.main()

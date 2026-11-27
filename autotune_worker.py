import json
import os
import sys
from pathlib import Path

sys.path.insert(0,sys.argv[1])
from llama_autotune.config import POLICY,RuntimeConfig,Workload
from llama_autotune.llama.capabilities import discover,runtime_args
from llama_autotune.tuning.policy import variants
from llama_autotune.executor import Executor

if POLICY!='v0.1.8-smoke-frontier-1':raise ValueError('Unsupported native search policy')
d=json.load(sys.stdin)
cap=discover(Path(d['runtime']),Executor(Path(d['logs'])))
base=RuntimeConfig(ngl=d['ngl'],ctx=d['ctx'],parallel=1,devices='CUDA0',batch=d['batch'],ubatch=d['ubatch'])
if 'CUDA0' not in cap.devices:raise ValueError('Visible CUDA0 not advertised')
rows=variants(base,cap,True,'quick',workload=Workload(budget='quick',objective='performance'))
out={'policy':POLICY,'capability_fingerprint':cap.fingerprint,'devices':cap.devices,'candidates':[{'config':c.json(),'args':runtime_args(cap,d['model'],c)} for c in rows]}
json.dump(out,sys.stdout,allow_nan=False)

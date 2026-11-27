import os
import signal
import socket
import subprocess
import time
from pathlib import Path
import adaptive
import evaluation
from registry import digest,fingerprint,read_json
import runtime
import runtime_opt

SAMPLING={'temperature':0,'top_p':1,'top_k':0,'min_p':0,'presence_penalty':0,'repeat_penalty':1,'frequency_penalty':0}
MESSAGE={'role':'user','content':'Return only 42.'}

def execution(model,gpu,port,context,threads,ngl):
    if not runtime.UUID.fullmatch(gpu):raise ValueError('Require one GPU UUID')
    for n,a,b in ((port,1024,65535),(context,256,131072),(threads,1,128),(ngl,1,999)):
        if type(n) is not int or not a<=n<=b:raise ValueError('Invalid runtime setting')
    args=['-m',model,'--alias','model','--host','127.0.0.1','--port',str(port),'-c',str(context),'-np','1','-ngl',str(ngl),'-t',str(threads),'--seed','602','--jinja','--reasoning','off','--reasoning-format','deepseek','--no-webui','--cache-type-k','f16','--cache-type-v','f16','--flash-attn','on','--batch-size','512','--ubatch-size','128','--device','CUDA0','--fit','off','--no-warmup']
    for k,v in SAMPLING.items():args.extend((runtime.SAMPLING_FLAGS[k],str(v)))
    return {'runtime_args':args,'gpu_uuids':[gpu],'sampling':dict(SAMPLING),'reasoning_policy':'off','template_binding':'observed model-specific apply-template probe; no completion during capture','runtime_measurement':'nvml-sampled-v1'}

def check_binding(spec):
    binding=spec.get('execution',{}).get('template_capture')
    if binding is None:return
    refs=[fingerprint(x['path'],x['role']) for x in spec['inputs']]
    observed=[{k:r[k] for k in ('role','sha256','bytes')} for r in refs]
    if sorted(observed,key=lambda r:r['role'])!=binding['inputs']:raise ValueError('Captured spec inputs changed; create a new runtime spec')

def capture(model,binary,archive,gpu,out,port=18429,context=4096,threads=8,ngl=999,execute=False,timeout=90):
    if execute is not True:raise ValueError('Explicit --execute required for template-only GPU model load')
    refs=[fingerprint(x,role) for x,role in ((model,'model'),(binary,'runtime'),(archive,'evaluator'))]
    model,binary,archive=[r['path'] for r in refs]
    if refs[2]['sha256']!=adaptive.PIN:raise ValueError('Pinned BudgetFit archive required')
    ex=execution(model,gpu,port,context,threads,ngl)
    plan={'inputs':refs+[{'role':'template','path':'pending'}],'execution':ex}
    runtime.validate(plan)
    root=Path(out).expanduser().resolve()
    if root.exists():raise ValueError('Require fresh runtime-spec output directory')
    root.mkdir(parents=True,exist_ok=False)
    with runtime.lease(gpu):
        before=runtime.gpu_state().get(gpu)
        if not before or runtime.compute_apps(gpu) or before['used_mib']>512:raise ValueError('Selected GPU occupied or unavailable')
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind(('127.0.0.1',port))
        proc=None
        def interrupt(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
        previous=signal.signal(signal.SIGTERM,interrupt)
        try:
            with (root/'server.log').open('x') as log:
                proc=subprocess.Popen([binary,*ex['runtime_args']],stdout=log,stderr=subprocess.STDOUT,env=runtime_opt.clean_env(gpu),start_new_session=True)
                end=time.monotonic()+timeout
                while time.monotonic()<end:
                    if proc.poll() is not None:raise ValueError('Server exited before template capture; see server.log')
                    if runtime.owns_listener(proc.pid,port):
                        try:
                            if runtime.http(port,'/health',timeout=1).get('status')=='ok':break
                        except (OSError,ValueError):pass
                    time.sleep(.3)
                else:raise ValueError('Template capture startup timeout')
                if not runtime.owns_listener(proc.pid,port):raise ValueError('Listener ownership lost')
                template=runtime.http(port,'/apply-template',{'messages':[MESSAGE]})
                if not isinstance(template,dict) or not isinstance(template.get('prompt'),str) or not template['prompt']:raise ValueError('Native template response missing prompt')
                if not runtime.owns_listener(proc.pid,port):raise ValueError('Listener ownership lost')
        finally:
            try:
                if proc is not None:runtime.stop_owned(proc)
            finally:signal.signal(signal.SIGTERM,previous)
        after=runtime.gpu_state()[gpu]
        if runtime.compute_apps(gpu) or after['used_mib']>before['used_mib']+128:raise ValueError('GPU did not return to baseline')
        if refs!=[fingerprint(r['path'],r['role']) for r in refs]:raise ValueError('Capture input changed')
    evaluation.write_new(root/'template.json',template)
    refs.append(fingerprint(root/'template.json','template'))
    ex['template_capture']={'inputs':sorted([{k:r[k] for k in ('role','sha256','bytes')} for r in refs],key=lambda r:r['role']),'probe_sha256':digest(template)}
    spec={'inputs':[{k:r[k] for k in ('role','path')} for r in refs],'execution':ex,'workload':{'protocol':'toolchain-inference-v1','seed':0,'budgets':[256]}}
    check_binding(spec)
    evaluation.write_new(root/'spec.json',spec)
    return {'status':'template_captured','spec':str(root/'spec.json'),'model_sha256':refs[0]['sha256'],'template_sha256':refs[-1]['sha256'],'new_inference_requests':0,'gpu_cleanup_verified':True,'scope':'Native template rendering only; literal smoke and scoring still required during run.'}

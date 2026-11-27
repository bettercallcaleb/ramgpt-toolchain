import contextlib
import fcntl
import json
import math
import os
import re
import signal
import socket
import stat
import subprocess
import time
import urllib.request
from pathlib import Path
from registry import now, read_json, fingerprint

SCHEMA='ramgpt-managed-runtime-v1'
SAMPLING_FLAGS={'temperature':'--temp','top_p':'--top-p','top_k':'--top-k','min_p':'--min-p','presence_penalty':'--presence-penalty','repeat_penalty':'--repeat-penalty','frequency_penalty':'--frequency-penalty'}
UUID=re.compile(r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
VALUES={'-m','--alias','--host','--port','-c','-np','-ngl','-fitt','-t','--seed','--reasoning','--reasoning-format','--temp','--top-p','--top-k','--min-p','--presence-penalty','--repeat-penalty','--frequency-penalty'}
FLAGS={'--jinja','--no-webui','--no-warmup'}
TUNING_VALUES={'--cache-type-k','--cache-type-v','--flash-attn','--batch-size','--ubatch-size','--fit','--device'}
VALUES |= TUNING_VALUES

def atomic(path,value):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)

def validate(plan):
    inputs={r['role']:r for r in plan['inputs'] if r['role'] in ('model','runtime','template')}
    e=plan['execution'];args=e['runtime_args'];opts={};i=0
    while i<len(args):
        key=args[i];i+=1
        if key in opts:raise ValueError('Duplicate runtime option: '+key)
        if key in FLAGS:opts[key]=True;continue
        if key not in VALUES or i==len(args) or args[i] in VALUES|FLAGS:
            raise ValueError('Unsupported runtime option: '+key)
        opts[key]=args[i];i+=1
    if opts.get('-m')!=inputs['model']['path'] or opts.get('--host')!='127.0.0.1' or opts.get('-np')!='1':
        raise ValueError('Require frozen model, loopback host and one slot')
    if not opts.get('--jinja') or not opts.get('--no-webui') or opts.get('--reasoning')!='off' or e['reasoning_policy']!='off':
        raise ValueError('Require jinja, no web UI and reasoning off')
    sampling=e.get('sampling',{})
    if not isinstance(sampling,dict) or set(sampling)-SAMPLING_FLAGS.keys():raise ValueError('Unsupported sampling parameters')
    for name,value in sampling.items():
        if type(value) not in (int,float) or not math.isfinite(value) or SAMPLING_FLAGS[name] not in opts or float(opts[SAMPLING_FLAGS[name]])!=value:
            raise ValueError('Sampling differs from frozen runtime flags: '+name)
    for k in ('--cache-type-k','--cache-type-v'):
        if k in opts and opts[k] not in ('f16','q8_0','q4_0'):raise ValueError('Unsupported KV type')
    if '--flash-attn' in opts and opts['--flash-attn'] not in ('on','off','auto'):raise ValueError('Unsupported flash mode')
    if '--device' in opts and opts['--device']!='CUDA0':raise ValueError('Require single visible CUDA device')
    if '--fit' in opts and opts['--fit']!='off':raise ValueError('Require explicit fitting off')
    for k in ('--batch-size','--ubatch-size'):
        if k in opts and not 1<=int(opts[k])<=8192:raise ValueError('Invalid batch size')
    if '--batch-size' in opts and '--ubatch-size' in opts and int(opts['--ubatch-size'])>int(opts['--batch-size']):raise ValueError('Invalid microbatch size')
    port=int(opts['--port'])
    if not 1024<=port<=65535:raise ValueError('Invalid port')
    alias=opts['--alias']
    if not re.fullmatch(r'[A-Za-z0-9._-]{1,128}',alias):raise ValueError('Invalid alias')
    gpus=e['gpu_uuids']
    if len(gpus)!=1 or not UUID.fullmatch(gpus[0]):raise ValueError('This version requires exactly one GPU UUID')
    if Path(inputs['runtime']['path']).name!='llama-server' or not os.access(inputs['runtime']['path'],os.X_OK):
        raise ValueError('Expected executable llama-server')
    return inputs,opts,port,alias,gpus[0]

def gpu_state():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,driver_version','--format=csv,noheader,nounits'],text=True,timeout=10)
    rows={}
    for line in raw.splitlines():
        uuid,name,total,used,driver=[s.strip() for s in line.split(',')]
        rows[uuid]={'uuid':uuid,'name':name,'total_mib':int(total),'used_mib':int(used),'driver':driver}
    return rows

def compute_apps(uuid):
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=10)
    return [line.strip() for line in raw.splitlines() if line.split(',')[0].strip()==uuid]

@contextlib.contextmanager
def lease(uuid):
    if not UUID.fullmatch(uuid):raise ValueError('Invalid GPU UUID')
    root=Path('/tmp')/f'ramgpt-gpu-lease-{os.getuid()}'
    root.mkdir(mode=0o700,exist_ok=True)
    st=root.lstat()
    if not stat.S_ISDIR(st.st_mode) or st.st_uid!=os.getuid() or st.st_mode&0o077:raise ValueError('Unsafe GPU lease directory')
    fd=os.open(root/(uuid+'.lock'),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        s=os.fstat(fd)
        if not stat.S_ISREG(s.st_mode) or s.st_uid!=os.getuid():raise ValueError('Unsafe GPU lease file')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('GPU already leased: '+uuid)
        yield
    finally:os.close(fd)

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def http(port,path,body=None,timeout=5):
    data=None if body is None else json.dumps(body,allow_nan=False).encode()
    req=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=data,headers={'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    with opener.open(req,timeout=timeout) as response:
        return json.load(response)

def owns_listener(pid,port):
    try:
        sockets=set()
        for entry in Path(f'/proc/{pid}/fd').iterdir():
            try:link=os.readlink(entry)
            except FileNotFoundError:continue
            if link.startswith('socket:['):sockets.add(link[8:-1])
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            fields=line.split()
            if fields[1]==f'0100007F:{port:04X}' and fields[3]=='0A' and fields[9] in sockets:return True
    except FileNotFoundError:return False
    return False

def stop_owned(proc):
    if proc.poll() is None:
        proc.terminate()
        try:proc.wait(timeout=15)
        except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=10)
    return proc.returncode

def check_usage(raw,cap):
    c=raw['choices']
    if len(c)!=1:raise ValueError('Expected one completion')
    c=c[0];m=c['message'];u=raw['usage']
    if c['finish_reason']!='stop' or m.get('tool_calls') or m.get('reasoning_content') or m.get('reasoning') or m.get('content','').strip()!='42':
        raise ValueError('Literal smoke failed')
    if type(u.get('completion_tokens')) is not int or not 1<=u['completion_tokens']<=cap:raise ValueError('Invalid completion usage')
    return u

def mapped_libraries(pid,plan):
    bound={r['path']:r for r in plan['inputs'] if r['role']=='runtime_dependency'}
    paths=set()
    for line in Path(f'/proc/{pid}/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6 or not fields[5].startswith('/'):continue
        path=fields[5]
        if path.endswith(' (deleted)') and '.so' in path:raise ValueError('Deleted mapped shared library')
        if '.so' in Path(path).name:paths.add(str(Path(path).resolve(strict=True)))
    if not paths:raise ValueError('No mapped shared libraries')
    return [{'path':p,'sha256':bound[p]['sha256'] if p in bound else fingerprint(p,'runtime_dependency')['sha256'],'frozen':p in bound} for p in sorted(paths)]

def smoke(reg,cid,out,timeout=90,work=None):
    plan=reg.contract(cid);inputs,opts,port,alias,uuid=validate(plan)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    receipt={'schema':SCHEMA,'contract_id':cid,'status':'started','started_at':now(),'gpu_uuid':uuid,'input_precheck':False,'input_postcheck':False,'template_probe_match':False,'smoke_pass':False,'cleanup_complete':False}
    path=out/'runtime_receipt.json';atomic(path,receipt)
    def interrupt(signum,frame):raise KeyboardInterrupt('Signal '+str(signum))
    previous=signal.signal(signal.SIGTERM,interrupt)
    try:
        with lease(uuid):
            reg.verify(cid);receipt['input_precheck']=True
            before=gpu_state()[uuid];receipt['gpu_before']=before
            if compute_apps(uuid) or before['used_mib']>512:raise ValueError('GPU occupied; refusing to launch')
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                s.bind(('127.0.0.1',port))
            env={k:v for k,v in os.environ.items() if not k.startswith(('CUDA_VISIBLE_DEVICES','GGML_','LLAMA_ARG_','LD_PRELOAD','LD_LIBRARY_PATH'))}
            env['CUDA_VISIBLE_DEVICES']=uuid
            cmd=[inputs['runtime']['path'],*plan['execution']['runtime_args']]
            proc=None
            trace=None
            if plan['execution'].get('runtime_measurement')=='nvml-sampled-v1':
                from telemetry import MemoryTrace
                trace=MemoryTrace(uuid)
                trace.__enter__()
            try:
                with (out/'server.log').open('x') as log:
                    start=time.monotonic()
                    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    receipt['pid']=proc.pid;atomic(path,receipt)
                    deadline=start+timeout
                    while time.monotonic()<deadline:
                        if proc.poll() is not None:raise RuntimeError('Server exited before health')
                        if owns_listener(proc.pid,port):
                            try:
                                if http(port,'/health',timeout=1).get('status')=='ok':break
                            except (OSError,ValueError):pass
                        time.sleep(.3)
                    else:raise RuntimeError('Server startup timeout')
                    receipt['startup_s']=time.monotonic()-start
                    if not owns_listener(proc.pid,port):raise ValueError('Server listener ownership lost')
                    messages=[{'role':'user','content':'Return only 42.'}]
                    observed=http(port,'/apply-template',{'messages':messages})
                    atomic(out/'template_observed.json',observed)
                    if observed!=read_json(inputs['template']['path']):raise ValueError('Frozen template probe mismatch')
                    receipt['template_probe_match']=True
                    if not owns_listener(proc.pid,port):raise ValueError('Server listener ownership lost')
                    body={'model':alias,'messages':messages,'max_tokens':64,**plan['execution']['sampling'],'seed':plan['workload']['seed'],'cache_prompt':False}
                    atomic(out/'smoke_request.json',body)
                    start=time.monotonic();raw=http(port,'/v1/chat/completions',body,timeout=60)
                    receipt['smoke_elapsed_s']=time.monotonic()-start
                    atomic(out/'smoke_response.json',raw)
                    if not owns_listener(proc.pid,port):raise ValueError('Server listener ownership lost')
                    receipt['usage']=check_usage(raw,64)
                    if raw.get('model')!=alias:raise ValueError('Response alias mismatch')
                    receipt['response_identity']={k:raw[k] for k in ('model','system_fingerprint') if k in raw}
                    receipt['smoke_pass']=True;receipt['gpu_loaded']=gpu_state()[uuid]
                    if work is not None:
                        def send(body,timeout=120):
                            if proc.poll() is not None or not owns_listener(proc.pid,port):raise ValueError('Server listener ownership lost')
                            started=time.monotonic();raw=http(port,'/v1/chat/completions',body,timeout=timeout)
                            if proc.poll() is not None or not owns_listener(proc.pid,port):raise ValueError('Server listener ownership lost')
                            return raw,time.monotonic()-started
                        receipt['workload_result']=work(send,receipt['response_identity'])
                    if trace is not None:
                        receipt['mapped_libraries']=mapped_libraries(proc.pid,plan)
            finally:
                if proc is not None:
                    receipt['server_exit_code']=stop_owned(proc)
                    receipt['cleanup_complete']=proc.poll() is not None
                if trace is not None:
                    trace.__exit__(None,None,None)
                    receipt['memory_trace']=trace.report()
                receipt['gpu_after']=gpu_state()[uuid]
                if compute_apps(uuid) or receipt['gpu_after']['used_mib']>before['used_mib']+128:
                    raise ValueError('GPU not released to expected baseline')
                reg.verify(cid);receipt['input_postcheck']=True
            receipt['status']='runtime_smoke_passed' if work is None else 'runtime_session_passed'
    except BaseException as e:
        receipt['status']='failed';receipt['error']=str(e);raise
    finally:
        signal.signal(signal.SIGTERM,previous)
        receipt['finished_at']=now();atomic(path,receipt)
    return receipt

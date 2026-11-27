import hashlib
import os
import subprocess
import urllib.request
from pathlib import Path
import adaptive
import doctor
import evaluation
from registry import fingerprint

URL='https://github.com/bettercallcaleb/budgetfit/releases/download/v0.6.0/budgetfit.pyz'
LIMIT=16*1024*1024
FLAGS=('--jinja','--reasoning','--reasoning-format','--cache-type-k','--cache-type-v','--flash-attn','--fit','--device','--no-warmup','--no-webui')

def runtime_check(binary):
    ref=fingerprint(binary,'runtime')
    if Path(ref['path']).name!='llama-server' or not os.access(ref['path'],os.X_OK):raise ValueError('Expected executable llama-server')
    env={k:v for k,v in os.environ.items() if not k.startswith(('GGML_','LLAMA_ARG_','LD_PRELOAD','LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES'))}
    env['CUDA_VISIBLE_DEVICES']=''
    h=subprocess.run([ref['path'],'--help'],capture_output=True,text=True,timeout=30,env=env)
    missing=[f for f in FLAGS if f not in h.stdout+h.stderr]
    if h.returncode or missing:raise ValueError('llama-server help check failed or missing flags: '+', '.join(missing))
    return ref

def fetch(path,archive=None):
    h=hashlib.sha256();n=0
    source=Path(archive).expanduser().resolve(strict=True).open('rb') if archive else urllib.request.urlopen(URL,timeout=30)
    created=False
    try:
        with path.open('xb') as f:
            created=True
            while chunk:=source.read(65536):
                n+=len(chunk)
                if n>LIMIT:raise ValueError('Evaluator exceeds download size cap')
                h.update(chunk);f.write(chunk)
        if h.hexdigest()!=adaptive.PIN:raise ValueError('BudgetFit SHA-256 mismatch')
    except BaseException:
        if created:path.unlink(missing_ok=True)
        raise
    finally:source.close()
    return n

def setup(runtime,out,archive=None,execute=False):
    ref=runtime_check(runtime)
    root=Path(out).expanduser().resolve()
    if root.exists():raise ValueError('Require fresh setup directory')
    if archive and fingerprint(archive,'evaluator')['sha256']!=adaptive.PIN:raise ValueError('Pinned BudgetFit archive required')
    if not execute:return {'status':'setup_planned','runtime_sha256':ref['sha256'],'evaluator_sha256':adaptive.PIN,'download_url':None if archive else URL,'out':str(root),'new_inference_requests':0}
    root.mkdir(parents=True,exist_ok=False)
    dest=root/'budgetfit.pyz';n=fetch(dest,archive)
    if fingerprint(ref['path'],'runtime')!=ref:raise ValueError('Runtime changed during setup')
    config={'llama-autotune':{'runtime':ref['path']},'budgetfit':{'evaluator':str(dest)},'trustmebro':{'evaluator':str(dest)}}
    evaluation.write_new(root/'tools.json',{'tools':config})
    d=doctor.check(str(dest),ref['path'],gpu=True)
    ready=d['status']=='checks_passed' and d['optional_dependencies']['nvidia_ml_py']['available']
    result={'status':'gguf_dependencies_ready' if ready else 'gguf_dependencies_incomplete','config':str(root/'tools.json'),'runtime_sha256':ref['sha256'],'evaluator_sha256':adaptive.PIN,'evaluator_bytes':n,'doctor':d,'next_steps':[] if ready else ['Install the GPU extra and verify NVIDIA driver availability'],'new_inference_requests':0,'scope':'Pinned evaluator and trusted existing llama-server; no GPU model load or native runtime installation'}
    evaluation.write_new(root/'setup.json',result)
    return result

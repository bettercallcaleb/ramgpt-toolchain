import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from registry import fingerprint
from release_info import VERSION

def package(module,dist):
    try:
        return {'available':importlib.util.find_spec(module) is not None,'version':importlib.metadata.version(dist)}
    except (ImportError,importlib.metadata.PackageNotFoundError):
        return {'available':False,'version':None}

def check(evaluator=None,runtime=None,model=None,gpu=False):
    from adaptive import PIN
    linux=sys.platform.startswith('linux')
    python_ok=sys.version_info>=(3,12)
    numpy=package('numpy','numpy')
    nvml=package('pynvml','nvidia-ml-py')
    devices=None;gpu_error=None
    if gpu:
        cmd=shutil.which('nvidia-smi')
        if not cmd:gpu_error='nvidia-smi not found'
        else:
            try:
                p=subprocess.run([cmd,'--query-gpu=name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True,timeout=10)
                devices=[]
                for line in p.stdout.splitlines():
                    name,mem,driver=[x.strip() for x in line.split(',')]
                    devices.append({'name':name,'total_mib':int(mem),'driver':driver})
                if not devices:gpu_error='No GPU reported'
            except (OSError,ValueError,subprocess.SubprocessError):gpu_error='GPU query failed'
    artifacts={}
    for role,path in [('evaluator',evaluator),('runtime',runtime),('model',model)]:
        if path is None:continue
        try:
            r=fingerprint(path,role)
            result={'sha256':r['sha256'],'bytes':r['bytes'],'valid':True}
            if role=='evaluator':result.update(pinned_budgetfit_match=r['sha256']==PIN,valid=r['sha256']==PIN and Path(path).suffix=='.pyz')
            if role=='runtime':result.update(valid=Path(r['path']).name=='llama-server' and os.access(r['path'],os.X_OK))
            artifacts[role]=result
        except (OSError,ValueError) as e:artifacts[role]={'valid':False,'error_type':type(e).__name__}
    base=linux and python_ok
    supplied_valid=all(x['valid'] for x in artifacts.values())
    return {'version':VERSION,'platform':platform.system(),'python':platform.python_version(),'core_supported':base,
            'optional_dependencies':{'numpy':numpy,'nvidia_ml_py':nvml},
            'commands':{name:shutil.which(name) is not None for name in ('nvidia-smi','ldd')},
            'gpu_query':{'requested':gpu,'devices':devices,'error':gpu_error},'supplied_inputs':artifacts,
            'capabilities':{'offline_core':base,'quantbench_compact':base and numpy['available'],'nvml_sampling_dependency':nvml['available'],'pinned_evaluator_supplied':artifacts.get('evaluator',{}).get('valid',False),'runtime_executable_supplied':artifacts.get('runtime',{}).get('valid',False)},
            'status':'checks_passed' if base and supplied_valid and not gpu_error else 'checks_failed',
            'scope':'Read-only checks; no model load, server launch, inference, download or GPU allocation. Dependencies are detected, not certified.'}

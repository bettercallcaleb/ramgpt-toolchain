import json
import math
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from registry import digest,fingerprint,read_json
import evaluation
import evidence
import inference
import optimize
import runtime
import telemetry

SCHEMA='ramgpt-runtime-comparison-v1'
ALIASES={'--model':'-m','--gpu-layers':'-ngl','--n-gpu-layers':'-ngl','--ctx-size':'-c','--parallel':'-np','-ctk':'--cache-type-k','-ctv':'--cache-type-v','-fa':'--flash-attn','-b':'--batch-size','-ub':'--ubatch-size','-fit':'--fit','-dev':'--device'}

def clean_env(uuid=None):
    env={k:v for k,v in os.environ.items() if not k.startswith(('CUDA_VISIBLE_DEVICES','GGML_','LLAMA_ARG_','LD_PRELOAD','LD_LIBRARY_PATH'))}
    if uuid:env['CUDA_VISIBLE_DEVICES']=uuid
    return env

def dependencies(binary):
    p=subprocess.run(['ldd',str(binary)],env=clean_env(),text=True,capture_output=True,check=True,timeout=30)
    if 'not found' in p.stdout:raise ValueError('Unresolved runtime dependency')
    paths=set()
    for line in p.stdout.splitlines():
        m=re.search(r'(?:=>\s+|^\s*)(/\S+)\s+\(',line)
        if m:paths.add(str(Path(m[1]).resolve(strict=True)))
    if not paths:raise ValueError('No resolved dependencies')
    return sorted(paths)

def source_archive(source,out):
    source=Path(source).resolve(strict=True)
    files=sorted((source/'src'/'llama_autotune').rglob('*.py'))
    if not files:raise ValueError('Missing native source')
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.writestr(zipfile.ZipInfo(str(p.relative_to(source/'src'))),p.read_bytes())
    return fingerprint(out,'runtime_dependency')

def native_candidates(archive,d,uuid):
    script=Path(__file__).with_name('autotune_worker.py')
    before=fingerprint(archive,'runtime_dependency')
    p=subprocess.run([sys.executable,'-I',str(script),str(archive)],input=json.dumps(d),env=clean_env(uuid),text=True,capture_output=True,timeout=120)
    if p.returncode:raise ValueError('Native autotune failed: '+p.stderr[-4000:])
    if fingerprint(archive,'runtime_dependency')!=before:raise ValueError('Native archive changed')
    return json.loads(p.stdout)

def args_for(base,native):
    opts={};a=iter(native[1:])
    for k in a:
        key=ALIASES.get(k,k)
        if key in opts:raise ValueError('Duplicate native option')
        if key=='--no-warmup':opts[key]=True
        else:opts[key]=next(a)
    replace={'-m','-ngl','-c','-np','-fitt',*runtime.TUNING_VALUES,'--no-warmup'}
    if set(opts)-replace:raise ValueError('Unsupported native option')
    result=[];a=iter(base)
    for k in a:
        if k in runtime.FLAGS:
            if k not in replace:result.append(k)
        else:
            v=next(a)
            if k not in replace:result.extend([k,v])
    for k,v in opts.items():
        result.append(k)
        if v is not True:result.append(v)
    return result

def prepare(reg,source_spec,autotune_source,out,seed=611):
    import adaptive
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    spec=read_json(source_spec)
    refs={r['role']:r['path'] for r in spec['inputs']}
    native=out/'autotune.zip';source_archive(autotune_source,native)
    uuid=spec['execution']['gpu_uuids'][0]
    data=native_candidates(native,{'runtime':refs['runtime'],'model':refs['model'],'ctx':4096,'ngl':999,'batch':512,'ubatch':128,'logs':str(out/'probe')},uuid)
    evaluation.write_new(out/'native-candidates.json',data)
    chosen=[x for x in data['candidates'] if x['config']['ctk']==x['config']['ctv'] and x['config']['ctk'] in ('f16','q8_0')]
    if [x['config']['ctk'] for x in chosen]!=['f16','q8_0']:raise ValueError('Expected f16 and q8 native candidates')
    archive=refs['evaluator']
    m=adaptive.worker(archive,{'action':'generate','seed':seed,'variants':1,'suite':'core','ladder_variants':1})
    dataset=out/'tasks.json';schedule=out/'schedule.json'
    evaluation.write_new(dataset,m)
    evaluation.prepare(dataset,archive,[1024],['allocated'],64,schedule)
    deps=dependencies(refs['runtime'])
    root=Path(__file__).resolve().parent
    deps+= [str(native),*[str(root/n) for n in ('runtime.py','telemetry.py','runtime_opt.py','autotune_worker.py')]]
    spec['inputs']=[{'role':r['role'],'path':str(dataset if r['role']=='dataset' else schedule if r['role']=='configuration' else r['path'])} for r in spec['inputs']]
    spec['inputs'] += [{'role':'runtime_dependency','path':p} for p in sorted(set(deps))]
    spec['workload'].update(seed=seed,budgets=[1024],request_count=len(m['tasks']),task_count=len(m['tasks']))
    contracts=[]
    for row in chosen:
        import copy
        s=copy.deepcopy(spec);name=row['config']['ctk']
        s['execution']['runtime_args']=args_for(s['execution']['runtime_args'],row['args'])
        s['execution']['runtime_measurement']='nvml-sampled-v1'
        plan=reg.freeze(s);runtime.validate(plan)
        evaluation.write_new(out/(name+'-contract.json'),plan)
        contracts.append({'name':name,'contract_id':plan['contract_id'],'config':row['config']})
    decision={'schema':SCHEMA,'candidates':contracts,'seed':seed,'min_accuracy':.5,'accuracy_tolerance':.02,'minimum_free_bytes':2*1024**3,'scope':'paired runtime pilot; not independent deployment validation'}
    evaluation.write_new(out/'comparison-plan.json',decision)
    return decision

def compare(store,decision,eids):
    if decision['schema']!=SCHEMA or len(eids)!=len(decision['candidates']) or len(eids)!=2:raise ValueError('Invalid comparison')
    results=[];context=None;task_manifest=None;library_set=None
    for candidate,eid in zip(decision['candidates'],eids):
        source,report=optimize.load_evaluation(store,eid)
        plan=source['contract'];w=plan['workload']
        if plan['contract_id']!=candidate['contract_id'] or w['seed']!=decision['seed']:raise ValueError('Unexpected candidate contract')
        c=optimize.context(source,report)
        execution=dict(c['execution']);args=execution.pop('runtime_args')
        options=runtime.validate(plan)[1]
        c['execution']=execution
        c['options']={k:v for k,v in options.items() if k not in ('--cache-type-k','--cache-type-v')}
        if context is None:context=c;task_manifest=source['manifest']
        elif c!=context or source['manifest']!=task_manifest:raise ValueError('Incomparable runtime evidence')
        if len(source['sessions'])!=1:raise ValueError('Require one complete measured session')
        receipt=source['sessions'][0]['receipt']
        trace=receipt['memory_trace'];samples=trace['samples']
        if len(samples)<2 or max(x['used_bytes'] for x in samples)!=trace['sampled_peak_bytes'] or min(x['free_bytes'] for x in samples)!=trace['minimum_free_bytes']:raise ValueError('Invalid memory trace')
        libs=receipt['mapped_libraries']
        if library_set is None:library_set=libs
        elif libs!=library_set:raise ValueError('Mapped library identity mismatch')
        rows=evidence.validate_source(source)
        ps=optimize.profiles(report)
        if len(ps)!=1:raise ValueError('Expected one fixed profile')
        result={'name':candidate['name'],'evaluation_id':eid,'source_id':digest(source),'config':candidate['config'],'metrics':ps[0],
                'performance':telemetry.summarize(rows),'startup_s':receipt['startup_s'],'sampled_peak_bytes':trace['sampled_peak_bytes'],'minimum_free_bytes':trace['minimum_free_bytes'],'samples':len(samples),'max_sample_gap_s':trace['max_sample_gap_s'],
                'observed_unfrozen_libraries':sum(not x['frozen'] for x in libs)}
        results.append(result)
    baseline=results[0]['metrics']['accuracy']
    for r in results:
        r['eligible']=r['metrics']['accuracy']>=max(decision['min_accuracy'],baseline-decision['accuracy_tolerance']) and r['minimum_free_bytes']>=decision['minimum_free_bytes']
    eligible=[r for r in results if r['eligible']]
    def dominates(a,b):
        x=(a['metrics']['accuracy'],-a['sampled_peak_bytes'],a['performance']['prefill_tokens_per_s'],a['performance']['decode_tokens_per_s'])
        y=(b['metrics']['accuracy'],-b['sampled_peak_bytes'],b['performance']['prefill_tokens_per_s'],b['performance']['decode_tokens_per_s'])
        return all(i>=j for i,j in zip(x,y)) and x!=y
    result={'schema':SCHEMA,'plan_digest':digest(decision),'results':results,'pareto':[r['name'] for r in eligible if not any(dominates(q,r) for q in eligible)],'scope':'exploratory paired measurements; sampled memory is a lower bound, not exact instantaneous peak; no deployment claim'}
    with store.db:rid=store.put_blob(result)
    return {'comparison_id':rid,**result}

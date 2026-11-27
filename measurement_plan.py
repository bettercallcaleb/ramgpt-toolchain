import copy
import fcntl
import os
from pathlib import Path
from registry import digest,fingerprint,read_json
import adaptive
import evaluation
import evidence
import inference
import runtime
import runtime_opt

SCHEMA='ramgpt-measurement-plan-v1'
KV=('f16','q8_0','q4_0')

def init(store):
    store.db.execute('CREATE TABLE IF NOT EXISTS measurement_plans(id TEXT PRIMARY KEY REFERENCES blobs(id), cohort_key TEXT UNIQUE NOT NULL, root TEXT UNIQUE NOT NULL)')
    store.db.execute('CREATE TABLE IF NOT EXISTS measurement_started(contract_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL REFERENCES measurement_plans(id))')

def limits(budgets,strategies,reserve,kv,seed,variants,max_requests,max_tokens):
    if not budgets or len(set(budgets))!=len(budgets) or any(type(x) is not int or not 2<=x<=65536 for x in budgets):raise ValueError('Invalid budgets')
    if not strategies or len(set(strategies))!=len(strategies):raise ValueError('Invalid strategies')
    if type(reserve) is not int or reserve<0 or ('allocated' in strategies and not 0<reserve<min(budgets)):raise ValueError('Invalid reserve')
    if not kv or len(set(kv))!=len(kv) or set(kv)-set(KV):raise ValueError('Unsupported or repeated KV cache')
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Invalid seed')
    if type(variants) is not int or not 1<=variants<=32:raise ValueError('Variants must be 1..32')
    if type(max_requests) is not int or not 1<=max_requests<=10000 or type(max_tokens) is not int or not 1<=max_tokens<=10**8:raise ValueError('Invalid explicit request or completion cap')

def cost(schedule,arms):
    requests=len(schedule)*arms
    tokens=sum(x['max_tokens'] for x in schedule)*arms
    return {'workload_requests':requests,'smoke_requests':arms,'total_completion_requests':requests+arms,'workload_completion_token_cap':tokens,'smoke_completion_token_cap':64*arms,'total_completion_token_cap':tokens+64*arms,'server_sessions':arms}

def check_cost(c,max_requests,max_tokens):
    if c['total_completion_requests']>max_requests:raise ValueError('Request cap exceeded (including smoke requests)')
    if c['total_completion_token_cap']>max_tokens:raise ValueError('Completion token cap exceeded (including smoke requests)')

def args_for(base,model,kv):
    opts={};a=iter(base)
    for k in a:
        if k in opts:raise ValueError('Duplicate runtime option')
        if k in runtime.FLAGS:opts[k]=True
        elif k in runtime.VALUES:opts[k]=next(a)
        else:raise ValueError('Unsupported runtime option')
    opts['-m']=model
    opts['--cache-type-k']=kv;opts['--cache-type-v']=kv
    out=[]
    for k,v in opts.items():
        out.append(k)
        if v is not True:out.append(v)
    return out

def used_prompts(store,manifest,aid,ignore=()):
    prompts={' '.join(t['prompt'].split()) for t in manifest['tasks']}
    for row in store.find():
        if row['contract_id'] in ignore:continue
        s=store.get(row['source_id']);w=s['contract']['workload']
        if s['artifact_sha256']!=aid and 'joint_decision_id' not in w and 'router_decision_id' not in w:continue
        if prompts & {' '.join(t['prompt'].split()) for t in s['manifest']['tasks']}:raise ValueError('Task overlap with prior same-artifact evidence or sealed holdout')

def prepare(reg,store,spec_path,out,seed,variants,budgets,strategies,reserve,kv,max_requests,max_tokens):
    limits(budgets,strategies,reserve,kv,seed,variants,max_requests,max_tokens)
    spec=read_json(spec_path)
    if set(spec)!={'inputs','execution','workload'}:raise ValueError('Expected base experiment specification')
    refs={}
    extra=[]
    for x in spec['inputs']:
        if x['role']=='runtime_dependency':extra.append(x['path'])
        elif x['role'] in ('model','runtime','template','evaluator','dataset','configuration'):
            if x['role'] in refs:raise ValueError('Duplicate base input')
            refs[x['role']]=x['path']
        else:raise ValueError('Unknown input role')
    if not {'model','runtime','template','evaluator'}<=refs.keys():raise ValueError('Require model, runtime, template and evaluator')
    inputs=[fingerprint(refs[k],k) for k in ('model','runtime','template','evaluator')]
    paths={r['role']:r['path'] for r in inputs}
    aid=inputs[0]['sha256']
    if inputs[3]['sha256']!=adaptive.PIN:raise ValueError('Measurement planner requires pinned BudgetFit archive')
    if Path(paths['runtime']).name!='llama-server' or not os.access(paths['runtime'],os.X_OK):raise ValueError('Expected trusted executable llama-server')
    manifest=adaptive.worker(paths['evaluator'],{'action':'generate','seed':seed,'variants':variants,'suite':'core','ladder_variants':1})
    if manifest['seed']!=seed or manifest['variants']!=variants or manifest['suite']!='core':raise ValueError('Unexpected native manifest')
    schedule=evaluation.worker(paths['evaluator'],{'action':'prepare','manifest':manifest,'budgets':budgets,'strategies':strategies,'reserve':reserve})
    c=cost(schedule['requests'],len(kv));check_cost(c,max_requests,max_tokens)
    used_prompts(store,manifest,aid)
    key=digest({'artifact_sha256':aid,'manifest_sha256':digest(manifest)})
    init(store)
    if store.db.execute('SELECT id FROM measurement_plans WHERE cohort_key=?',(key,)).fetchone():raise ValueError('Measurement cohort already planned')
    root=Path(out).resolve()
    if root.exists():raise FileExistsError('Measurement output already exists')
    dep=runtime_opt.dependencies(paths['runtime'])
    here=Path(__file__).resolve().parent
    dep=sorted({str(Path(x).resolve(strict=True)) for x in dep+extra+[str(here/n) for n in ('runtime.py','telemetry.py','evaluator_worker.py','measurement_plan.py')]})
    root.mkdir(parents=True,exist_ok=False)
    dataset=root/'tasks.json';schedule_path=root/'schedule.json'
    evaluation.write_new(dataset,manifest);evaluation.write_new(schedule_path,schedule)
    base=[{'role':r['role'],'path':r['path']} for r in inputs]+[{'role':'dataset','path':str(dataset)},{'role':'configuration','path':str(schedule_path)}]+[{'role':'runtime_dependency','path':p} for p in dep]
    arms=[]
    for cache in kv:
        ex=copy.deepcopy(spec['execution'])
        ex['runtime_args']=args_for(ex['runtime_args'],paths['model'],cache)
        ex['runtime_measurement']='nvml-sampled-v1'
        w={'protocol':'toolchain-inference-v1','seed':seed,'budgets':budgets,'strategies':strategies,'reserve':reserve,'task_count':len(manifest['tasks']),'request_count':len(schedule['requests']),'evaluator':'BudgetFit 0.6.0','measurement_cohort_key':key}
        p=reg.freeze({'inputs':base,'execution':ex,'workload':w})
        if any(r not in p['inputs'] for r in inputs):raise ValueError('Base input changed during preparation')
        rows=inference.schedule(p);evaluation.validate_schedule(p,rows)
        evaluation.write_new(root/(cache+'-contract.json'),p)
        arms.append({'kv':cache,'contract_id':p['contract_id'],'run_path':str(root/(cache+'-run'))})
    body={'schema':SCHEMA,'artifact_sha256':aid,'manifest_sha256':digest(manifest),'schedule_sha256':digest(schedule),'seed':seed,'root':str(root),'arms':arms,'cost':c,'max_requests':max_requests,'max_completion_tokens':max_tokens,'scope':'Bounded local calibration only; one complete session per arm; no deployment or automatic holdout'}
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        pid=store.put_blob(body)
        store.db.execute('INSERT INTO measurement_plans VALUES (?,?,?)',(pid,key,str(root)))
    evaluation.write_new(root/'measurement-plan.json',{'plan_id':pid,**body})
    return {'plan_id':pid,**body,'status':'frozen_not_executed','new_inference_requests':0}

def load(reg,store,pid):
    evidence.checked_id(pid);init(store);p=store.get(pid)
    if p['schema']!=SCHEMA:raise ValueError('Unsupported measurement plan')
    key=digest({'artifact_sha256':p['artifact_sha256'],'manifest_sha256':p['manifest_sha256']})
    if store.db.execute('SELECT cohort_key,root FROM measurement_plans WHERE id=?',(pid,)).fetchone()!=(key,p['root']):raise ValueError('Unregistered measurement plan')
    root=Path(p['root'])
    if root.resolve()!=root or read_json(root/'measurement-plan.json')!={'plan_id':pid,**p}:raise ValueError('Measurement plan file mismatch')
    manifest=read_json(root/'tasks.json');sched=read_json(root/'schedule.json')
    if digest(manifest)!=p['manifest_sha256'] or digest(sched)!=p['schedule_sha256']:raise ValueError('Measurement inputs changed')
    if len({a['kv'] for a in p['arms']})!=len(p['arms']) or not p['arms']:raise ValueError('Invalid measurement arms')
    c=cost(sched['requests'],len(p['arms']))
    if c!=p['cost']:raise ValueError('Measurement cost mismatch')
    check_cost(c,p['max_requests'],p['max_completion_tokens'])
    for arm in p['arms']:
        cache=arm['kv'];cid=arm['contract_id']
        if cache not in KV or arm['run_path']!=str(root/(cache+'-run')):raise ValueError('Invalid measurement run binding')
        contract=reg.contract(cid);reg.verify(cid)
        if contract!=read_json(root/(cache+'-contract.json')):raise ValueError('Measurement contract file mismatch')
        if next(x['sha256'] for x in contract['inputs'] if x['role']=='model')!=p['artifact_sha256']:raise ValueError('Model identity mismatch')
        if contract['workload']['seed']!=p['seed'] or contract['workload']['request_count']!=len(sched['requests']) or contract['workload']['measurement_cohort_key']!=key:raise ValueError('Measurement workload mismatch')
        refs={x['role']:x for x in contract['inputs'] if x['role'] in ('dataset','configuration')}
        if refs['dataset']['path']!=str(root/'tasks.json') or refs['configuration']['path']!=str(root/'schedule.json'):raise ValueError('Measurement input binding mismatch')
        opts=runtime.validate(contract)[1]
        if opts.get('--cache-type-k')!=cache or opts.get('--cache-type-v')!=cache:raise ValueError('Measurement cache mismatch')
        rows=inference.schedule(contract);evaluation.validate_schedule(contract,rows)
    used_prompts(store,manifest,p['artifact_sha256'],[x['contract_id'] for x in p['arms']])
    return p

def preflight(reg,store,p):
    result=[]
    for arm in p['arms']:
        cid=arm['contract_id'];out=Path(arm['run_path']);rows=inference.schedule(reg.contract(cid))
        archived=store.find(contract=cid)
        started=store.db.execute('SELECT plan_id FROM measurement_started WHERE contract_id=?',(cid,)).fetchone()
        if started and started!=(digest(p),):raise ValueError('Measurement start binding mismatch')
        if out.exists():
            if out.resolve()!=out or not started:raise ValueError('Unexpected or unregistered run path')
            sessions=inference.capture_sessions(out)
            complete,_,n=inference.validate_sessions(sessions,rows,cid)
            if n!=1 or len(complete)!=len(rows):raise ValueError('Partial or multi-session measurement; inspect manually, no automatic retry')
            if len(archived)>1:raise ValueError('Ambiguous archived measurement')
        elif archived or started:raise ValueError('Previously started measurement missing bound run directory; no retry')
        result.append((arm,bool(started),archived))
    return result

def run(reg,store,pid,execute=False):
    if execute is not True:raise ValueError('Explicit --execute required; inspect frozen cost first')
    p=load(reg,store,pid);root=Path(p['root'])
    with (root/'.measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=load(reg,store,pid)
        pending=preflight(reg,store,p)
        results=[];new=0;smokes=0
        for arm,started,archived in pending:
            cid=arm['contract_id'];out=arm['run_path']
            if not started:
                with store.db:store.db.execute('INSERT INTO measurement_started VALUES (?,?)',(cid,pid))
                summary=inference.run(reg,cid,out,measurement_authorized=True)
                if summary['status']!='complete' or summary['sessions']!=1:raise ValueError('Expected one complete session')
                new+=summary['new_requests'];smokes+=1
            if not archived:r=store.ingest(reg,cid,out)
            else:
                hit=archived[0]
                snapshot=evidence.snapshot(reg,cid,out)
                if digest(snapshot)!=hit['source_id']:raise ValueError('Archived measurement source mismatch')
                r=hit
            results.append({'kv':arm['kv'],'contract_id':cid,'source_id':r['source_id'],'evaluation_id':r['evaluation_id']})
        result={'plan_id':pid,'status':'measurement_complete','arms':results,'new_inference_requests':new,'new_smoke_requests':smokes,'deployment_authorized':False}
        runtime.atomic(root/'measurement-result.json',result)
        return result

import json
import math
import subprocess
import sys
from pathlib import Path
from registry import digest,fingerprint,read_json,now
import evidence
import evaluation
import optimize

ENGINE='BudgetFit 0.6.0 Router'
PIN='367ceb63bf9ff5f26671d7316cf87749d7f8c4cf834b1c0394dbfddf4c245eeb'
DEFAULT={'min_accuracy':.5,'tolerance':.02,'max_validation_drop':.1,'min_bucket':2,'min_savings':.05}
SCHEMA='ramgpt-adaptive-router-v1'

def adapter_hash():
    return digest([fingerprint(Path(__file__).with_name(n),'configuration')['sha256'] for n in ('adaptive.py','router_worker.py')])

def worker(archive,data):
    path=Path(archive).resolve(strict=True)
    script=Path(__file__).with_name('router_worker.py').resolve()
    before=adapter_hash()
    if fingerprint(path,'evaluator')['sha256']!=PIN:raise ValueError('Untrusted BudgetFit archive')
    p=subprocess.run([sys.executable,'-I',str(script),str(path)],input=json.dumps(data,allow_nan=False),text=True,capture_output=True,timeout=180)
    if p.returncode:raise ValueError('Router worker: '+p.stderr[-4000:])
    if adapter_hash()!=before or fingerprint(path,'evaluator')['sha256']!=PIN:raise ValueError('Router engine changed')
    return json.loads(p.stdout)

def thresholds(t):
    if set(t)!=set(DEFAULT) or type(t['min_bucket']) is not int or t['min_bucket']<2:raise ValueError('Invalid router thresholds')
    for k in set(t)-{'min_bucket'}:
        if type(t[k]) not in (int,float) or not math.isfinite(t[k]) or not 0<=t[k]<=1:raise ValueError('Invalid threshold: '+k)
    if t['min_savings']<=0:raise ValueError('Require positive savings')
    return t

def archive_for(plan):
    return next(x['path'] for x in plan['inputs'] if x['role']=='evaluator')

def historical(archive,folder):
    p=Path(folder)
    def load(n):return read_json(p/(n+'.json'))
    plan=load('router_plan');report=load('router_report');frozen=load('router_selection')
    cal=load('calibration_tasks');val=load('validation_tasks')
    cr=load('calibration_rows');vr=load('validation_rows')
    result=worker(archive,{'action':'historical','calibration':cal,'tasks':val,'configs':plan['configs'],'thresholds':plan['thresholds'],'cal_rows':cr,'rows':vr})
    if (result['policy'],result['fixed'],result['profiles'],result['validation'],result['passed'])!=(frozen['policy'],frozen['fixed'],report['calibration'],report['validation'],report['status']=='validation passed'):raise ValueError('Historical parity mismatch')
    return {'status':'historical_parity_passed','requests':len(cr)+len(vr),'new_inference_requests':0,'passed':result['passed'],'validation':result['validation'],'scope':'unbound historical records; not controlled evidence'}

def freeze(store,eid,archive,seed,t):
    thresholds(t)
    source,report=optimize.load_evaluation(store,eid)
    if evidence.native_report(source,archive)!=report:raise ValueError('Calibration replay mismatch')
    cal=source['manifest'];w=source['contract']['workload']
    if type(seed) is not int or seed==cal['seed']:raise ValueError('Require independent seed')
    configs=[[p['budget'],p['strategy'],p['reserve']] for p in optimize.profiles(report)]
    if len(configs)<2 or len(cal['tasks'])<2*t['min_bucket']:raise ValueError('Insufficient calibration')
    selection=worker(archive,{'action':'train','manifest':cal,'configs':configs,'rows':report['rows'],'thresholds':t})
    if selection['policy'] is None:raise ValueError('No adaptive candidate')
    manifest=worker(archive,{'action':'generate','seed':seed,**{k:cal[k] for k in ('variants','suite','ladder_variants')}})
    prompts=lambda m:{' '.join(x['prompt'].split()) for x in m['tasks']}
    if prompts(cal)&prompts(manifest):raise ValueError('Calibration/holdout prompt overlap')
    if any(x['receipt']['finished_at']>now() for x in source['sessions']):raise ValueError('Calibration timing mismatch')
    decision={'schema':SCHEMA,'calibration_evaluation_id':eid,'calibration_source_id':digest(source),'selection':selection,'thresholds':t,'validation_seed':seed,'manifest_digest':digest(manifest),'context':optimize.context(source,report),'adapter_sha256':adapter_hash(),'reserve':w['reserve'],'frozen_at':now()}
    with store.db:
        store.db.execute('CREATE TABLE IF NOT EXISTS router_decisions(id TEXT PRIMARY KEY REFERENCES blobs(id),calibration_evaluation_id TEXT NOT NULL UNIQUE REFERENCES evaluations(id),contract_id TEXT UNIQUE,run_path TEXT)')
        if store.db.execute('SELECT id FROM router_decisions WHERE calibration_evaluation_id=?',(eid,)).fetchone():raise ValueError('Router decision already frozen')
        did=store.put_blob(decision)
        store.db.execute('INSERT INTO router_decisions VALUES (?,?,NULL,NULL)',(did,eid))
    return {'decision_id':did,'decision':decision,'manifest':manifest}

def prepare(archive,manifest,decision):
    if decision['schema']!=SCHEMA or decision['adapter_sha256']!=adapter_hash() or digest(manifest)!=decision['manifest_digest'] or manifest['seed']!=decision['validation_seed']:raise ValueError('Frozen router mismatch')
    thresholds(decision['thresholds'])
    return worker(archive,{'action':'prepare','manifest':manifest,'selection':decision['selection']})

def validate_schedule(plan,rows):
    w=plan['workload'];d=w['router_decision']
    if w['evaluator']!=ENGINE or digest(d)!=w['router_decision_id'] or w['seed']!=d['validation_seed'] or w['reserve']!=d['reserve']:raise ValueError('Router contract mismatch')
    refs={r['role']:r for r in plan['inputs']}
    for role in ('dataset','evaluator'):
        if fingerprint(refs[role]['path'],role)!=refs[role]:raise ValueError('Frozen input changed')
    manifest=read_json(refs['dataset']['path'])
    expected=prepare(refs['evaluator']['path'],manifest,d)['requests']
    if [{k:v for k,v in x.items() if k!='body'} for x in rows]!=expected:raise ValueError('Sparse schedule differs from native policy')
    ctx=d['context']
    actual=sorted([{k:r[k] for k in ('role','sha256','bytes')} for r in plan['inputs'] if r['role'] not in ('dataset','configuration')],key=lambda r:(r['role'],r['sha256']))
    if actual!=ctx['inputs'] or plan['execution']!=ctx['execution']:raise ValueError('Router execution context mismatch')
    return True

def native_report(source,archive):
    rows=evidence.validate_source(source);p=source['contract'];w=p['workload'];d=w['router_decision']
    if digest(d)!=w['router_decision_id'] or d['adapter_sha256']!=adapter_hash():raise ValueError('Router decision identity mismatch')
    prepare(archive,source['manifest'],d)
    if any(s['receipt']['started_at']<=d['frozen_at'] for s in source['sessions']):raise ValueError('Validation predates decision')
    if source['response_identity']!=d['context']['backend']:raise ValueError('Backend identity mismatch')
    for s in source['sessions']:
        g=s['receipt']['gpu_before']
        if [g[k] for k in ('uuid','name','driver','total_mib')]!=d['context']['gpu']:raise ValueError('GPU identity mismatch')
    result=worker(archive,{'action':'score','manifest':source['manifest'],'selection':d['selection'],'rows':rows,'reserve':d['reserve'],'thresholds':d['thresholds']})
    return {'schema':SCHEMA,'contract_id':p['contract_id'],'decision_id':w['router_decision_id'],'artifact_sha256':source['artifact_sha256'],'evaluator_sha256':PIN,'adapter_sha256':adapter_hash(),'requests':len(rows),'status':'heldout_pilot_passed' if result['passed'] else 'heldout_failed','scope':'eight-task engineering pilot; no production deployment claim',**result}

def bind(store,did,cid):
    with store.db:
        row=store.db.execute('SELECT contract_id FROM router_decisions WHERE id=?',(did,)).fetchone()
        if row is None or row[0] not in (None,cid):raise ValueError('Different holdout already bound')
        store.db.execute('UPDATE router_decisions SET contract_id=? WHERE id=?',(cid,did))

def check_binding(store,plan):
    w=plan['workload'];did=w['router_decision_id'];d=w['router_decision']
    row=store.db.execute('SELECT calibration_evaluation_id,contract_id FROM router_decisions WHERE id=?',(did,)).fetchone()
    if row!=(d['calibration_evaluation_id'],plan['contract_id']) or store.get(did)!=d:raise ValueError('Unbound router decision or holdout')
    source,report=optimize.load_evaluation(store,row[0])
    if digest(source)!=d['calibration_source_id'] or optimize.context(source,report)!=d['context']:raise ValueError('Calibration evidence mismatch')

def authorize(db,plan,out):
    store=evidence.Store(db)
    try:
        with store.db:
            check_binding(store,plan)
            did=plan['workload']['router_decision_id']
            path=str(Path(out).resolve())
            store.db.execute('UPDATE router_decisions SET run_path=? WHERE id=? AND run_path IS NULL',(path,did))
            if store.db.execute('SELECT run_path FROM router_decisions WHERE id=?',(did,)).fetchone()!=(path,):raise ValueError('Router holdout already assigned to another run path')
    finally:store.close()

def plan_holdout(reg,store,eid,seed,t,out):
    out=Path(out).resolve()
    if out.exists():raise ValueError('Holdout directory already exists')
    source,report=optimize.load_evaluation(store,eid)
    cal=source['contract'];archive=archive_for(cal)
    result=freeze(store,eid,archive,seed,t)
    d=result['decision'];did=result['decision_id'];manifest=result['manifest']
    out.mkdir(parents=True)
    evaluation.write_new(out/'decision.json',result)
    dataset=out/'tasks.json';schedule=out/'schedule.json'
    evaluation.write_new(dataset,manifest)
    requests=prepare(archive,manifest,d)
    evaluation.write_new(schedule,requests)
    spec={'inputs':[{'role':r['role'],'path':str(dataset if r['role']=='dataset' else schedule if r['role']=='configuration' else r['path'])} for r in cal['inputs']],
          'execution':cal['execution'],'workload':{**cal['workload'],'seed':seed,'evaluator':ENGINE,'task_count':len(manifest['tasks']),'request_count':len(requests['requests']),'router_decision':d,'router_decision_id':did}}
    evaluation.write_new(out/'spec.json',spec)
    plan=reg.freeze(spec)
    import inference
    validate_schedule(plan,inference.schedule(plan))
    bind(store,did,plan['contract_id'])
    evaluation.write_new(out/'contract.json',plan)
    return {'status':'router_holdout_frozen','contract_id':plan['contract_id'],'decision_id':did,'requests':len(requests['requests']),'selection':d['selection'],'out':str(out)}

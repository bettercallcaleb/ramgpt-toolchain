import math
from pathlib import Path
from registry import digest,now
import evidence

SCHEMA='ramgpt-fixed-optimization-v1'
OBJECTIVES={'tokens':'mean_completion_tokens','latency':'mean_latency_s'}

def config(p):
    return (p['budget'],p['strategy'],p['reserve'])

def limits(policy):
    if set(policy)!={'objective','min_accuracy','accuracy_tolerance','max_validation_drop','min_savings','max_truncation','min_tasks','reference'}:
        raise ValueError('Incomplete optimization constraints')
    if policy['objective'] not in OBJECTIVES:raise ValueError('Unknown objective')
    for k in ('min_accuracy','accuracy_tolerance','max_validation_drop','min_savings','max_truncation'):
        v=policy[k]
        if type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1:raise ValueError('Invalid constraint: '+k)
    if policy['min_savings']<=0 or type(policy['min_tasks']) is not int or policy['min_tasks']<8:
        raise ValueError('Require positive savings and at least eight tasks')
    r=policy['reference']
    if not isinstance(r,list) or len(r)!=3 or type(r[0]) is not int or r[0]<1 or not isinstance(r[1],str) or type(r[2]) is not int or r[2]<0:raise ValueError('Invalid fixed reference')
    return policy

def profiles(report):
    out=[p['budgetfit'] for p in report['profiles']]
    if not out or len({config(p) for p in out})!=len(out):raise ValueError('Missing or duplicate profiles')
    n=out[0]['n']
    for p in out:
        if type(n) is not int or n<1 or p['n']!=n or p['attempted']!=n or p['complete'] is not True:
            raise ValueError('Incomplete or unequal profile denominators')
        if type(p['joint_success']) is not int or not 0<=p['joint_success']<=n or abs(p['accuracy']-p['joint_success']/n)>1e-12:
            raise ValueError('Invalid joint accuracy')
        for key in ('accuracy','semantic_accuracy','truncation_rate','mean_completion_tokens','mean_latency_s'):
            v=p[key]
            if type(v) not in (int,float) or not math.isfinite(v) or v<0 or key in ('accuracy','semantic_accuracy','truncation_rate') and v>1:
                raise ValueError('Invalid metric: '+key)
    return out

def dominates(a,b):
    return (a['accuracy']>=b['accuracy'] and a['mean_completion_tokens']<=b['mean_completion_tokens'] and a['mean_latency_s']<=b['mean_latency_s']
            and (a['accuracy']>b['accuracy'] or a['mean_completion_tokens']<b['mean_completion_tokens'] or a['mean_latency_s']<b['mean_latency_s']))

def savings(candidate,reference,key):
    return 1-candidate[key]/reference[key] if reference[key]>0 else None

def select(report,policy):
    limits(policy);ps=profiles(report);key=OBJECTIVES[policy['objective']]
    reference=next((p for p in ps if config(p)==tuple(policy['reference'])),None)
    if reference is None:raise ValueError('Fixed reference not present')
    best=max(p['accuracy'] for p in ps)
    floor=max(policy['min_accuracy'],reference['accuracy']-policy['accuracy_tolerance'],best-policy['accuracy_tolerance'])
    frontier=[p for p in ps if not any(dominates(q,p) for q in ps)]
    candidates=[];reasons=[]
    for p in ps:
        why=[];gain=savings(p,reference,key)
        if config(p)==config(reference):why.append('reference')
        if p['n']<policy['min_tasks']:why.append('insufficient_tasks')
        if p['accuracy']<=0 or p['accuracy']+1e-12<floor:why.append('accuracy_floor')
        if p['truncation_rate']>policy['max_truncation']:why.append('truncation_limit')
        if gain is None or gain+1e-12<policy['min_savings']:why.append('insufficient_savings')
        if not why:candidates.append(p)
        reasons.append({'config':list(config(p)),'reasons':why,'objective_savings':gain})
    selected=min(candidates,key=lambda p:(p[key],-p['accuracy'],p['mean_completion_tokens'],p['mean_latency_s'],config(p))) if candidates else None
    return {'status':'candidate_selected' if selected else 'no_eligible_candidate','reference':reference,'candidate':selected,
            'accuracy_floor':floor,'pareto_frontier':frontier,'screening':reasons}

def load_evaluation(store,eid):
    matches=[x for x in store.find() if x['evaluation_id']==evidence.checked_id(eid)]
    if len(matches)!=1:raise ValueError('Evaluation not indexed')
    row=matches[0];source=store.get(row['source_id']);derived=store.get(eid)
    if derived['schema']!=evidence.DERIVED or derived['source_id']!=row['source_id']:raise ValueError('Invalid evaluation')
    evidence.validate_source(source)
    return source,derived['report']

def context(source,report):
    p=source['contract']
    refs=[{k:r[k] for k in ('role','sha256','bytes')} for r in p['inputs'] if r['role'] not in ('dataset','configuration')]
    refs.sort(key=lambda r:(r['role'],r['sha256']))
    observed=[]
    for session in source['sessions']:
        gpu=session['receipt']['gpu_before']
        observed.append((gpu['uuid'],gpu['name'],gpu['driver'],gpu['total_mib']))
    if len(set(observed))!=1:raise ValueError('Mixed observed GPU environment')
    return {'inputs':refs,'execution':p['execution'],'backend':source['response_identity'],
            'gpu':list(observed[0]),'adapter_sha256':report['adapter_sha256'],'evaluator_sha256':report['evaluator_sha256']}

def freeze(store,eid,policy,validation_seed=None):
    source,report=load_evaluation(store,eid)
    result=select(report,policy)
    if validation_seed is None:validation_seed=source['manifest']['seed']+1
    if type(validation_seed) is not int or validation_seed==source['manifest']['seed']:raise ValueError('Require independent validation seed')
    decision={'schema':SCHEMA,'calibration_evaluation_id':eid,'calibration_source_id':digest(source),
              'policy':policy,'result':result,'context':context(source,report),'validation_seed':validation_seed,'frozen_at':now(),
              'scope':'fixed budget/strategy exploratory selection; no deployment'}
    with store.db:
        did=store.put_blob(decision)
        store.db.execute('CREATE TABLE IF NOT EXISTS optimization_decisions(id TEXT PRIMARY KEY REFERENCES blobs(id),calibration_evaluation_id TEXT NOT NULL REFERENCES evaluations(id))')
        store.db.execute('INSERT INTO optimization_decisions VALUES (?,?)',(did,eid))
    return {'decision_id':did,**decision}

def validate_metrics(candidate,reference,calibration,policy):
    key=OBJECTIVES[policy['objective']];gain=savings(candidate,reference,key)
    failures=[]
    if candidate['n']!=reference['n'] or candidate['n']<policy['min_tasks']:failures.append('insufficient_tasks')
    if candidate['accuracy']<=0 or candidate['accuracy']+1e-12<policy['min_accuracy']:failures.append('absolute_accuracy')
    if candidate['accuracy']+1e-12<reference['accuracy']-policy['accuracy_tolerance']:failures.append('reference_accuracy_drop')
    if candidate['accuracy']+1e-12<calibration['accuracy']-policy['max_validation_drop']:failures.append('calibration_accuracy_drop')
    if candidate['truncation_rate']>policy['max_truncation']:failures.append('truncation_limit')
    if gain is None or gain+1e-12<policy['min_savings']:failures.append('insufficient_savings')
    return {'status':'heldout_pilot_passed' if not failures else 'heldout_failed','failures':failures,'objective_savings':gain}

def validate(store,did,eid):
    decision=store.get(did)
    if decision.get('schema')!=SCHEMA:raise ValueError('Unsupported decision')
    row=store.db.execute('SELECT calibration_evaluation_id FROM optimization_decisions WHERE id=?',(did,)).fetchone()
    if row is None or row[0]!=decision['calibration_evaluation_id']:raise ValueError('Unregistered decision')
    cs,cr=load_evaluation(store,row[0])
    policy=limits(decision['policy'])
    if digest(cs)!=decision['calibration_source_id'] or select(cr,policy)!=decision['result'] or context(cs,cr)!=decision['context']:raise ValueError('Calibration decision mismatch')
    selected=decision['result']['candidate']
    if selected is None:raise ValueError('No frozen candidate to validate')
    hs,hr=load_evaluation(store,eid)
    if context(hs,hr)!=decision['context']:raise ValueError('Validation environment mismatch')
    if hs['manifest']['seed']!=decision['validation_seed'] or hs['manifest']['seed']==cs['manifest']['seed']:raise ValueError('Wrong frozen validation seed')
    for key in ('version','suite','variants','ladder_variants'):
        if hs['manifest'][key]!=cs['manifest'][key]:raise ValueError('Validation task generator mismatch: '+key)
    def prompts(s):return {' '.join(t['prompt'].split()) for t in s['manifest']['tasks']}
    if prompts(cs)&prompts(hs):raise ValueError('Calibration/validation prompt overlap')
    if any(s['receipt']['started_at']<=decision['frozen_at'] for s in hs['sessions']):raise ValueError('Validation execution predates frozen decision')
    if any(s['receipt']['finished_at']>decision['frozen_at'] for s in cs['sessions']):raise ValueError('Calibration execution not finished before selection')
    ps=profiles(hr);lookup={config(p):p for p in ps}
    ck=config(selected);rk=tuple(policy['reference'])
    if set(lookup)!={ck,rk}:raise ValueError('Validation must contain exactly the frozen candidate and reference')
    c=lookup[ck];r=lookup[rk]
    paired={}
    for d in hr['rows']:
        k=(d['budget'],d['strategy'],d['reserve'])
        if k in (ck,rk):
            slot=paired.setdefault(d['id'],{})
            if k in slot:raise ValueError('Duplicate validation pair')
            slot[k]=d['grade']['joint_success']
    if len(paired)!=c['n'] or any(set(p)!={ck,rk} for p in paired.values()):raise ValueError('Incomplete paired validation')
    wins=sum(p[ck] and not p[rk] for p in paired.values());losses=sum(p[rk] and not p[ck] for p in paired.values())
    result={'schema':'ramgpt-optimization-validation-v1','decision_id':did,'validation_evaluation_id':eid,
            'validation_source_id':digest(hs),'candidate':c,'reference':r,'paired':{'n':len(paired),'wins':wins,'losses':losses,'ties':len(paired)-wins-losses},
            **validate_metrics(c,r,selected,policy),'deployment_authorized':False,'scope':'held-out instances of same task families; pilot only'}
    with store.db:
        store.db.execute('CREATE TABLE IF NOT EXISTS optimization_validations(decision_id TEXT PRIMARY KEY REFERENCES optimization_decisions(id),validation_id TEXT NOT NULL REFERENCES blobs(id),evaluation_id TEXT NOT NULL REFERENCES evaluations(id))')
        store.db.execute('BEGIN IMMEDIATE')
        prior=store.db.execute('SELECT validation_id,evaluation_id FROM optimization_validations WHERE decision_id=?',(did,)).fetchone()
        if prior is not None:
            if prior[1]!=eid or store.get(prior[0])!=result:raise ValueError('Decision already validated; no reselection or alternate holdout')
            vid=prior[0]
        else:
            vid=store.put_blob(result)
            store.db.execute('INSERT INTO optimization_validations VALUES (?,?,?)',(did,vid,eid))
    return {'validation_id':vid,**result}

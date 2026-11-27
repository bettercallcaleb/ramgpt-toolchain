import copy
import math
from pathlib import Path
from registry import digest,now
import adaptive
import evidence
import evaluation
import optimize
import runtime
import telemetry
import weight

SCHEMA='ramgpt-joint-selection-v1'
TUNABLE={'-c','-ngl','--cache-type-k','--cache-type-v','--flash-attn','--batch-size','--ubatch-size'}
OBJECTIVES={'memory':'sampled_peak_bytes','latency':'mean_latency_s','tokens':'mean_completion_tokens'}
POLICY={'objective','min_accuracy','max_latency_s','max_sampled_peak_bytes','min_free_bytes','max_mean_tokens','max_truncation','min_tasks','max_validation_drop','accuracy_tolerance','fidelity_report_id','max_kl','max_artifact_bytes','max_sample_gap_s'}

def policy(p):
    if set(p)!=POLICY or p['objective'] not in OBJECTIVES:raise ValueError('Incomplete joint policy')
    for k in ('min_accuracy','max_truncation','max_validation_drop','accuracy_tolerance'):weight.number(p[k],0,1)
    for k in ('max_latency_s','max_mean_tokens','max_sample_gap_s'):weight.number(p[k],1e-12)
    for k in ('max_sampled_peak_bytes','min_free_bytes','max_artifact_bytes','min_tasks'):weight.size(p[k])
    if p['min_tasks']<8:raise ValueError('Require at least eight tasks')
    if (p['fidelity_report_id'] is None)!=(p['max_kl'] is None):raise ValueError('Specify both fidelity cohort and KL limit, or neither')
    if p['max_kl'] is not None:
        evidence.checked_id(p['fidelity_report_id']);weight.number(p['max_kl'])
    return p

def init(store):
    store.db.executescript("""
    CREATE TABLE IF NOT EXISTS joint_decisions(id TEXT PRIMARY KEY REFERENCES blobs(id),cohort_key TEXT UNIQUE NOT NULL,holdout_id TEXT REFERENCES blobs(id),validation_id TEXT REFERENCES blobs(id));
    CREATE TABLE IF NOT EXISTS joint_bindings(contract_id TEXT PRIMARY KEY,decision_id TEXT NOT NULL REFERENCES joint_decisions(id),arm TEXT NOT NULL,run_path TEXT NOT NULL UNIQUE,UNIQUE(decision_id,arm));
    """)

def fidelity(store,p):
    if p['fidelity_report_id'] is None:return {}
    bid=p['fidelity_report_id']
    report=store.get(bid)
    weight.select_fidelity(store,bid,p['max_kl'],p['max_artifact_bytes'])
    return {r['artifact_sha256']:{'report_id':bid,'reference_sha256':r['reference_sha256'],'mean_kl':r['metrics']['mean_kl_bf16_to_artifact'],'artifact_bytes':r['artifact_bytes']} for r in report['records']}

def execution_context(source,report):
    c=optimize.context(source,report)
    plan=source['contract'];opts=runtime.validate(plan)[1]
    opts={k:v for k,v in opts.items() if k not in ('-m','--port')}
    c['execution']={k:v for k,v in c['execution'].items() if k!='runtime_args'}
    c['inputs']=[r for r in c['inputs'] if r['role']!='model']
    c['options']={k:v for k,v in opts.items() if k not in TUNABLE}
    c['workload_protocol']=plan['workload']['protocol']
    return c,{k:v for k,v in opts.items() if k in TUNABLE}

def memory(source):
    peaks=[];free=[];gaps=[];libs=None
    for s in source['sessions']:
        r=s['receipt'];t=r.get('memory_trace')
        if t is None:return None
        if t['method']!='NVML device-wide sampled high-water mark; lower bound on instantaneous peak':raise ValueError('Unsupported memory measurement')
        samples=t['samples']
        if len(samples)<2:raise ValueError('Incomplete memory samples')
        for x in samples:
            for k in ('used_bytes','reserved_bytes','free_bytes'):weight.number(x[k])
            if x['used_bytes']+x['reserved_bytes']+x['free_bytes']!=x['total_bytes'] or x['total_bytes']!=t['total_bytes']:raise ValueError('Invalid NVML accounting')
            weight.number(x['elapsed_s'])
        gaps0=[b['elapsed_s']-a['elapsed_s'] for a,b in zip(samples,samples[1:])]
        if min(gaps0)<=0 or max(gaps0)!=t['max_sample_gap_s'] or max(x['used_bytes'] for x in samples)!=t['sampled_peak_bytes'] or min(x['free_bytes'] for x in samples)!=t['minimum_free_bytes']:raise ValueError('Memory trace summary mismatch')
        mapped=r.get('mapped_libraries',[])
        bound={x['path']:x['sha256'] for x in source['contract']['inputs'] if x['role']=='runtime_dependency'}
        if not mapped or any(not x['frozen'] or bound.get(x['path'])!=x['sha256'] for x in mapped):return None
        if libs is not None and mapped!=libs:raise ValueError('Mixed mapped libraries')
        libs=mapped;peaks.append(t['sampled_peak_bytes']);free.append(t['minimum_free_bytes']);gaps.append(t['max_sample_gap_s'])
    return {'sampled_peak_bytes':max(peaks),'minimum_free_bytes':min(free),'max_sample_gap_s':max(gaps),'mapped_libraries':libs,'measurement_scope':'entire session set; conservative attribution to each observed profile'}

def load(store,eid,p,fids):
    source,report=optimize.load_evaluation(store,eid)
    if report['schema']!=evaluation.SCHEMA:return source,[],None,'unsupported_adaptive_profile'
    c,tuning=execution_context(source,report)
    ps=optimize.profiles(report);m=memory(source)
    model=next(x for x in source['contract']['inputs'] if x['role']=='model')
    f=fids.get(model['sha256'])
    if f and f['artifact_bytes']!=model['bytes']:raise ValueError('Fidelity artifact size mismatch')
    rows=[]
    for profile in ps:
        cfg=list(optimize.config(profile))
        key={'artifact_sha256':model['sha256'],'runtime':tuning,'fixed':cfg,'context_sha256':digest(c)}
        r={'candidate_id':digest(key),'evaluation_id':eid,'source_id':digest(source),'contract_id':source['contract']['contract_id'],'artifact_sha256':model['sha256'],'artifact_bytes':model['bytes'],'runtime':tuning,'fixed':cfg,'context_sha256':digest(c),'metrics':profile,'memory':m,'fidelity':f}
        rows.append(r)
    return source,rows,c,None

def screen(r,p):
    m=r['metrics'];mem=r['memory'];f=r['fidelity'];why=[]
    if m['n']<p['min_tasks']:why.append('insufficient_tasks')
    if m['accuracy']<p['min_accuracy']:why.append('accuracy_floor')
    if m['mean_latency_s']>p['max_latency_s']:why.append('latency_limit')
    if m['mean_completion_tokens']>p['max_mean_tokens']:why.append('token_limit')
    if m['truncation_rate']>p['max_truncation']:why.append('truncation_limit')
    if r['artifact_bytes']>p['max_artifact_bytes']:why.append('artifact_size_limit')
    if mem is None:why.append('missing_verified_memory')
    else:
        if mem['sampled_peak_bytes']>p['max_sampled_peak_bytes']:why.append('sampled_vram_limit')
        if mem['minimum_free_bytes']<p['min_free_bytes']:why.append('minimum_free_memory')
        if mem['max_sample_gap_s']>p['max_sample_gap_s']:why.append('telemetry_gap')
    if p['max_kl'] is not None:
        if f is None:why.append('missing_matched_fidelity')
        elif f['mean_kl']>p['max_kl']:why.append('fidelity_limit')
    return why

def vector(r,p):
    m=r['metrics']
    v=(-m['accuracy'],r['memory']['sampled_peak_bytes'],m['mean_latency_s'],m['mean_completion_tokens'],r['artifact_bytes'])
    return v+(r['fidelity']['mean_kl'],) if p['max_kl'] is not None else v

def frontier(rows,p):
    def dominates(a,b):
        x,y=vector(a,p),vector(b,p)
        return all(i<=j for i,j in zip(x,y)) and x!=y
    return [r for r in rows if not any(dominates(q,r) for q in rows)]

def rank_key(r,p):
    v=r['memory']['sampled_peak_bytes'] if p['objective']=='memory' else r['metrics'][OBJECTIVES[p['objective']]]
    return (v,vector(r,p),r['candidate_id'])

def select(store,eids,anchor_eid,anchor_fixed,p):
    policy(p);fids=fidelity(store,p)
    if len(eids)!=len(set(eids)) or anchor_eid not in eids:raise ValueError('Invalid evaluation cohort')
    a,ar,ac,err=load(store,anchor_eid,p,fids)
    if err:raise ValueError('Anchor must be fixed evaluation')
    baseline=next((r for r in ar if r['fixed']==anchor_fixed),None)
    if baseline is None:raise ValueError('Missing anchor configuration')
    rows=[];excluded=[];seen=set()
    for eid in sorted(eids):
        s,rs,c,err=load(store,eid,p,fids)
        if err:excluded.append({'evaluation_id':eid,'reason':err});continue
        if s['manifest']!=a['manifest'] or c!=ac:
            excluded.append({'evaluation_id':eid,'reason':'incompatible_task_or_execution_context'});continue
        for r in rs:
            if r['candidate_id'] in seen:raise ValueError('Duplicate observed configuration; choose one preregistered evaluation')
            seen.add(r['candidate_id'])
            reasons=screen(r,p)
            if r['artifact_sha256']!=baseline['artifact_sha256'] and (r['fidelity'] is None or baseline['fidelity'] is None):reasons.append('unmatched_weight_reference')
            if r['metrics']['accuracy']+1e-12<baseline['metrics']['accuracy']-p['accuracy_tolerance']:reasons.append('anchor_accuracy_drop')
            if r['memory'] and baseline['memory'] and r['memory']['mapped_libraries']!=baseline['memory']['mapped_libraries']:reasons.append('mapped_library_mismatch')
            rows.append({**r,'rejections':reasons})
    eligible=[r for r in rows if not r['rejections']]
    pf=frontier(eligible,p);chosen=min(pf,key=lambda r:rank_key(r,p)) if pf else None
    observed={r['artifact_sha256'] for r in rows}
    missing=[{'artifact_sha256':h,'reason':'no_matched_managed_configuration'} for h in sorted(set(fids)-observed)]
    return {'status':'candidate_selected' if chosen else 'no_eligible_candidate','baseline':baseline,'selected':chosen,'pareto':[r['candidate_id'] for r in pf],'screening':rows,'excluded_evaluations':excluded,'unmeasured_fidelity_candidates':missing,'manifest_sha256':digest(a['manifest']),'context_sha256':digest(ac)}

def freeze(store,eids,anchor_eid,anchor_fixed,p,seed):
    init(store)
    result=select(store,eids,anchor_eid,anchor_fixed,p)
    source,_=optimize.load_evaluation(store,anchor_eid)
    if type(seed) is not int or seed==source['manifest']['seed']:raise ValueError('Independent seed required')
    key=digest({'evaluation_ids':sorted(eids),'anchor_eid':anchor_eid,'anchor_fixed':anchor_fixed,'policy':p})
    with store.db:
        prior=store.db.execute('SELECT id FROM joint_decisions WHERE cohort_key=?',(key,)).fetchone()
        if prior:
            d=store.get(prior[0])
            if d['validation_seed']!=seed or d['result']!=result:raise ValueError('Cohort already frozen')
            return {'decision_id':prior[0],**d}
        for row in store.find():
            prior=store.get(row['source_id'])
            if prior['manifest']['seed']==seed and all(prior['manifest'][k]==source['manifest'][k] for k in ('version','suite','variants','ladder_variants')):raise ValueError('Validation seed already observed')
        d={'schema':SCHEMA,'evaluation_ids':sorted(eids),'anchor_eid':anchor_eid,'anchor_fixed':anchor_fixed,'policy':p,'validation_seed':seed,'result':result,'frozen_at':now(),'scope':'observed fixed configurations only; no deployment authorization'}
        did=store.put_blob(d)
        store.db.execute('INSERT INTO joint_decisions(id,cohort_key) VALUES (?,?)',(did,key))
    return {'decision_id':did,**d}

def decision(store,did):
    d=store.get(did)
    if d['schema']!=SCHEMA or store.db.execute('SELECT id FROM joint_decisions WHERE id=?',(did,)).fetchone()!=(did,):raise ValueError('Unregistered joint decision')
    if select(store,d['evaluation_ids'],d['anchor_eid'],d['anchor_fixed'],d['policy'])!=d['result']:raise ValueError('Joint decision replay mismatch')
    return d

def prepare(reg,store,did,out,bounded_workflow=False):
    init(store);d=decision(store,did)
    chosen=d['result']['selected'];base=d['result']['baseline']
    if chosen is None or chosen['candidate_id']==base['candidate_id']:raise ValueError('No distinct frozen challenger')
    if store.db.execute('SELECT holdout_id FROM joint_decisions WHERE id=?',(did,)).fetchone()[0] is not None:raise ValueError('Holdout already bound')
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    a,_=optimize.load_evaluation(store,base['evaluation_id'])
    original=a['manifest'];archive=next(r['path'] for r in a['contract']['inputs'] if r['role']=='evaluator')
    manifest=adaptive.worker(archive,{'action':'generate','seed':d['validation_seed'],'variants':original['variants'],'suite':original['suite'],'ladder_variants':original['ladder_variants']})
    for k in ('version','suite','variants','ladder_variants'):
        if manifest[k]!=original[k]:raise ValueError('Holdout generator mismatch')
    if manifest['seed']!=d['validation_seed']:raise ValueError('Holdout seed mismatch')
    def prompts(m):return {' '.join(t['prompt'].split()) for t in m['tasks']}
    for row in store.find():
        old=store.get(row['source_id'])['manifest']
        if prompts(old)&prompts(manifest):raise ValueError('Holdout prompt overlap with prior evidence')
    dataset=out/'tasks.json';evaluation.write_new(dataset,manifest)
    arms=[]
    for name,r in [('baseline',base),('candidate',chosen)]:
        old,_=optimize.load_evaluation(store,r['evaluation_id']);cal=old['contract']
        reg.verify(cal['contract_id'])
        budget,strategy,reserve=r['fixed']
        schedule=out/(name+'-schedule.json')
        evaluation.prepare(dataset,archive,[budget],[strategy],reserve,schedule)
        work={**cal['workload'],'seed':d['validation_seed'],'budgets':[budget],'strategies':[strategy],'reserve':reserve,'request_count':len(manifest['tasks']),'task_count':len(manifest['tasks']),'joint_decision_id':did,'joint_arm':name}
        if bounded_workflow:work['bounded_workflow']=True
        spec={'inputs':[{'role':x['role'],'path':str(dataset if x['role']=='dataset' else schedule if x['role']=='configuration' else x['path'])} for x in cal['inputs']],'execution':cal['execution'],'workload':work}
        plan=reg.freeze(spec)
        import inference
        evaluation.validate_schedule(plan,inference.schedule(plan))
        evaluation.write_new(out/(name+'-contract.json'),plan)
        arms.append({'arm':name,'candidate_id':r['candidate_id'],'contract_id':plan['contract_id'],'run_path':str(out/(name+'-run'))})
    holdout={'schema':'ramgpt-joint-holdout-v1','decision_id':did,'manifest_sha256':digest(manifest),'seed':d['validation_seed'],'arms':arms,'frozen_at':now()}
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        if store.db.execute('SELECT holdout_id FROM joint_decisions WHERE id=?',(did,)).fetchone()!=(None,):raise ValueError('Holdout already bound')
        hid=store.put_blob(holdout)
        for r in arms:store.db.execute('INSERT INTO joint_bindings VALUES (?,?,?,?)',(r['contract_id'],did,r['arm'],r['run_path']))
        store.db.execute('UPDATE joint_decisions SET holdout_id=? WHERE id=?',(hid,did))
    evaluation.write_new(out/'holdout.json',{'holdout_id':hid,**holdout})
    return {'status':'joint_holdout_frozen','holdout_id':hid,**holdout}

def check_binding(store,plan,out):
    w=plan['workload'];did=w['joint_decision_id'];arm=w['joint_arm'];cid=plan['contract_id'];path=str(Path(out).resolve())
    row=store.db.execute('SELECT decision_id,arm,run_path FROM joint_bindings WHERE contract_id=?',(cid,)).fetchone()
    if row!=(did,arm,path):raise ValueError('Unbound joint holdout or run path')
    hid=store.db.execute('SELECT holdout_id FROM joint_decisions WHERE id=?',(did,)).fetchone()[0]
    h=store.get(hid)
    if h['decision_id']!=did or not any(x['contract_id']==cid and x['arm']==arm and x['run_path']==path for x in h['arms']):raise ValueError('Joint binding integrity mismatch')
    return h

def authorize(db,plan,out):
    s=evidence.Store(db)
    try:check_binding(s,plan,out)
    finally:s.close()

def validation_metrics(candidate,baseline,calibration,p):
    why=screen(candidate,p);c=candidate['metrics'];b=baseline['metrics']
    if c['accuracy']+1e-12<b['accuracy']-p['accuracy_tolerance']:why.append('holdout_reference_accuracy_drop')
    if c['accuracy']+1e-12<calibration['metrics']['accuracy']-p['max_validation_drop']:why.append('holdout_calibration_accuracy_drop')
    if baseline['memory'] is None:why.append('missing_reference_memory')
    else:
        key=OBJECTIVES[p['objective']]
        cm=candidate['memory'][key] if p['objective']=='memory' and candidate['memory'] else c.get(key)
        bm=baseline['memory'][key] if p['objective']=='memory' else b[key]
        if cm is None or cm>=bm:why.append('objective_not_improved')
    return why

def validate(store,did,eids):
    init(store);d=decision(store,did);p=d['policy'];fids=fidelity(store,p)
    hid,vid=store.db.execute('SELECT holdout_id,validation_id FROM joint_decisions WHERE id=?',(did,)).fetchone()
    if hid is None:raise ValueError('No frozen holdout')
    h=store.get(hid)
    if len(eids)!=2 or len(set(eids))!=2:raise ValueError('Require exactly two holdout evaluations')
    results={};grades={}
    for eid in eids:
        s,rs,c,err=load(store,eid,p,fids)
        if err or len(rs)!=1:raise ValueError('Require one frozen fixed profile per holdout arm')
        r=rs[0];plan=s['contract'];w=plan['workload'];arm=w.get('joint_arm')
        match=next((x for x in h['arms'] if x['arm']==arm and x['contract_id']==plan['contract_id']),None)
        if not match or arm in results or w.get('joint_decision_id')!=did:raise ValueError('Unexpected holdout arm')
        check_binding(store,plan,match['run_path'])
        original=d['result']['selected' if arm=='candidate' else 'baseline']
        if r['candidate_id']!=original['candidate_id'] or r['context_sha256']!=original['context_sha256'] or r['fixed']!=original['fixed'] or r['artifact_sha256']!=original['artifact_sha256']:raise ValueError('Frozen configuration changed')
        if digest(s['manifest'])!=h['manifest_sha256'] or s['manifest']['seed']!=h['seed']:raise ValueError('Holdout manifest mismatch')
        if any(x['receipt']['started_at']<=h['frozen_at'] for x in s['sessions']):raise ValueError('Holdout predates binding')
        if r['memory'] is not None and original['memory'] is not None and r['memory']['mapped_libraries']!=original['memory']['mapped_libraries']:raise ValueError('Holdout mapped libraries changed')
        report=store.get(eid)['report']
        g={x['id']:x['grade']['joint_success'] for x in report['rows']}
        if len(g)!=r['metrics']['n'] or len(g)!=len(report['rows']):raise ValueError('Invalid paired grade rows')
        results[arm]=r;grades[arm]=g
    if set(results)!={'baseline','candidate'} or set(grades['baseline'])!=set(grades['candidate']):raise ValueError('Incomplete paired holdout')
    b=grades['baseline'];c=grades['candidate'];wins=sum(c[k] and not b[k] for k in c);losses=sum(b[k] and not c[k] for k in c)
    why=validation_metrics(results['candidate'],results['baseline'],d['result']['selected'],p)
    result={'schema':'ramgpt-joint-validation-v1','decision_id':did,'holdout_id':hid,'status':'heldout_pilot_passed' if not why else 'heldout_failed','failures':why,'baseline':results['baseline'],'candidate':results['candidate'],'paired':{'n':len(c),'wins':wins,'losses':losses,'ties':len(c)-wins-losses},'deployment_authorized':False,'scope':'held-out instances of the same task families; sampled memory only; pilot'}
    with store.db:
        store.db.execute('BEGIN IMMEDIATE')
        prior=store.db.execute('SELECT validation_id FROM joint_decisions WHERE id=?',(did,)).fetchone()[0]
        if prior is not None:
            if store.get(prior)!=result:raise ValueError('Decision already validated')
            vid=prior
        else:
            vid=store.put_blob(result);store.db.execute('UPDATE joint_decisions SET validation_id=? WHERE id=?',(vid,did))
    return {'validation_id':vid,**result}

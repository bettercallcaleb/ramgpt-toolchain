import decimal
import re
from registry import digest
import evidence
import evaluation
import optimize
import joint
import weight
import passport
import runtime

SCHEMA='ramgpt-optimization-plan-v1'

def bytes_arg(value):
    m=re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)(GB|GiB|MB|MiB|B)',value,re.I)
    if not m:raise ValueError('Specify memory with B, MB, MiB, GB or GiB')
    units={'b':1,'mb':10**6,'mib':2**20,'gb':10**9,'gib':2**30}
    n=decimal.Decimal(m[1])*units[m[2].lower()]
    if n<=0 or n!=n.to_integral_value() or n>2**63-1:raise ValueError('Invalid memory limit')
    return int(n)

def policy(vram,objective,min_accuracy,min_tasks,headroom,context,max_gap,report_id,max_kl):
    if objective not in ('balanced','memory','latency','tokens'):raise ValueError('Unsupported objective')
    weight.number(min_accuracy,0,1);weight.size(min_tasks);weight.number(max_gap,1e-12)
    if min_tasks<8:raise ValueError('Require at least eight tasks')
    if context is not None:weight.size(context)
    cap=bytes_arg(vram);reserve=bytes_arg(headroom)
    if reserve>=cap:raise ValueError('Headroom must be below VRAM limit')
    if (report_id is None)!=(max_kl is None):raise ValueError('Specify fidelity report and KL limit together')
    if report_id is not None:evidence.checked_id(report_id);weight.number(max_kl)
    return {'vram_bytes':cap,'headroom_bytes':reserve,'objective':objective,'min_accuracy':min_accuracy,'min_tasks':min_tasks,'context_tokens':context,'max_sample_gap_s':max_gap,'fidelity_report_id':report_id,'max_kl':max_kl}

def historical_context(source,report):
    c=optimize.context(source,report);a=iter(source['contract']['execution']['runtime_args']);opts={}
    for k in a:
        if k in opts:raise ValueError('Duplicate stored runtime flag')
        if k in runtime.FLAGS:opts[k]=True
        elif k in runtime.VALUES:opts[k]=next(a)
        else:raise ValueError('Unsupported stored runtime flag')
    opts={k:v for k,v in opts.items() if k not in ('-m','--port')}
    c['execution']={k:v for k,v in c['execution'].items() if k!='runtime_args'}
    c['inputs']=[r for r in c['inputs'] if r['role']!='model']
    c['options']={k:v for k,v in opts.items() if k not in joint.TUNABLE}
    c['workload_protocol']=source['contract']['workload']['protocol']
    return c,{k:v for k,v in opts.items() if k in joint.TUNABLE}

def identity(aid,context,tuning,fixed):
    return digest({'artifact_sha256':aid,'runtime':tuning,'fixed':fixed,'context_sha256':context})

def validation_history(store,aid):
    failed={};history=[];holdout_eids=set()
    if passport.table_exists(store,'joint_decisions'):
        for did,hid,vid in store.db.execute('SELECT id,holdout_id,validation_id FROM joint_decisions ORDER BY id'):
            if vid is None:continue
            v=store.get(vid)
            if v['decision_id']!=did or v['holdout_id']!=hid:raise ValueError('Joint validation binding mismatch')
            for arm in ('baseline','candidate'):holdout_eids.add(v[arm]['evaluation_id'])
            c=v['candidate']
            if c['artifact_sha256']!=aid:continue
            history.append({'kind':'joint','decision_id':did,'validation_id':vid,'candidate_id':c['candidate_id'],'status':v['status'],'failures':v['failures']})
            if v['status']=='heldout_failed':failed.setdefault(c['candidate_id'],[]).append(vid)
    if passport.table_exists(store,'optimization_validations'):
        for did,vid,eid in store.db.execute('SELECT decision_id,validation_id,evaluation_id FROM optimization_validations ORDER BY decision_id'):
            v=store.get(vid);d=store.get(did)
            if v['decision_id']!=did or v['validation_evaluation_id']!=eid:raise ValueError('Fixed validation binding mismatch')
            holdout_eids.add(eid)
            s,r=optimize.load_evaluation(store,d['calibration_evaluation_id'])
            if s['artifact_sha256']!=aid:continue
            c,t=historical_context(s,r);cid=identity(aid,digest(c),t,list(optimize.config(v['candidate'])))
            history.append({'kind':'fixed_budget','decision_id':did,'validation_id':vid,'candidate_id':cid,'status':v['status'],'failures':v['failures']})
            if v['status']=='heldout_failed':failed.setdefault(cid,[]).append(vid)
    return failed,history,holdout_eids

def reasons(r,p,failed):
    m=r['metrics'];mem=r['memory'];why=[]
    if m['n']<p['min_tasks']:why.append('insufficient_tasks')
    if m['accuracy']<p['min_accuracy']:why.append('accuracy_floor')
    if m['truncation_rate']>0:why.append('truncation')
    if p['context_tokens'] is not None and int(r['runtime'].get('-c',0))!=p['context_tokens']:why.append('context_mismatch')
    if mem is None:why.append('missing_verified_memory')
    else:
        if mem['sampled_peak_bytes']>p['vram_bytes']-p['headroom_bytes']:why.append('vram_budget')
        if mem['minimum_free_bytes']<p['headroom_bytes']:why.append('observed_headroom')
        if mem['max_sample_gap_s']>p['max_sample_gap_s']:why.append('telemetry_gap')
    if p['max_kl'] is not None:
        if r['fidelity'] is None:why.append('missing_matched_fidelity')
        elif r['fidelity']['mean_kl']>p['max_kl']:why.append('fidelity_limit')
    if r['candidate_id'] in failed:why.append('previous_heldout_failure')
    return why

def select_group(rows,p,failed):
    counts={}
    for r in rows:counts[r['candidate_id']]=counts.get(r['candidate_id'],0)+1
    for r in rows:
        r['rejections']=reasons(r,p,failed)
        if counts[r['candidate_id']]>1:r['rejections'].append('replicate_resolution_required')
        r['failed_validation_ids']=failed.get(r['candidate_id'],[])
    eligible=[r for r in rows if not r['rejections']]
    pf=joint.frontier(eligible,p)
    selected=None
    if pf and p['objective']!='balanced':
        selected=min(pf,key=lambda r:joint.rank_key(r,p))['candidate_id']
    elif len(pf)==1:selected=pf[0]['candidate_id']
    return {'status':'exploratory_frontier' if pf else 'no_eligible_candidate','screening':rows,'pareto':[r['candidate_id'] for r in pf],'exploratory_pick':selected}

def build(reg,store,model,p):
    model=reg.register(model,'model');aid=model['sha256']
    failed,history,holdout_eids=validation_history(store,aid)
    f=None
    if p['fidelity_report_id'] is not None:
        report_id=p['fidelity_report_id'];weight.select_fidelity(store,report_id,p['max_kl'],2**63-1)
        r=next((x for x in store.get(report_id)['records'] if x['artifact_sha256']==aid),None)
        if r:
            if r['artifact_bytes']!=model['bytes']:raise ValueError('Fidelity artifact size mismatch')
            f={'report_id':report_id,'reference_sha256':r['reference_sha256'],'mean_kl':r['metrics']['mean_kl_bf16_to_artifact']}
    groups={};excluded=[]
    for row in store.find(artifact=aid):
        eid=row['evaluation_id'];s,r=optimize.load_evaluation(store,eid);w=s['contract']['workload']
        if r['schema']!=evaluation.SCHEMA:
            excluded.append({'evaluation_id':eid,'reason':'adaptive_profile_not_comparable','status':r.get('status')});continue
        if 'joint_decision_id' in w or eid in holdout_eids:
            excluded.append({'evaluation_id':eid,'reason':'sealed_holdout_not_calibration'});continue
        c,tuning=historical_context(s,r);ctx=digest(c);manifest=digest(s['manifest']);mem=joint.memory(s)
        libs=None if mem is None else digest(mem['mapped_libraries'])
        k=digest({'context':ctx,'manifest':manifest,'mapped_libraries':libs})
        group=groups.setdefault(k,{'cohort_id':k,'context_sha256':ctx,'manifest_sha256':manifest,'seed':s['manifest']['seed'],'suite':s['manifest']['suite'],'hardware':{'name':c['gpu'][1],'driver':c['gpu'][2],'total_mib':c['gpu'][3]},'rows':[]})
        m=None if mem is None else passport.fields(mem,('sampled_peak_bytes','minimum_free_bytes','max_sample_gap_s','measurement_scope'))
        for profile in optimize.profiles(r):
            fixed=list(optimize.config(profile))
            group['rows'].append({'candidate_id':identity(aid,ctx,tuning,fixed),'evaluation_id':eid,'source_id':row['source_id'],'artifact_sha256':aid,'artifact_bytes':model['bytes'],'runtime':tuning,'fixed':fixed,'metrics':profile,'memory':m,'fidelity':f})
    cohorts=[]
    for k,g in sorted(groups.items()):
        rows=g.pop('rows');cohorts.append({**g,**select_group(rows,p,failed)})
    missing=[]
    if not cohorts:missing.append('measure_fixed_profiles_on_target_runtime')
    if not any(x['memory'] for g in cohorts for x in g['screening']):missing.append('collect_frozen_sampled_vram_and_library_evidence')
    if p['max_kl'] is not None and f is None:missing.append('import_matched_fidelity_cohort')
    eligible=sum(len(g['pareto']) for g in cohorts)
    if eligible:missing.append('freeze_explicit_anchor_and_independent_validation_before_deployment')
    elif cohorts:missing.append('review_rejections_before_collecting_new_evidence')
    body={'schema':SCHEMA,'artifact':{'sha256':aid,'bytes':model['bytes']},'policy':p,'status':'exploratory_candidates' if eligible else 'no_eligible_candidate' if cohorts else 'measurement_required','cohorts':cohorts,'excluded_evaluations':excluded,'validation_history':history,'next_steps':missing,'deployment_authorized':False,'new_inference_requests':0,'scope':'Same artifact historical fixed profiles only. No cross-cohort ranking, no sealed holdout reuse, no GPU inference, no automatic validation or deployment. Balanced returns Pareto choices without a weighted score.'}
    passport.no_private(body)
    return {'plan_id':digest(body),**body}

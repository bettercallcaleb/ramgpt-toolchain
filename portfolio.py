from registry import digest
import evidence
import joint
import native_bridge
import native_tools
import optimize
import passport
import planner
import weight
import workspace

SCHEMA='ramgpt-portfolio-v1'

def subject(model):
    return model['sha256'] if model['format']=='gguf' else 'exl3:'+model['artifact_id']

def exl_memory(r):
    t=r.get('memory_trace')
    if t is None:return None
    if t['method']!='NVML device-wide sampled high-water mark; lower bound on instantaneous peak':raise ValueError('Unsupported EXL3 memory measurement')
    samples=t['samples']
    if len(samples)<2:raise ValueError('Incomplete EXL3 memory trace')
    for x in samples:
        for k in ('used_bytes','free_bytes','reserved_bytes','elapsed_s'):weight.number(x[k])
        if x['total_bytes']!=t['total_bytes'] or x['used_bytes']+x['reserved_bytes']+x['free_bytes']!=x['total_bytes']:raise ValueError('Invalid EXL3 memory accounting')
    gaps=[b['elapsed_s']-a['elapsed_s'] for a,b in zip(samples,samples[1:])]
    if min(gaps)<=0 or max(gaps)!=t['max_sample_gap_s'] or max(x['used_bytes'] for x in samples)!=t['sampled_peak_bytes'] or min(x['free_bytes'] for x in samples)!=t['minimum_free_bytes']:raise ValueError('EXL3 memory summary mismatch')
    return {**passport.fields(t,('sampled_peak_bytes','minimum_free_bytes','max_sample_gap_s')),'measurement_scope':'entire EXL3 session; native library identity not frozen'}

def axes(store,model):
    if model['format']=='exl3':return {'producer_reports':[r['report_id'] for r in native_bridge.find(store,'exl3-first-mover') if r['subject'].get('artifact_id')==model['artifact_id'] and 'declared_bits' in r]}
    aid=model['sha256']
    return {'weight_reports':[{'report_id':r['report_id'],'tool':r['producer'],'kind':r['kind'],'reference_sha256':r['reference_sha256']} for r in weight.find(store,artifact=aid)],
            'native_receipts':[{'receipt_id':r['receipt_id'],'tool':r['tool'],'track':r.get('track',r.get('result',{}).get('track'))} for r in native_tools.find(store,aid)]}

def select(store,reg,p,candidates=None,max_model_bytes=None,manifest=None,gpu_name=None):
    if manifest is not None:evidence.checked_id(manifest)
    if max_model_bytes is not None:weight.size(max_model_bytes)
    known={subject(m):m for m in workspace.models(reg,store)['models']}
    if candidates:
        if len(set(candidates))!=len(candidates) or set(candidates)-known.keys():raise ValueError('Unknown or duplicate model identity')
        known={k:known[k] for k in sorted(candidates)}
    groups={};models=[]
    exl=native_bridge.find(store,'exl3-first-mover')
    def add(key,meta,row):
        groups.setdefault(key,{**meta,'cohort_id':key,'rows':[]})['rows'].append(row)
    for sid,m in known.items():
        info={**m,'subject':sid,'evidence_axes':axes(store,m),'observed_profiles':0,'eligible_profiles':0,'next_steps':[]}
        if max_model_bytes is not None and m['bytes']>max_model_bytes:
            info['next_steps']=['artifact_exceeds_size_limit'];models.append(info);continue
        if m['format']=='gguf':
            plan=planner.build_artifact(store,m,p)
            info['validation_history']=plan['validation_history']
            for cohort in plan['cohorts']:
                if manifest is not None and cohort['manifest_sha256']!=manifest:continue
                if gpu_name is not None and cohort['hardware']['name']!=gpu_name:continue
                meta={k:cohort[k] for k in ('manifest_sha256','context_sha256','hardware','suite','seed')}
                key=digest({'format':'gguf','context_cohort':cohort['cohort_id']})
                for r in cohort['screening']:
                    add(key,{**meta,'format':'gguf'},dict(r,subject=sid))
                    info['observed_profiles']+=1
            if not info['observed_profiles']:info['next_steps']=['measure_or_import_matching_fixed_profiles']
            else:info['next_steps']=plan['next_steps']
        else:
            runs=[r for r in exl if r['subject'].get('artifact_id')==m['artifact_id'] and r.get('kind')=='shared-native-evaluation']
            for run in runs:
                if manifest is not None and run['manifest_sha256']!=manifest:continue
                if gpu_name is not None and run['gpu_name']!=gpu_name:continue
                if run['artifact_bytes']!=m['bytes']:raise ValueError('EXL3 artifact size mismatch')
                hardware={'name':run['gpu_name'],'driver':run.get('gpu_driver'),'total_mib':run.get('gpu_total_mib')}
                context={k:run.get(k) for k in ('backend_source_sha256','worker_sha256','evaluator_sha256','adapter_sha256','schedule_sha256','context','sampling','reasoning_policy','native_cache_policy')}
                # EXL3 model-specific template/tokenizer and environment differ; never rank different artifacts as one runtime cohort.
                key=digest({'format':'exl3','subject':sid,'hardware':hardware,'context':context,'manifest':run['manifest_sha256']})
                for metrics in optimize.profiles(run):
                    fixed=list(optimize.config(metrics))
                    row={'subject':sid,'candidate_id':digest({'cohort':key,'fixed':fixed}),'report_id':run['report_id'],'artifact_bytes':m['bytes'],'runtime':{'-c':str(run['context'])},'fixed':fixed,'metrics':metrics,'memory':exl_memory(run),'fidelity':None}
                    row['rejections']=planner.reasons(row,p,{})
                    # No independent EXL3 validation protocol or frozen native library evidence yet.
                    row['rejections'].append('exl3_runtime_provenance_not_frozen')
                    add(key,{'format':'exl3','manifest_sha256':run['manifest_sha256'],'hardware':hardware,'context_sha256':digest(context)},row)
                    info['observed_profiles']+=1
            info['next_steps']=['freeze_native_runtime_provenance'] if info['observed_profiles'] else ['run_exl3_with_matching_dataset_and_memory_telemetry']
        models.append(info)
    by_subject={m['subject']:m for m in models}
    cohorts=[]
    for k,g in sorted(groups.items()):
        rows=g.pop('rows')
        counts={}
        for r in rows:counts[r['candidate_id']]=counts.get(r['candidate_id'],0)+1
        for r in rows:
            if counts[r['candidate_id']]>1 and 'replicate_resolution_required' not in r['rejections']:r['rejections'].append('replicate_resolution_required')
        eligible=[r for r in rows if not r['rejections']]
        frontier=joint.frontier(eligible,p)
        chosen=None
        if frontier and p['objective']!='balanced':chosen=min(frontier,key=lambda r:joint.rank_key(r,p))['candidate_id']
        elif len(frontier)==1:chosen=frontier[0]['candidate_id']
        for r in eligible:by_subject[r['subject']]['eligible_profiles']+=1
        cohorts.append({**g,'screening':rows,'pareto':[r['candidate_id'] for r in frontier],'exploratory_pick':chosen})
    eligible=sum(len(g['pareto']) for g in cohorts)
    body={'schema':SCHEMA,'policy':p,'max_model_bytes':max_model_bytes,'manifest_filter':manifest,'gpu_name_filter':gpu_name,'models':models,'cohorts':cohorts,'status':'exploratory_candidates' if eligible else 'no_eligible_candidate','deployment_authorized':False,'new_inference_requests':0,'scope':'Catalog-wide historical screening. Rank only within compatible cohorts; no ranking across backends or workloads. Native receipts are separate evidence axes, not substituted for runtime data. EXL3 candidates remain blocked pending frozen runtime provenance.'}
    passport.no_private(body)
    return {'plan_id':digest(body),**body}

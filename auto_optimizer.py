import fcntl
from pathlib import Path
import adaptive
import evaluation
import joint
import measurement_plan as mp
import model_passport
import planner
import runtime_spec
import workflow
from registry import digest,fingerprint,read_json

SCHEMA='ramgpt-autooptimizer-v1'

def workload(archive,budgets,kv,seed,validation_seed,variants,reserve,vram,headroom,objective,min_accuracy,max_requests,max_tokens):
    mp.limits(budgets,['allocated'],reserve,kv,seed,variants,max_requests,max_tokens)
    if type(validation_seed) is not int or not 0<=validation_seed<2**31 or seed==validation_seed:raise ValueError('Require distinct calibration and validation seeds')
    if 'f16' not in kv:raise ValueError('Require f16 anchor')
    if objective not in joint.OBJECTIVES:raise ValueError('Choose memory, latency or tokens')
    archive=fingerprint(archive,'evaluator')
    if archive['sha256']!=adaptive.PIN:raise ValueError('Pinned evaluator required')
    p=planner.policy(vram,objective,min_accuracy,8,headroom,None,.25,None,None)
    policy={'objective':objective,'min_accuracy':min_accuracy,'max_latency_s':120,'max_sampled_peak_bytes':p['vram_bytes']-p['headroom_bytes'],'min_free_bytes':p['headroom_bytes'],'max_mean_tokens':max(budgets),'max_truncation':0,'min_tasks':8*variants,'max_validation_drop':.125,'accuracy_tolerance':0,'fidelity_report_id':None,'max_kl':None,'max_artifact_bytes':p['vram_bytes']-p['headroom_bytes'],'max_sample_gap_s':.25}
    joint.policy(policy)
    manifests=[adaptive.worker(archive['path'],{'action':'generate','seed':s,'variants':variants,'suite':'core','ladder_variants':1}) for s in (seed,validation_seed)]
    def prompts(m):return {' '.join(t['prompt'].split()) for t in m['tasks']}
    if prompts(manifests[0])&prompts(manifests[1]):raise ValueError('Calibration and holdout prompts overlap')
    n=len(manifests[0]['tasks'])
    if n!=len(manifests[1]['tasks']) or n!=8*variants:raise ValueError('Unexpected native task count')
    sched=evaluation.worker(archive['path'],{'action':'prepare','manifest':manifests[0],'budgets':budgets,'strategies':['allocated'],'reserve':reserve})['requests']
    cal=mp.cost(sched,len(kv))
    hold=workflow.holdout_cost(n,[max(budgets)],[max(budgets)])
    total={k:cal[k]+hold[k] for k in ('total_completion_requests','total_completion_token_cap')}
    mp.check_cost(total,max_requests,max_tokens)
    return {'budgets':budgets,'kv':kv,'seed':seed,'validation_seed':validation_seed,'variants':variants,'reserve':reserve,'policy':policy,'anchor':[max(budgets),'allocated',reserve],'calibration_manifest':digest(manifests[0]),'validation_manifest':digest(manifests[1]),'calibration_cost':cal,'max_holdout_cost':hold,'max_total_cost':total,'max_requests':max_requests,'max_completion_tokens':max_tokens},manifests

def prepare_request(model,spec_path,budgets,kv,seed,validation_seed,variants,reserve,vram,headroom,objective,min_accuracy,max_requests,max_tokens):
    spec=read_json(spec_path);runtime_spec.check_binding(spec)
    inputs={r['role']:r['path'] for r in spec['inputs'] if r['role']!='runtime_dependency'}
    model_ref=fingerprint(model,'model')
    if fingerprint(inputs['model'],'model')!=model_ref:raise ValueError('Spec model mismatch')
    w,manifests=workload(inputs['evaluator'],budgets,kv,seed,validation_seed,variants,reserve,vram,headroom,objective,min_accuracy,max_requests,max_tokens)
    return {'schema':SCHEMA,'model':model_ref,'spec':fingerprint(spec_path,'configuration'),**w},manifests

def check_prior(store,manifest):
    prompts={' '.join(t['prompt'].split()) for t in manifest['tasks']}
    for row in store.find():
        old=store.get(row['source_id'])['manifest']
        if prompts&{' '.join(t['prompt'].split()) for t in old['tasks']}:raise ValueError('Validation tasks overlap existing evidence')

def run(reg,store,model,spec,out,budgets,kv,seed,validation_seed,variants,reserve,vram,headroom,objective,min_accuracy,max_requests,max_tokens,execute=False):
    req,manifests=prepare_request(model,spec,budgets,kv,seed,validation_seed,variants,reserve,vram,headroom,objective,min_accuracy,max_requests,max_tokens)
    root=Path(out).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.auto.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'auto-plan.json'
        if path.exists():
            record=read_json(path)
            if record['request']!=req or digest(record['request'])!=record['request_id']:raise ValueError('AutoOptimizer parameters changed; create a new plan')
            if store.get(record['record_id'])!={k:v for k,v in record.items() if k!='record_id'}:raise ValueError('AutoOptimizer plan binding mismatch')
        else:
            if any(x.name!='.auto.lock' for x in root.iterdir()):raise ValueError('Require fresh output directory')
            check_prior(store,manifests[1])
            cal=req['calibration_cost']
            p=mp.prepare(reg,store,spec,root/'calibration',seed,variants,budgets,['allocated'],reserve,kv,cal['total_completion_requests'],cal['total_completion_token_cap'])
            if p['cost']!=cal or p['manifest_sha256']!=req['calibration_manifest']:raise ValueError('Calibration plan mismatch')
            record={'request':req,'request_id':digest(req),'plan_id':p['plan_id']}
            with store.db:rid=store.put_blob(record)
            record={**record,'record_id':rid}
            evaluation.write_new(path,record)
        pid=record['plan_id']
        plan=mp.load(reg,store,pid)
        if plan['artifact_sha256']!=req['model']['sha256'] or plan['manifest_sha256']!=req['calibration_manifest'] or plan['cost']!=req['calibration_cost'] or Path(plan['root'])!=root/'calibration':raise ValueError('AutoOptimizer calibration binding mismatch')
        if not execute:return {'status':'auto_plan_frozen','plan_id':pid,'max_total_cost':req['max_total_cost'],'out':str(root),'new_inference_requests':0}
        cal=mp.run(reg,store,pid,True)
        h=req['max_holdout_cost']
        decision=workflow.select(reg,store,pid,req['policy'],'f16',req['anchor'],validation_seed,h['total_completion_requests'],h['total_completion_token_cap'])
        if decision['holdout_id']:
            if digest(read_json(root/'calibration/holdout/tasks.json'))!=req['validation_manifest']:raise ValueError('Frozen holdout differs from planned manifest')
        final=workflow.finish(reg,store,pid,True)
        packet=model_passport.build(reg,store,req['model']['sha256'])
        model_passport.verify(packet)
        portable=root/'model.passport.json'
        if not portable.exists():evaluation.write_new(portable,packet)
        elif read_json(portable)!=packet:raise ValueError('Existing passport snapshot differs')
        selected=None
        if final['status']=='heldout_pilot_passed':
            v=store.get(final['validation_id']);c=v['candidate']
            if c['artifact_sha256']!=req['model']['sha256']:raise ValueError('Unexpected candidate model')
            contract=reg.contract(c['contract_id']);reg.verify(c['contract_id'])
            selected={'runtime':next(r['path'] for r in contract['inputs'] if r['role']=='runtime'),'runtime_args':contract['execution']['runtime_args'],'gpu_uuids':contract['execution']['gpu_uuids'],'generation_budget':c['fixed'][0],'strategy':c['fixed'][1],'reserve':c['fixed'][2],'validation_id':final['validation_id'],'scope':'Pilot-validated local settings, not deployment approval'}
            profile=root/'selected-profile.json'
            if not profile.exists():evaluation.write_new(profile,selected)
            elif read_json(profile)!=selected:raise ValueError('Existing selected profile differs')
        result={'status':final['status'],'plan_id':pid,'decision_id':final['decision_id'],'validation_id':final['validation_id'],'reason':final['reason'],'failures':final['failures'],'passport_id':packet['passport_id'],'selected_profile':str(root/'selected-profile.json') if selected else None,'max_total_cost':req['max_total_cost'],'deployment_authorized':False}
        result_path=root/'auto-result.json'
        if result_path.exists():
            if read_json(result_path)!=result:raise ValueError('Existing AutoOptimizer result differs')
        else:evaluation.write_new(result_path,result)
        return {**result,'new_inference_requests':cal['new_inference_requests']+final['new_inference_requests'],'new_smoke_requests':cal['new_smoke_requests']+final['new_smoke_requests']}

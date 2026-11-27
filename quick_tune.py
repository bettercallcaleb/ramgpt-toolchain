import fcntl
from pathlib import Path
import auto_optimizer as auto
import evaluation
import install_native
import measurement_plan
import runtime
import runtime_spec
from registry import digest,fingerprint,read_json

def choose_gpu(explicit):
    states=runtime.gpu_state()
    if explicit:
        if explicit not in states:raise ValueError('Unknown GPU UUID')
        return explicit
    idle=[u for u,s in states.items() if s['used_mib']<=512 and not runtime.compute_apps(u)]
    if len(idle)!=1:raise ValueError('Specify --gpu-uuid (expected exactly one idle GPU)')
    return idle[0]

def run(reg,store,model,out,config,gpu=None,context=4096,port=18429,threads=8,ngl=999,budgets=(256,1024),kv=('f16','q8_0'),seed=941,validation_seed=942,variants=1,reserve=64,vram='24GiB',headroom='1GiB',objective='memory',min_accuracy=.85,max_requests=52,max_tokens=37120,execute=False):
    binary=config.get('llama-autotune',{}).get('runtime')
    archive=config.get('budgetfit',{}).get('evaluator')
    if not binary or not archive:raise ValueError('Configure llama-autotune.runtime and budgetfit.evaluator using setup or supply')
    budgets=list(budgets);kv=list(kv)
    w,manifests=auto.workload(archive,budgets,kv,seed,validation_seed,variants,reserve,vram,headroom,objective,min_accuracy,max_requests,max_tokens)
    root=Path(out).expanduser().resolve()
    ref=fingerprint(model,'model')
    binary_ref=install_native.runtime_check(binary)
    archive_ref=fingerprint(archive,'evaluator')
    lock_path=root/'quick-plan.json'
    prior=read_json(lock_path) if lock_path.is_file() else None
    if prior:
        selected=prior['request']['gpu_uuid']
        if gpu and gpu!=selected:raise ValueError('GPU differs from frozen quick plan')
    else:selected=choose_gpu(gpu)
    ex=runtime_spec.execution(ref['path'],selected,port,context,threads,ngl)
    request={'schema':'ramgpt-quick-tune-v1','model':ref,'runtime':binary_ref,'evaluator':archive_ref,'gpu_uuid':selected,'runtime_settings':ex,'workload':w,'root':str(root)}
    packet={'request_id':digest(request),'request':request}
    if prior and prior!=packet:raise ValueError('Quick tune parameters changed; use a fresh output directory')
    if not prior:
        measurement_plan.used_prompts(store,manifests[0],ref['sha256'])
        auto.check_prior(store,manifests[1])
    if not execute:return {'status':'quick_tune_preview','request_id':packet['request_id'],'gpu_uuid':selected,'max_total_cost':w['max_total_cost'],'out':str(root),'new_inference_requests':0,'gpu_model_loads':0}
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.quick.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if lock_path.exists():
            if read_json(lock_path)!=packet:raise ValueError('Quick plan changed')
            if store.get(packet['request_id'])!=request:raise ValueError('Quick plan database binding mismatch')
        else:
            if any(p.name!='.quick.lock' for p in root.iterdir()):raise ValueError('Require fresh quick tune directory')
            measurement_plan.used_prompts(store,manifests[0],ref['sha256'])
            auto.check_prior(store,manifests[1])
            with store.db:store.put_blob(request)
            evaluation.write_new(lock_path,packet)
        folder=root/'runtime';spec=folder/'spec.json';loads=0
        if not folder.exists():
            runtime_spec.capture(ref['path'],binary_ref['path'],archive_ref['path'],selected,folder,port,context,threads,ngl,True)
            loads=1
        if not spec.is_file():raise ValueError('Incomplete runtime capture; inspect manually, no automatic retry')
        captured=read_json(spec)
        runtime_spec.check_binding(captured)
        if {k:v for k,v in captured['execution'].items() if k!='template_capture'}!=ex:raise ValueError('Captured runtime settings differ')
        refs={r['role']:fingerprint(r['path'],r['role']) for r in captured['inputs']}
        if any(refs[k]!=v for k,v in [('model',ref),('runtime',binary_ref),('evaluator',archive_ref)]):raise ValueError('Captured runtime input differs')
        result=auto.run(reg,store,ref['path'],spec,root/'search',budgets,kv,seed,validation_seed,variants,reserve,vram,headroom,objective,min_accuracy,max_requests,max_tokens,True)
        return {**result,'quick_request_id':packet['request_id'],'template_capture_loads':loads,'runtime_spec':str(spec)}

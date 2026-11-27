import json
import math
import os
import subprocess
from pathlib import Path
from registry import digest,fingerprint,read_json
import adaptive
import evaluation
import evidence
import inference
import native_bridge as nb
import native_tools as nt
import runtime
import telemetry

def schedule(archive,manifest,budgets,strategies,reserve):
    if fingerprint(archive,'evaluator')['sha256']!=adaptive.PIN:raise ValueError('Pinned BudgetFit archive required')
    return evaluation.worker(archive,{'action':'prepare','manifest':manifest,'budgets':budgets,'strategies':strategies,'reserve':reserve})['requests']

def score_rows(archive,manifest,sched,observed,budgets,strategies,reserve):
    if len(observed)!=len(sched):raise ValueError('Incomplete native responses')
    rows=[]
    for i,(q,r) in enumerate(zip(sched,observed)):
        if r['request_id']!=q['request_id']:raise ValueError('Native response order mismatch')
        if type(r['elapsed_s']) not in (float,int) or not math.isfinite(r['elapsed_s']) or r['elapsed_s']<0:raise ValueError('Invalid native elapsed time')
        req={**q,'body':{'model':'exl3','messages':[{'role':'user','content':q['prompt']}]}}
        inference.check_response(r['raw'],req,{'model':'exl3'})
        rows.append({'request':req,'raw':r['raw'],'ordinal':i,'elapsed_s':r['elapsed_s']})
    return evaluation.worker(archive,{'action':'score','manifest':manifest,'rows':rows,'budgets':budgets,'strategies':strategies,'reserve':reserve})

def model_identity(store,folder,native):
    files=nt.worker(nb.EXL3_MANIFEST,[native,folder],300)
    aid=digest({'format':'exl3','files':files})
    reports=[r for r in nb.find(store,'exl3-first-mover') if r['subject'].get('artifact_id')==aid and r.get('files')==files]
    if not reports:raise ValueError('Import the matching EXL3 manifest before running')
    if any(p.is_symlink() for p in Path(folder).rglob('*')):raise ValueError('Artifact symlinks unsupported')
    return aid,files

def run(store,folder,dataset,archive,native,backend,python,gpu_uuid,out,budgets,strategies,reserve,context,max_requests,max_tokens,execute=False):
    if execute is not True:raise ValueError('Explicit --execute required')
    if not runtime.UUID.fullmatch(gpu_uuid):raise ValueError('Select one GPU UUID')
    if type(context) is not int or not 256<=context<=131072 or type(max_requests) is not int or not 1<=max_requests<=10000 or type(max_tokens) is not int or not 1<=max_tokens<=10**8:raise ValueError('Invalid context or limits')
    folder=Path(folder).resolve(strict=True);dataset=Path(dataset).resolve(strict=True);archive=Path(archive).resolve(strict=True);native=Path(native).resolve(strict=True);backend=Path(backend).resolve(strict=True);out=Path(out).resolve()
    python=os.path.abspath(os.path.expanduser(python))
    if not Path(python).is_file() or not os.access(python,os.X_OK):raise ValueError('Executable backend Python required')
    if out.exists() or out.is_relative_to(folder) or out.is_relative_to(backend):raise ValueError('Require fresh output outside inputs')
    if native.name!='fm.py' or (native.parent/'config.json').exists():raise ValueError('Expected clean native First Mover script')
    manifest=read_json(dataset)
    sched=schedule(archive,manifest,budgets,strategies,reserve)
    cost=sum(q['max_tokens'] for q in sched)
    if len(sched)>max_requests or cost>max_tokens:raise ValueError('Explicit request or completion cap exceeded')
    if read_json(folder/'config.json').get('model_type')!='qwen3':raise ValueError('Initial EXL3 text adapter supports Qwen3 only')
    aid,files=model_identity(store,folder,native)
    source=nt.tree(backend/'exllamav3');native_ref=nt.ref(native);archive_ref=nt.ref(archive);dataset_ref=nt.ref(dataset)
    script=Path(__file__).with_name('exl3_worker.py');worker_ref=nt.ref(script)
    out.mkdir(parents=True,exist_ok=False)
    evaluation.write_new(out/'schedule.json',sched)
    evaluation.write_new(out/'plan.json',{'artifact_id':aid,'schedule_sha256':digest(sched),'requests':len(sched),'completion_cap':cost,'context':context,'backend_source_sha256':source,'worker_sha256':worker_ref['sha256'],'evaluator_sha256':archive_ref['sha256']})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu_uuid,CUDA_DEVICE_ORDER='PCI_BUS_ID',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with runtime.lease(gpu_uuid):
        state=runtime.gpu_state().get(gpu_uuid)
        if not state or runtime.compute_apps(gpu_uuid) or state['used_mib']>512:raise ValueError('Selected GPU occupied')
        cmd=[python,'-I',str(script),'--backend',str(backend),'--model',str(folder),'--context',str(context),'--schedule',str(out/'schedule.json'),'--out',str(out/'responses.jsonl')]
        with telemetry.MemoryTrace(gpu_uuid) as trace, (out/'runtime.log').open('x') as log:
            proc=subprocess.Popen(cmd,env=env,cwd=backend,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                if proc.wait()!=0:raise ValueError('Native EXL3 execution failed; see runtime.log')
            finally:
                if proc.poll() is None:runtime.stop_owned(proc)
        if runtime.compute_apps(gpu_uuid) or runtime.gpu_state()[gpu_uuid]['used_mib']>state['used_mib']+128:raise ValueError('GPU did not return to baseline')
    memory_trace=trace.report()
    evaluation.write_new(out/'memory.json',memory_trace)
    observed=[json.loads(x) for x in (out/'responses.jsonl').read_text().splitlines()]
    if read_json(out/'schedule.json')!=sched or model_identity(store,folder,native)!=(aid,files) or source!=nt.tree(backend/'exllamav3') or native_ref!=nt.ref(native) or archive_ref!=nt.ref(archive) or dataset_ref!=nt.ref(dataset) or worker_ref!=nt.ref(script):raise ValueError('Run input changed')
    scored=score_rows(archive,manifest,sched,observed,budgets,strategies,reserve)
    body={'kind':'shared-native-evaluation','manifest_sha256':digest(manifest),'schedule_sha256':digest(sched),'evaluator_sha256':archive_ref['sha256'],'adapter_sha256':evidence.adapter_hash(),'backend_source_sha256':source,'worker_sha256':worker_ref['sha256'],'artifact_bytes':sum(v['size'] for v in files.values()),'context':context,'sampling':'greedy','reasoning_policy':'Qwen3 local chat template enable_thinking=False; thinking output rejected','native_cache_policy':'native default; prefix reuse possible','gpu_name':state['name'],'gpu_driver':state['driver'],'gpu_total_mib':state['total_mib'],'memory_trace':memory_trace,'requests':len(sched),'completion_cap':cost,'profiles':scored['profiles'],'rows':scored['rows'],'native_responses':observed,'scope':'Same native task scorer; tokenizer, template, cache and runtime can differ from GGUF. No fidelity or deployment claim.'}
    report=nb.save(store,'exl3-first-mover',{'format':'exl3','artifact_id':aid},body)
    evaluation.write_new(out/'evaluation.json',report)
    return {'report_id':report['report_id'],'artifact_id':aid,'requests':len(sched),'profiles':[x['budgetfit'] for x in scored['profiles']]}

def compare(store,gguf_eid,exl3_report_id):
    gg=evidence.checked_id(gguf_eid);eid=evidence.checked_id(exl3_report_id)
    hit=[r for r in store.find() if r['evaluation_id']==gg]
    if len(hit)!=1:raise ValueError('Expected one managed GGUF evaluation')
    g=store.get(gg)['report'];src=store.get(hit[0]['source_id'])
    x=store.get(eid)
    if not any(r['report_id']==eid for r in nb.find(store,'exl3-first-mover')) or x.get('kind')!='shared-native-evaluation':raise ValueError('Expected registered EXL3 shared evaluation')
    if g['schema']!=evaluation.SCHEMA:raise ValueError('Fixed native GGUF profiles required')
    sched=[{k:r[k] for k in ('request_id','task_id','strategy','prompt','max_tokens')} for r in src['schedule']]
    if (digest(src['manifest']),digest(sched),g['evaluator_sha256'],g['adapter_sha256'])!=(x['manifest_sha256'],x['schedule_sha256'],x['evaluator_sha256'],x['adapter_sha256']):raise ValueError('Different native task, prompt schedule or scorer')
    def key(p):return tuple(p['budgetfit'][k] for k in ('budget','strategy','reserve'))
    a={key(p):p['budgetfit'] for p in g['profiles']};b={key(p):p['budgetfit'] for p in x['profiles']}
    if len(a)!=len(g['profiles']) or len(b)!=len(x['profiles']) or a.keys()!=b.keys():raise ValueError('Different fixed profile matrix')
    return {'gguf':{'artifact_sha256':src['artifact_sha256'],'evaluation_id':gg},'exl3':{'artifact_id':x['subject']['artifact_id'],'report_id':eid},'profiles':[{'budget':k[0],'strategy':k[1],'reserve':k[2],'gguf':a[k],'exl3':b[k]} for k in sorted(a)],'scope':'Matched tasks, prompt text and native scoring only. Runtime, chat template, tokenizer, token counts and cache policy may differ; no causal or deployment ranking.'}

import re
from pathlib import Path
from registry import digest,fingerprint
import evidence
import evaluation
import optimize
import joint
import weight
import runtime
import telemetry

SCHEMA='ramgpt-model-passport-v1'
ENVELOPE='ramgpt-passport-envelope-v1'
LIMITATIONS=[
    'This is a historical evidence summary, not a signature, certificate or deployment authorization.',
    'Hash verification alone detects content changes against a known digest; it does not authenticate the producer.',
    'Independent metric axes are not pooled across task manifests, references or runtime settings.',
    'Sampled device memory is a lower bound on instantaneous peak; startup is not necessarily cold-disk load.',
    'Historical QuantBench replay does not regenerate full logits or reverify the model binaries.',
    'A missing metric is unknown, never zero; pilot holdouts use instances of the same task families.',
    'Runtime arguments are observed settings, not a portable executable command or deployment recommendation.'
]

def fields(d,keys):
    return {k:d[k] for k in keys if k in d}

def no_private(value):
    if isinstance(value,dict):
        for k,v in value.items():
            no_private(k);no_private(v)
    elif isinstance(value,list):
        for v in value:no_private(v)
    elif isinstance(value,str):
        if re.search(r'(?:^|[\s"\x27=])(?:/|~[/\\]|[A-Za-z]:[\\/]|\\\\)|GPU-[0-9a-fA-F-]{36}|file://',value):
            raise ValueError('Local path or device identity in portable export')

def table_exists(store,name):
    return store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",(name,)).fetchone() is not None

def opts(plan):
    a=iter(plan['execution']['runtime_args']);out={}
    for key in a:
        if key in out:raise ValueError('Duplicate runtime flag')
        if key in runtime.FLAGS:out[key]=True
        elif key in runtime.VALUES:out[key]=next(a)
        else:raise ValueError('Unknown historical runtime flag')
    return {k:v for k,v in out.items() if k not in ('-m','--port','--host','--alias')}

def profile(p):
    result={'metrics':p['budgetfit'],'mode':p['mode'],'prompt_contract':p['prompt_contract'],'diagnostics':[]}
    for t in p['tracks']:
        metrics=('n','attempted','complete','confirmed_correct','confirmed_wrong','unknown','joint_success','format_compliance','strict_accuracy','truncation_rate','repetition_flag_rate','accuracy_upper_bound')
        result['diagnostics'].append({'track':t['track'],'measurement':fields(t['measurement'],metrics),'breakdown':{axis:{name:fields(m,metrics) for name,m in groups.items()} for axis,groups in t['breakdown'].items()}})
    return result

def managed(store,row):
    s,r=optimize.load_evaluation(store,row['evaluation_id'])
    p=s['contract'];w=p['workload'];m=s['manifest']
    refs=[fields(x,('role','sha256','bytes')) for x in p['inputs']]
    refs.sort(key=lambda x:(x['role'],x['sha256']))
    gpu=[]
    for ses in s['sessions']:
        g=fields(ses['receipt']['gpu_before'],('name','total_mib','driver'))
        if g not in gpu:gpu.append(g)
    mem=joint.memory(s)
    if mem:
        mem={**fields(mem,('sampled_peak_bytes','minimum_free_bytes','max_sample_gap_s','measurement_scope')),'mapped_library_sha256':sorted(x['sha256'] for x in mem['mapped_libraries'])}
    rows=evidence.validate_source(s)
    perf=None
    try:perf=telemetry.summarize(rows)
    except (KeyError,ValueError,ZeroDivisionError):pass
    body={**row,'evaluation_schema':r['schema'],'task_manifest':{**fields(m,('version','seed','suite','variants','ladder_variants')),'sha256':digest(m),'tasks':len(m['tasks'])},
          'inputs':refs,'runtime':{'engine':'llama-server','flags':opts(p),'sampling':p['execution']['sampling'],'reasoning_policy':p['execution']['reasoning_policy'],'hardware':gpu,'backend_fingerprint':s['response_identity']['system_fingerprint'],'template_binding':p['execution'].get('template_binding')},
          'workload':fields(w,('protocol','evaluator','budgets','strategies','reserve','joint_decision_id','joint_arm','router_decision_id')),
          'sessions':len(s['sessions']),'memory':mem,'performance':perf,'startup_seconds':[ses['receipt']['startup_s'] for ses in s['sessions']],
          'status':r.get('status','observed'),'profiles':[],'router':None}
    if r['schema']==evaluation.SCHEMA:
        optimize.profiles(r);body['profiles']=[profile(x) for x in r['profiles']]
    elif r['schema']=='ramgpt-adaptive-router-v1':
        d=store.get(r['decision_id'])
        if digest(d)!=w['router_decision_id'] or d!=w['router_decision']:raise ValueError('Router binding mismatch')
        body['router']={'decision_id':r['decision_id'],'policy':d['selection']['policy'],'thresholds':d['thresholds'],'validation':r['validation'],'passed':r['passed']}
    else:raise ValueError('Unsupported managed evaluation schema')
    return body

def weight_summary(store,row):
    r=store.get(row['report_id'])
    result={**row,'limitations':r['limitations']}
    if row['kind']=='fidelity':
        d=r['source']['result']
        result['source_manifest_sha256']=r['source']['release_manifest_sha256']
        result['reference_revision']=r['source']['run_spec']['resolved_reference_revision']
        ranking=next(x for x in d['rankings'] if x['artifact_sha256']==row['artifact_sha256'])
        result['artifact_label']=ranking['artifact']
        result['quantization_label']=d['quantization_label']
        result['paired_uncertainty']=[x for x in d['paired_uncertainty']['winner_vs_all'] if ranking['artifact'] in (x['winner'],x['other'])]
        result['native_script_sha256']=r['source']['native_script_sha256']
    elif row['kind']=='knowledge_depth':
        result['source_sha256']=r['source']['export']['evidence_sha256']
        result['native_script_sha256']=r['source']['native_script_sha256']
    elif row['kind']=='quantization_recipe':
        result['source_sha256']=fields(r['source'],('manifest_sha256','recipe_sha256','tensor_types_sha256'))
        result['metrics']={k:v for k,v in row['metrics'].items() if k!='imatrix'}
        result['metrics']['imatrix_declared']=row['metrics']['imatrix'] is not None
    else:raise ValueError('Unsupported weight evidence kind')
    return result

def decisions(store,aid,eids):
    result=[]
    if table_exists(store,'optimization_decisions'):
        for did,eid in store.db.execute('SELECT id,calibration_evaluation_id FROM optimization_decisions ORDER BY id'):
            if eid not in eids:continue
            d=store.get(did);r=d['result']
            if d['calibration_evaluation_id']!=eid:raise ValueError('Fixed decision index mismatch')
            entry={'kind':'fixed_budget','decision_id':did,'calibration_evaluation_id':eid,'status':r['status'],'policy':d['policy'],'selected':r['candidate'],'screening':r['screening'],'validation':None}
            if table_exists(store,'optimization_validations'):
                v=store.db.execute('SELECT validation_id,evaluation_id FROM optimization_validations WHERE decision_id=?',(did,)).fetchone()
                if v:
                    b=store.get(v[0])
                    if b['decision_id']!=did or b['validation_evaluation_id']!=v[1]:raise ValueError('Fixed validation index mismatch')
                    entry['validation']={'validation_id':v[0],**fields(b,('status','failures','validation_evaluation_id','candidate','reference','paired','objective_savings'))}
            result.append(entry)
    if table_exists(store,'joint_decisions'):
        for did,hid,vid in store.db.execute('SELECT id,holdout_id,validation_id FROM joint_decisions ORDER BY id'):
            d=store.get(did);r=d['result']
            related=[x for x in r['screening'] if x['artifact_sha256']==aid]
            if not related and not any(x['artifact_sha256']==aid for x in r['unmeasured_fidelity_candidates']):continue
            if d['schema']!=joint.SCHEMA:raise ValueError('Unknown joint decision schema')
            v=store.get(vid) if vid else None
            roles=[k for k in ('baseline','candidate') if v and v[k]['artifact_sha256']==aid]
            entry={'kind':'joint','decision_id':did,'validated_artifact_roles':roles,'status':r['status'],'policy':d['policy'],'holdout_id':hid,'validation':None,
                   'selected':fields(r['selected'],('candidate_id','evaluation_id','artifact_sha256','runtime','fixed')) if r['selected'] else None,
                   'screening':[fields(x,('candidate_id','evaluation_id','rejections')) for x in related],
                   'unmeasured_fidelity_candidate':any(x['artifact_sha256']==aid for x in r['unmeasured_fidelity_candidates'])}
            if vid and roles:
                v=store.get(vid)
                if v['decision_id']!=did or v['holdout_id']!=hid:raise ValueError('Joint validation index mismatch')
                entry['validation']={'validation_id':vid,**fields(v,('status','failures','paired','deployment_authorized')),'arms':{k:fields(v[k],('evaluation_id','artifact_sha256','candidate_id','runtime','fixed','metrics')) for k in ('baseline','candidate')}}
            result.append(entry)
    return result

def build(reg,store,aid):
    evidence.checked_id(aid)
    views=weight.find(store,artifact=aid);rows=store.find(artifact=aid)
    weights=[weight_summary(store,x) for x in views]
    runs=[managed(store,x) for x in rows]
    sizes={x['artifact_bytes'] for x in views}
    sizes.update(next(x['bytes'] for x in r['inputs'] if x['role']=='model') for r in runs)
    local=reg.db.execute('SELECT bytes FROM artifacts WHERE sha256=?',(aid,)).fetchone()
    if local:sizes.add(local[0])
    if len(sizes)!=1:raise ValueError('Unknown artifact or conflicting byte size')
    selection=decisions(store,aid,{x['evaluation_id'] for x in rows})
    missing=[]
    for k in ('fidelity','knowledge_depth','quantization_recipe'):
        if not any(x['kind']==k for x in weights):missing.append(k)
    if not runs:missing.append('managed_task_accuracy')
    if not any(x['memory'] is not None for x in runs):missing.append('sampled_vram')
    if not any(x['performance'] is not None for x in runs):missing.append('native_prefill_decode')
    if not any(x['status'].startswith('heldout_') for x in runs) and not any(x['validation'] and (x['kind']=='fixed_budget' or any(a['artifact_sha256']==aid for a in x['validation']['arms'].values())) for x in selection):missing.append('independent_pilot')
    body={'schema':SCHEMA,'artifact':{'sha256':aid,'bytes':sizes.pop(),'registry_recorded':bool(local)},'weight_evidence':weights,'managed_evaluations':runs,'optimization_decisions':selection,'missing_axes':missing,'deployment':{'authorized':False,'recommended_configuration':None,'status':'not_certified'},'limitations':LIMITATIONS}
    no_private(body)
    return {'schema':ENVELOPE,'passport_id':digest(body),'passport':body}

def verify(document,expected_id=None):
    if not isinstance(document,dict) or set(document)!={'schema','passport_id','passport'} or document['schema']!=ENVELOPE:raise ValueError('Unsupported passport envelope')
    body=document['passport']
    if not isinstance(body,dict) or set(body)!={'schema','artifact','weight_evidence','managed_evaluations','optimization_decisions','missing_axes','deployment','limitations'} or body['limitations']!=LIMITATIONS:raise ValueError('Invalid passport fields')
    if body['schema']!=SCHEMA or digest(body)!=evidence.checked_id(document['passport_id']):raise ValueError('Passport digest mismatch')
    if expected_id is not None and evidence.checked_id(expected_id)!=document['passport_id']:raise ValueError('Unexpected passport identity')
    aid=evidence.checked_id(body['artifact']['sha256']);size=weight.size(body['artifact']['bytes'])
    for key in ('weight_evidence','managed_evaluations','optimization_decisions','missing_axes','limitations'):
        if not isinstance(body[key],list):raise ValueError('Invalid passport collection')
    for r in body['weight_evidence']:
        if r['artifact_sha256']!=aid or r['artifact_bytes']!=size:raise ValueError('Weight identity mismatch')
        evidence.checked_id(r['report_id']);evidence.checked_id(r['reference_sha256'])
        if r['kind']=='fidelity':weight.number(r['metrics']['mean_kl_bf16_to_artifact'])
    for r in body['managed_evaluations']:
        if r['artifact_sha256']!=aid:raise ValueError('Managed identity mismatch')
        for k in ('source_id','evaluation_id','contract_id','evaluator_sha256','adapter_sha256'):evidence.checked_id(r[k])
        models=[x for x in r['inputs'] if x['role']=='model']
        if len(models)!=1 or models[0]['sha256']!=aid or models[0]['bytes']!=size:raise ValueError('Managed model input mismatch')
        if r['evaluation_schema']==evaluation.SCHEMA:optimize.profiles({'profiles':[{'budgetfit':p['metrics']} for p in r['profiles']]})
        elif r['evaluation_schema']=='ramgpt-adaptive-router-v1':
            if r['router']['passed']!=(r['status']=='heldout_pilot_passed'):raise ValueError('Router status mismatch')
        else:raise ValueError('Unknown evaluation schema')
    for d in body['optimization_decisions']:evidence.checked_id(d['decision_id'])
    if body['deployment']!={'authorized':False,'recommended_configuration':None,'status':'not_certified'}:raise ValueError('Unsupported deployment claim')
    no_private(body)
    return {'status':'content_integrity_verified','passport_id':document['passport_id'],'artifact_sha256':aid,'authenticity_verified':False,'source_evidence_verified':False,'trusted_expected_id_matched':expected_id is not None,'artifact_bytes_verified':False}

def verify_local(reg,store,document,expected_id=None,model=None):
    result=verify(document,expected_id)
    current=build(reg,store,result['artifact_sha256'])
    if current!=document:raise ValueError('Passport differs from current local evidence snapshot')
    result.update(status='local_evidence_summary_verified',local_snapshot_matches=True,source_evidence_verified=False,artifact_bytes_verified=False)
    if model is not None:
        r=fingerprint(model,'model')
        if r['sha256']!=result['artifact_sha256'] or r['bytes']!=document['passport']['artifact']['bytes']:raise ValueError('Model identity mismatch')
        result['artifact_bytes_verified']=True
    return result


def export(reg,store,aid,out):
    packet=build(reg,store,aid)
    verify(packet)
    evaluation.write_new(out,packet)
    b=packet['passport']
    return {'passport_id':packet['passport_id'],'artifact_sha256':aid,'out':str(out),'weight_records':len(b['weight_evidence']),'managed_evaluations':len(b['managed_evaluations']),'missing_axes':b['missing_axes'],'deployment_authorized':False}

def verify_model(document,expected_id=None,model=None):
    r=verify(document,expected_id)
    if model is not None:
        actual=fingerprint(model,'model')
        if actual['sha256']!=r['artifact_sha256'] or actual['bytes']!=document['passport']['artifact']['bytes']:raise ValueError('Model identity mismatch')
        r['artifact_bytes_verified']=True
    return r

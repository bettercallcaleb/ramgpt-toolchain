from pathlib import Path
import re
from registry import digest,fingerprint
import evidence
import evaluation
import native_bridge
import native_tools
import optimize
import passport
import portfolio
import weight
import workspace

SCHEMA='ramgpt-consolidated-passport-v2'

def native_summary(r):
    tool=r['tool']
    out=passport.fields(r,('receipt_id','tool','artifact_sha256','reference_sha256','weight_report_id','native_sha256','manifest_sha256','raw_sha256','report_sha256','source_sha256','scope'))
    if tool=='q4fit':out.update(passport.fields(r,('default','quantizer_sha256','imatrix_sha256','gguf_source_sha256')))
    elif tool=='gguf-healer':
        out['inputs']={k:passport.fields(v,('sha256','bytes')) for k,v in r['inputs'].items()}
        out['metrics']=passport.fields(r['result'],('growth','verified_payloads','metadata_sha256'))
        out['replaced_tensor_count']=len(r['result']['tensors'])
    elif tool=='conceptladder':
        out['metrics']=passport.fields(r['result'],('track','n','expected_n','complete','dataset_sha256','confirmed_correct_count','joint_accuracy','contiguous_depth_lower','contiguous_depth_upper','strict_prompt','loose_prompt','strict_instruction','loose_instruction','truncation_rate'))
        out.update(passport.fields(r,('official_full_dataset','limit','legacy_report_version_added','ifeval_source_hashes')))
    return out

def exl_summary(r):
    out=passport.fields(r,('report_id','scope','artifact_bytes','declared_bits','local_source_id','native_sha256','manifest_sha256','converter_sha256','backend_source_sha256','native_manifest_report_id','calibration_options'))
    out['kind']=r.get('kind','producer_manifest' if 'files' in r else 'producer_conversion')
    if r.get('kind')=='shared-native-evaluation':
        out.update(passport.fields(r,('manifest_sha256','schedule_sha256','evaluator_sha256','adapter_sha256','worker_sha256','context','sampling','reasoning_policy','native_cache_policy','gpu_name','gpu_driver','gpu_total_mib','requests','completion_cap')))
        out['profiles']=optimize.profiles(r)
        out['memory']=portfolio.exl_memory(r)
    return out

def resolve_subject(reg,store,value,native=None):
    p=Path(value).expanduser()
    if value.startswith('exl3:'):return 'exl3:'+evidence.checked_id(value[5:])
    if re.fullmatch('[0-9a-f]{64}',value):return value
    if p.is_dir():
        if not native:raise ValueError('EXL3 directory lookup requires configured native First Mover script')
        import cross_format
        return 'exl3:'+cross_format.model_identity(store,p.resolve(),native)[0]
    return reg.register(p,'model')['sha256']

def build(reg,store,subject):
    known={portfolio.subject(m):m for m in workspace.models(reg,store)['models']}
    if subject not in known:raise ValueError('Unknown model identity')
    m=known[subject]
    legacy=None;native=[];files=None
    if m['format']=='gguf':
        aid=m['sha256']
        if store.find(artifact=aid) or weight.find(store,artifact=aid) or reg.db.execute('SELECT bytes FROM artifacts WHERE sha256=?',(aid,)).fetchone():
            legacy=passport.build(reg,store,aid)
            passport.verify(legacy)
        native=[native_summary(r) for r in native_tools.find(store,aid)]
        missing=list(legacy['passport']['missing_axes']) if legacy else ['fidelity','knowledge_depth','quantization_recipe','managed_task_accuracy','sampled_vram','native_prefill_decode','independent_pilot']
        if 'knowledge_depth' in missing and any(r['tool']=='conceptladder' and r.get('metrics',{}).get('track')=='knowledge' for r in native):missing.remove('knowledge_depth')
        if not any(r.get('metrics',{}).get('track')=='ifeval' for r in native):missing.append('native_ifeval')
    else:
        rows=[r for r in native_bridge.find(store,'exl3-first-mover') if r['subject'].get('artifact_id')==m['artifact_id']]
        manifests=[r['files'] for r in rows if 'files' in r]
        if not manifests or any(f!=manifests[0] for f in manifests):raise ValueError('Missing or conflicting EXL3 manifests')
        files=manifests[0]
        if digest({'format':'exl3','files':files})!=m['artifact_id'] or sum(v['size'] for v in files.values())!=m['bytes']:raise ValueError('EXL3 manifest identity mismatch')
        native=[exl_summary(r) for r in rows]
        missing=['fidelity','knowledge_depth','native_ifeval','independent_pilot','frozen_runtime_provenance']
        if not any(r['kind']=='shared-native-evaluation' for r in native):missing.append('managed_task_accuracy')
        if not any(r.get('memory') for r in native):missing.append('sampled_vram')
    body={'schema':SCHEMA,'artifact':{k:v for k,v in m.items() if k!='sources'},'legacy_gguf_passport':legacy,'native_evidence':native,'exl3_files':files,'missing_axes':missing,'deployment_authorized':False,'scope':'Content-addressed historical summary; native evidence does not authenticate historical weights or imply independent validation. Unbound paired reports are excluded.'}
    passport.no_private(body)
    return {'schema':SCHEMA,'passport_id':digest(body),'passport':body}

def verify(packet,expected=None):
    if not isinstance(packet,dict) or set(packet)!={'schema','passport_id','passport'} or packet['schema']!=SCHEMA:raise ValueError('Unsupported consolidated passport')
    b=packet['passport']
    if set(b)!={'schema','artifact','legacy_gguf_passport','native_evidence','exl3_files','missing_axes','deployment_authorized','scope'} or b['schema']!=SCHEMA or b['deployment_authorized'] is not False:raise ValueError('Invalid passport fields')
    pid=evidence.checked_id(packet['passport_id'])
    if digest(b)!=pid or expected is not None and evidence.checked_id(expected)!=pid:raise ValueError('Passport digest mismatch')
    a=b['artifact'];weight.size(a['bytes'])
    if not isinstance(b['native_evidence'],list) or not isinstance(b['missing_axes'],list):raise ValueError('Invalid evidence collections')
    if a['format']=='gguf':
        aid=evidence.checked_id(a['sha256'])
        if b['exl3_files'] is not None:raise ValueError('Unexpected EXL3 manifest')
        if b['legacy_gguf_passport'] is not None:
            v=b['legacy_gguf_passport'];passport.verify(v)
            if v['passport']['artifact']['sha256']!=aid or v['passport']['artifact']['bytes']!=a['bytes']:raise ValueError('GGUF identity mismatch')
        for r in b['native_evidence']:
            evidence.checked_id(r['receipt_id'])
            if r['artifact_sha256']!=aid:raise ValueError('Native artifact mismatch')
    elif a['format']=='exl3':
        aid=evidence.checked_id(a['artifact_id'])
        f=b['exl3_files']
        if b['legacy_gguf_passport'] is not None or not isinstance(f,dict) or not f:raise ValueError('Expected EXL3 manifest only')
        for k,v in f.items():
            p=Path(k)
            if p.is_absolute() or '..' in p.parts or p.as_posix()!=k or '\\' in k or p.name=='README.md':raise ValueError('Invalid EXL3 relative path')
            if type(v['size']) is not int or v['size']<0:raise ValueError('Invalid EXL3 size')
            evidence.checked_id(v['sha256'])
        if digest({'format':'exl3','files':f})!=aid or sum(v['size'] for v in f.values())!=a['bytes']:raise ValueError('EXL3 identity mismatch')
        for r in b['native_evidence']:
            evidence.checked_id(r['report_id'])
            if r['artifact_bytes']!=a['bytes']:raise ValueError('Native artifact size mismatch')
    else:raise ValueError('Unsupported artifact format')
    passport.no_private(b)
    return {'status':'content_integrity_verified','passport_id':pid,'format':a['format'],'artifact_id':aid,'artifact_bytes_verified':False,'authenticity_verified':False,'source_evidence_verified':False,'trusted_expected_id_matched':expected is not None}

def verify_model(packet,expected=None,model=None,reg=None,store=None):
    result=verify(packet,expected);a=packet['passport']['artifact']
    if reg is not None:
        sid=result['artifact_id'] if a['format']=='gguf' else 'exl3:'+result['artifact_id']
        if build(reg,store,sid)!=packet:raise ValueError('Passport differs from current local evidence')
        result.update(status='local_evidence_summary_verified',local_snapshot_matches=True)
    if model is not None:
        p=Path(model).expanduser().resolve(strict=True)
        if a['format']=='gguf':
            r=fingerprint(p,'model')
            if r['sha256']!=a['sha256'] or r['bytes']!=a['bytes']:raise ValueError('Model identity mismatch')
        else:
            import hashlib
            if not p.is_dir() or any(x.is_symlink() for x in p.rglob('*')):raise ValueError('Expected materialized EXL3 directory')
            files={}
            for x in p.rglob('*'):
                if not x.is_file() or x.name=='README.md':continue
                r=fingerprint(x,'configuration') if x.stat().st_size else {'bytes':0,'sha256':hashlib.sha256(b'').hexdigest()}
                files[str(x.relative_to(p))]={'size':r['bytes'],'sha256':r['sha256']}
            if files!=packet['passport']['exl3_files']:raise ValueError('EXL3 files differ from passport')
        result['artifact_bytes_verified']=True
    return result

def export(reg,store,subject,out,native=None):
    packet=build(reg,store,resolve_subject(reg,store,subject,native))
    r=verify(packet);evaluation.write_new(out,packet)
    return {**r,'out':str(out),'native_records':len(packet['passport']['native_evidence']),'missing_axes':packet['passport']['missing_axes'],'deployment_authorized':False}

import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from registry import digest,fingerprint,read_json,now
import evidence

SCHEMA='ramgpt-weight-evidence-v1'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def number(v,lo=0,hi=math.inf):
    if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('Invalid numeric evidence')
    return v

def size(v):
    if type(v) is not int or v<=0:raise ValueError('Invalid artifact size')
    return v

def safe_file(root,name):
    p=(root/name).resolve(strict=True)
    if not p.is_relative_to(root) or not p.is_file():raise ValueError('Invalid evidence path')
    return p

def file_record(path):
    return {'sha256':sha(path),'bytes':path.stat().st_size}

def manifest_files(folder,m):
    files=m['files']
    if not isinstance(files,dict) or not files:raise ValueError('Empty release manifest')
    for name,expected in files.items():
        if file_record(safe_file(folder,name))!=expected:raise ValueError('Release manifest mismatch: '+name)
    return files

def same_metrics(a,b,tol=1e-10):
    if abs(a['mean_kl_bf16_to_artifact']-b['mean_kl_bf16_to_artifact'])>=tol:raise ValueError('KL mismatch')
    for k in ('top1_flips','hcdfr','max_flipped_bf16_margin'):
        if a[k]!=b[k]:raise ValueError('Metric mismatch: '+k)
    if len(a['segment_mean_kl'])!=len(b['segment_mean_kl']) or any(abs(x-y)>=tol for x,y in zip(a['segment_mean_kl'],b['segment_mean_kl'])):raise ValueError('Segment KL mismatch')

def compact_metrics(path,geometry,thresholds):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'kl','segment_id','top1_flip','bf16_margin'}:raise ValueError('Unknown compact evidence format')
        kl,seg,flip,margin=(z[k] for k in ('kl','segment_id','top1_flip','bf16_margin'))
    n=geometry['total_positions'];s=geometry['segments'];p=geometry['positions_per_segment']
    if n!=s*p or any(x.shape!=(n,) for x in (kl,seg,flip,margin)) or flip.dtype.kind!='b' or seg.dtype.kind not in 'iu' or not np.isfinite(kl).all() or not np.isfinite(margin).all() or (margin<0).any() or (kl< -1e-10).any():raise ValueError('Invalid compact arrays')
    ids,counts=np.unique(seg,return_counts=True)
    if len(ids)!=s or not (counts==p).all():raise ValueError('Compact geometry mismatch')
    hc={}
    for t in thresholds:
        eligible=margin>=t;den=int(eligible.sum());num=int((flip&eligible).sum())
        hc[str(t)]={'eligible_positions':den,'flips':num,'hcdf_rate':num/den if den else None}
    return {'mean_kl_bf16_to_artifact':float(kl.mean()),'top1_flips':int(flip.sum()),'hcdfr':hc,'max_flipped_bf16_margin':float(margin[flip].max()) if flip.any() else None,'segment_mean_kl':[float(kl[seg==x].mean()) for x in ids]}

def native_call(script,code,args):
    script=Path(script).resolve(strict=True);before=sha(script)
    p=subprocess.run([sys.executable,'-I','-c',code,str(script),*map(str,args)],text=True,capture_output=True,timeout=120)
    if p.returncode:raise ValueError('Native adapter failed: '+p.stderr[-2500:])
    if sha(script)!=before:raise ValueError('Native source changed')
    return json.loads(p.stdout),before

def quantbench(folder,native):
    root=Path(folder).resolve(strict=True)
    manifest=read_json(root/'release-manifest.json')
    if manifest['schema']!='ramgpt-artifact-bench-release-manifest-v1':raise ValueError('Unknown release manifest')
    files=manifest_files(root,manifest)
    d=read_json(root/'verified-result.json');spec=read_json(root/'run-spec.json');validation=read_json(root/'independent-validation.json')
    if d['schema']!='ramgpt-quantbench-verified-open-artifact-result' or d['schema_version']!=1 or d['status']!='PASS' or d['comparison_class']!='OPEN_ARTIFACT_DOWNLOAD_RANK':raise ValueError('Unsupported QuantBench report')
    if any(x['benchmark_id']!=d['benchmark_id'] for x in (spec,validation,manifest)) or spec['execution_contract']!=d['execution_contract'] or validation['status']!='PASS':raise ValueError('QuantBench release context mismatch')
    for k in ('model','quantization_label'):
        if spec[k]!=d[k]:raise ValueError('Release identity mismatch')
    ref=evidence.checked_id(d['reference_sha256']);rows=d['rankings'];g=d['geometry'];contract=d['execution_contract']
    if not rows or len({x['artifact_sha256'] for x in rows})!=len(rows) or len({x['artifact'] for x in rows})!=len(rows):raise ValueError('Duplicate or empty ranking')
    if g!={k:contract[k] for k in ('total_positions','segments','positions_per_segment')} or contract['reference']!='native BF16':raise ValueError('Reference geometry mismatch')
    unused=list(validation['artifacts'])
    for row in rows:
        aid=evidence.checked_id(row['artifact_sha256']);size(row['artifact_bytes']);number(row['mean_kl_bf16_to_artifact'])
        if aid==ref or row['strict_recipe_eligible'] is not False:raise ValueError('Invalid open-artifact provenance')
        name='compact/'+re.sub(r'[^a-z0-9]+','-',row['artifact'].lower()).strip('-')+'.npz'
        if name not in files or files[name]['sha256']!=row['compact_sha256']:raise ValueError('Compact hash mismatch')
        calculated=compact_metrics(safe_file(root,name),g,contract['hcdf_thresholds'])
        same_metrics(row,calculated)
        matches=[]
        for v in unused:
            if v['content']['status']!='PASS':raise ValueError('Failed native validation')
            try:same_metrics(row,v['content']['independent'])
            except ValueError:continue
            matches.append(v)
        if len(matches)!=1:raise ValueError('Missing or ambiguous independent metric validation')
        v=matches[0];unused.remove(v)
        log=safe_file(root,v['file'])
        if sha(log)!=v['sha256'] or read_json(log)!=v['content']:raise ValueError('Native validation log mismatch')
    if unused or rows!=sorted(rows,key=lambda x:x['mean_kl_bf16_to_artifact']) or d['winner']!=rows[0]['artifact']:raise ValueError('Invalid native rank')
    code="import json,runpy,sys; m=runpy.run_path(sys.argv[1]); d=json.load(open(sys.argv[2])); print(json.dumps(m['bootstrap'](d['rankings']),allow_nan=False))"
    boot,tool_hash=native_call(native,code,[root/'verified-result.json'])
    if boot!=d['paired_uncertainty']:raise ValueError('Native paired bootstrap mismatch')
    if manifest_files(root,manifest)!=files:raise ValueError('Release changed')
    source={'manifest':manifest,'result':d,'run_spec':spec,'independent_validation':validation,'native_script_sha256':tool_hash,'release_manifest_sha256':sha(root/'release-manifest.json')}
    records=[]
    for r in rows:
        records.append({'artifact_sha256':r['artifact_sha256'],'artifact_bytes':r['artifact_bytes'],'reference_sha256':ref,'reference_precision':'BF16','kind':'fidelity','scope':{'benchmark_id':d['benchmark_id'],'execution_contract':contract,'llama_cpp_commit':spec['llama_cpp_commit'],'comparison_class':d['comparison_class']},'metrics':{k:r[k] for k in ('mean_kl_bf16_to_artifact','top1_flips','hcdfr','max_flipped_bf16_margin','segment_mean_kl')},'provenance_grade':r['provenance_grade'],'strict_recipe_eligible':False})
    return {'schema':SCHEMA,'producer':'quantbench','status':'historical_compact_and_native_bootstrap_replayed','source':source,'records':records,'limitations':['Model binaries and full logits not reverified; native independent validation is historical.','Different benchmark IDs are not pooled.','Open-artifact evidence is not controlled recipe evidence.']}

def q4fit(folder,native):
    root=Path(folder).resolve(strict=True)
    code="import json,runpy,sys; from pathlib import Path; m=runpy.run_path(sys.argv[1]); print(json.dumps(m['build'](Path(sys.argv[2])),allow_nan=False))"
    d,tool_hash=native_call(native,code,[root])
    receipt=read_json(root/'receipt.json')
    if d['schema']!='ramgpt-experiment-v1' or d['status']!='exploratory':raise ValueError('Unknown Q4Fit export')
    for name,h in d['evidence_sha256'].items():
        if sha(safe_file(root,name))!=h:raise ValueError('Depth source changed')
    r=d['artifacts'];result=d['results'];n=d['evaluation']['items']
    for k in ('candidate','reference'):
        evidence.checked_id(r[k]['sha256']);size(r[k]['bytes'])
    for k in ('candidate_accuracy','reference_accuracy'):number(result[k],0,1)
    if abs(result['reference_correct']/n-result['reference_accuracy'])>1e-6 or abs((result['reference_correct']-result['lost_count']+result['gained_count'])/n-result['candidate_accuracy'])>1e-6:raise ValueError('Depth count mismatch')
    return {'schema':SCHEMA,'producer':'q4fit','status':'historical_native_export_replayed','source':{'export':d,'native_script_sha256':tool_hash,'documents':{name:read_json(safe_file(root,name)) for name in d['evidence_sha256']}},'records':[{'artifact_sha256':r['candidate']['sha256'],'artifact_bytes':r['candidate']['bytes'],'reference_sha256':r['reference']['sha256'],'reference_bytes':r['reference']['bytes'],'reference_precision':'unknown','kind':'knowledge_depth','scope':d['evaluation'],'metrics':result}],'limitations':d['limitations']+['Reference precision not declared by native depth export; do not infer BF16.']}

def recipe(manifest_path):
    p=Path(manifest_path).resolve(strict=True);d=read_json(p)
    ref=fingerprint(d['source'],'model');candidate=fingerprint(d['output'],'model')
    if ref['sha256']!=d['source_sha256'] or candidate['sha256']!=d['output_sha256'] or candidate['bytes']!=d['bytes']:raise ValueError('Recipe artifact mismatch')
    folder=Path(d['output']).with_suffix('').with_name(Path(d['output']).stem+'-recipe')
    r=read_json(folder/'recipe.json')
    if r['default']!=d['default'] or r['overrides']!=d['overrides'] or r['estimated_bytes']!=d['predicted_bytes']:raise ValueError('Recipe manifest mismatch')
    types=folder/'tensor-types.txt'
    return {'schema':SCHEMA,'producer':'q4fit','status':'local_artifact_bytes_verified','source':{'manifest':d,'recipe':r,'manifest_sha256':sha(p),'recipe_sha256':sha(folder/'recipe.json'),'tensor_types_sha256':sha(types)},'records':[{'artifact_sha256':candidate['sha256'],'artifact_bytes':candidate['bytes'],'reference_sha256':ref['sha256'],'reference_bytes':ref['bytes'],'reference_precision':'unknown','kind':'quantization_recipe','scope':{'recipe_format':'q4fit build manifest'},'metrics':{'default':d['default'],'overrides':d['overrides'],'predicted_bytes':d['predicted_bytes'],'imatrix':d['imatrix']}}],'limitations':['Current bytes match native manifest; historical quantizer execution and reference precision are not independently proven.']}

def init(store):
    store.db.executescript("""
    CREATE TABLE IF NOT EXISTS weight_reports(id TEXT PRIMARY KEY REFERENCES blobs(id),producer TEXT NOT NULL,status TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS weight_records(report_id TEXT NOT NULL REFERENCES weight_reports(id),ordinal INTEGER NOT NULL,artifact_sha256 TEXT NOT NULL,reference_sha256 TEXT NOT NULL,kind TEXT NOT NULL,PRIMARY KEY(report_id,ordinal));
    CREATE INDEX IF NOT EXISTS weight_artifact_idx ON weight_records(artifact_sha256);
    CREATE INDEX IF NOT EXISTS weight_reference_idx ON weight_records(reference_sha256);
    """)

def ingest(store,report):
    init(store)
    if report['schema']!=SCHEMA or not report['records']:raise ValueError('Invalid weight report')
    for r in report['records']:
        evidence.checked_id(r['artifact_sha256']);evidence.checked_id(r['reference_sha256']);size(r['artifact_bytes'])
        if r['artifact_sha256']==r['reference_sha256']:raise ValueError('Identical candidate and reference')
    with store.db:
        bid=store.put_blob(report)
        store.db.execute('INSERT OR IGNORE INTO weight_reports VALUES (?,?,?)',(bid,report['producer'],report['status']))
        for i,r in enumerate(report['records']):
            store.db.execute('INSERT OR IGNORE INTO weight_records VALUES (?,?,?,?,?)',(bid,i,r['artifact_sha256'],r['reference_sha256'],r['kind']))
        if store.db.execute('SELECT producer,status FROM weight_reports WHERE id=?',(bid,)).fetchone()!=(report['producer'],report['status']):raise ValueError('Weight report index mismatch')
        expected=[(i,r['artifact_sha256'],r['reference_sha256'],r['kind']) for i,r in enumerate(report['records'])]
        if store.db.execute('SELECT ordinal,artifact_sha256,reference_sha256,kind FROM weight_records WHERE report_id=? ORDER BY ordinal',(bid,)).fetchall()!=expected:raise ValueError('Weight record index mismatch')
    return {'report_id':bid,'status':report['status'],'records':len(report['records']),'new_inference_requests':0}

def find(store,artifact=None,reference=None):
    init(store);where=[];args=[]
    for k,v in [('artifact_sha256',artifact),('reference_sha256',reference)]:
        if v is not None:where.append('r.'+k+'=?');args.append(evidence.checked_id(v))
    q='SELECT r.report_id,r.ordinal,r.artifact_sha256,r.reference_sha256,r.kind,w.producer,w.status FROM weight_records r JOIN weight_reports w ON r.report_id=w.id'
    if where:q+=' WHERE '+' AND '.join(where)
    result=[]
    for bid,i,a,ref,kind,producer,status in store.db.execute(q+' ORDER BY r.report_id,r.ordinal',args):
        body=store.get(bid);r=body['records'][i]
        if (r['artifact_sha256'],r['reference_sha256'],r['kind'],body['producer'],body['status'])!=(a,ref,kind,producer,status):raise ValueError('Weight index integrity mismatch')
        result.append({'report_id':bid,'ordinal':i,'producer':producer,'status':status,**r})
    return result

def artifact_view(store,artifact):
    evidence.checked_id(artifact)
    rows=find(store,artifact=artifact)
    managed=store.find(artifact=artifact)
    sizes={r['artifact_bytes'] for r in rows}
    task=[]
    for m in managed:
        source=store.get(m['source_id'])
        ref=next(r for r in source['contract']['inputs'] if r['role']=='model')
        sizes.add(ref['bytes'])
        report=store.get(m['evaluation_id'])['report']
        task.append({'source_id':m['source_id'],'evaluation_id':m['evaluation_id'],'contract_id':m['contract_id'],'evaluator_sha256':m['evaluator_sha256'],'schema':report['schema'],'profiles':[p['budgetfit'] for p in report.get('profiles',[])],'validation':report.get('validation'),'status':report.get('status','observed')})
    if len(sizes)>1:raise ValueError('Conflicting artifact size')
    return {'artifact_sha256':artifact,'artifact_bytes':next(iter(sizes),None),'weight_evidence':rows,'managed_task_evidence':task,'missing_axes':[k for k,v in [('fidelity',any(x['kind']=='fidelity' for x in rows)),('knowledge_depth',any(x['kind']=='knowledge_depth' for x in rows)),('quantization_recipe',any(x['kind']=='quantization_recipe' for x in rows)),('managed_task_accuracy',bool(task))] if not v],'scope':'Exact artifact identity only; no metric equivalence or composite score'}

def select_fidelity(store,report_id,max_kl,max_bytes):
    evidence.checked_id(report_id);number(max_kl);size(max_bytes)
    report=store.get(report_id)
    if report['schema']!=SCHEMA or report['producer']!='quantbench' or report['status']!='historical_compact_and_native_bootstrap_replayed':raise ValueError('Require one verified historical QuantBench cohort')
    indexed=[r for r in find(store) if r['report_id']==report_id]
    if len(indexed)!=len(report['records']):raise ValueError('Missing cohort index')
    rows=report['records']
    if len({(r['reference_sha256'],digest(r['scope'])) for r in rows})!=1:raise ValueError('Incomparable fidelity cohort')
    scored=[]
    for r in rows:
        why=[]
        if r['metrics']['mean_kl_bf16_to_artifact']>max_kl:why.append('fidelity_limit')
        if r['artifact_bytes']>max_bytes:why.append('size_limit')
        scored.append({'artifact_sha256':r['artifact_sha256'],'artifact_bytes':r['artifact_bytes'],'mean_kl':r['metrics']['mean_kl_bf16_to_artifact'],'rejections':why})
    eligible=[r for r in scored if not r['rejections']]
    frontier=[r for r in eligible if not any(q['artifact_bytes']<=r['artifact_bytes'] and q['mean_kl']<=r['mean_kl'] and (q['artifact_bytes']<r['artifact_bytes'] or q['mean_kl']<r['mean_kl']) for q in eligible)]
    return {'status':'historical_shortlist' if frontier else 'no_eligible_candidate','report_id':report_id,'reference_sha256':rows[0]['reference_sha256'],'max_kl':max_kl,'max_bytes':max_bytes,'screening':scored,'pareto':frontier,'scope':'Fidelity and size only; not a deployment recommendation. Missing task and runtime evidence is not inferred.'}

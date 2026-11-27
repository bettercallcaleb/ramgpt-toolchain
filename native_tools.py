import json
import hashlib
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from registry import digest,fingerprint,read_json
import evidence

SCHEMA='ramgpt-native-tool-receipt-v1'

def ref(path,role='configuration'):
    return {k:v for k,v in fingerprint(path,role).items() if k!='path'}

def tree(root):
    root=Path(root).resolve(strict=True)
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts)
    if not files:raise ValueError('Empty native source tree')
    return digest([(str(p.relative_to(root)),(hashlib.sha256(b'').hexdigest() if p.stat().st_size==0 else ref(p)['sha256'])) for p in files])

def worker(code,argv,timeout=180,env=None):
    p=subprocess.run([sys.executable,'-I','-c',code,*map(str,argv)],capture_output=True,text=True,timeout=timeout,env=env)
    if p.returncode:raise ValueError('Native tool failed: '+p.stderr[-2500:])
    return json.loads(p.stdout)

def init(store):
    store.db.execute('CREATE TABLE IF NOT EXISTS native_tool_receipts(id TEXT PRIMARY KEY REFERENCES blobs(id), tool TEXT NOT NULL, artifact_sha256 TEXT NOT NULL)')

def save(store,tool,artifact,body):
    init(store)
    r={'schema':SCHEMA,'tool':tool,'artifact_sha256':artifact,**body,'deployment_authorized':False}
    with store.db:
        rid=store.put_blob(r)
        store.db.execute('INSERT OR IGNORE INTO native_tool_receipts VALUES (?,?,?)',(rid,tool,artifact))
        if store.db.execute('SELECT tool,artifact_sha256 FROM native_tool_receipts WHERE id=?',(rid,)).fetchone()!=(tool,artifact):raise ValueError('Receipt index mismatch')
    return {'receipt_id':rid,**r}

def find(store,artifact=None):
    init(store)
    if artifact is not None:evidence.checked_id(artifact)
    q='SELECT id,tool,artifact_sha256 FROM native_tool_receipts'
    rows=store.db.execute(q+(' WHERE artifact_sha256=?' if artifact else '')+' ORDER BY id',(artifact,) if artifact else ())
    out=[]
    for rid,t,a in rows:
        d=store.get(rid)
        if d['schema']!=SCHEMA or d['tool']!=t or d['artifact_sha256']!=a:raise ValueError('Native receipt index mismatch')
        out.append({'receipt_id':rid,**d})
    return out

HEALER_VERIFY="""
import json,runpy,sys
from pathlib import Path
sys.path.insert(0,sys.argv[2])
h=runpy.run_path(sys.argv[1])
base,donor,out,manifest=sys.argv[3:]
m=json.loads(Path(manifest).read_text())
a,b,x,y=h['inventory'](base,donor)
c=h['read'](out)
chosen=set(m['tensors'])
if not chosen or len(chosen)!=len(m['tensors']) or not chosen<=x.keys():raise ValueError('Invalid selected tensors')
if any(y[k].n_bytes<=x[k].n_bytes for k in chosen):raise ValueError('Donor storage must increase')
if h['metadata'](c)!=h['metadata'](a) or len(c.tensors)!=len(x):raise ValueError('Output metadata mismatch')
import numpy as np
for t,(name,bt) in zip(c.tensors,x.items()):
 s=y[name] if name in chosen else bt
 if t.name!=name or t.tensor_type!=s.tensor_type or t.n_bytes!=s.n_bytes or tuple(t.shape)!=tuple(s.shape):raise ValueError('Tensor descriptor mismatch: '+name)
 if not np.array_equal(t.data.view(np.uint8),s.data.view(np.uint8)):raise ValueError('Tensor payload mismatch: '+name)
print(json.dumps({'verified_payloads':len(x),'metadata_sha256':h['metadata'](c),'tensors':sorted(chosen)}))
"""

def healer_import(reg,store,manifest,native,gguf_root,max_growth):
    if not math.isfinite(max_growth) or not 0<=max_growth<=1:raise ValueError('Invalid size growth limit')
    manifest=Path(manifest).resolve(strict=True);m=read_json(manifest)
    paths=[Path(m[k]).resolve(strict=True) for k in ('base','donor','output')]
    if len(set(paths))!=3:raise ValueError('Distinct model files required')
    ids=[fingerprint(p,'model') for p in paths]
    if len({r['sha256'] for r in ids})!=3:raise ValueError('Distinct artifacts required')
    before=ref(manifest);script=ref(native);library=tree(Path(gguf_root)/'gguf')
    growth=ids[2]['bytes']/ids[0]['bytes']-1
    if m['bytes']!=ids[2]['bytes'] or not math.isclose(m['growth'],growth,abs_tol=1e-12) or not 0<growth<=max_growth:raise ValueError('Manifest size or growth mismatch')
    result=worker(HEALER_VERIFY,[Path(native).resolve(),Path(gguf_root).resolve(),*paths,manifest],300)
    if any(result[k]!=m[k] for k in ('verified_payloads','metadata_sha256')) or result['tensors']!=sorted(m['tensors']):raise ValueError('Native verification mismatch')
    if before!=ref(manifest) or script!=ref(native) or library!=tree(Path(gguf_root)/'gguf') or ids!=[fingerprint(p,'model') for p in paths]:raise ValueError('Input changed during verification')
    with reg.db:
        for r in ids:reg.remember(r)
    portable=[{k:v for k,v in r.items() if k!='path'} for r in ids]
    return save(store,'gguf-healer',ids[2]['sha256'],{'inputs':{'base':portable[0],'donor':portable[1],'output':portable[2]},'manifest_sha256':before['sha256'],'native_sha256':script['sha256'],'gguf_source_sha256':library,'result':{**result,'growth':growth},'scope':'Current tensor payload provenance verified; checkpoint lineage and behavioral recovery are not established.'})

LADDER_SCORE="""
import runpy,sys
sys.path.insert(0,sys.argv[1])
if sys.argv[2]:sys.path.insert(1,sys.argv[2])
sys.argv=['conceptladder','measure','score',*sys.argv[3:]]
runpy.run_module('conceptladder',run_name='__main__')
"""

def ladder_import(reg,store,model,raw,native,seed,variants,track="knowledge",ifeval_root=None,limit=None,deps=None,nltk_data=None):
    import os
    raw=Path(raw).resolve(strict=True);native=Path(native).resolve(strict=True)
    manifest=raw.with_suffix(raw.suffix+'.manifest.json')
    report=raw.with_suffix(raw.suffix+'.report.json')
    m=read_json(manifest);expected=read_json(report)
    if track not in ('knowledge','ifeval') or m['track']!=track or m['version']!='measurement-v0.5':raise ValueError('Expected matching native v0.5 track')
    if track=='knowledge' and any(x is not None for x in (ifeval_root,limit,deps,nltk_data)):raise ValueError('IFEval options require --track ifeval')
    if track=='ifeval' and not ifeval_root:raise ValueError('IFEval requires --ifeval-root')
    if limit is not None and (type(limit) is not int or not 1<=limit<=541):raise ValueError('Invalid IFEval subset size')
    artifact=fingerprint(model,'model')
    profile=m['profile']
    parts=profile.split('; ')
    if 'model='+artifact['path'] not in parts or 'bytes='+str(artifact['bytes']) not in parts:raise ValueError('Manifest model path or size mismatch')
    refs=[ref(x) for x in (raw,manifest,report)];source=tree(native/'conceptladder')
    with tempfile.TemporaryDirectory() as tmp:
        copy=Path(tmp)/'responses.jsonl'
        copy.write_bytes(raw.read_bytes())
        copy.with_suffix('.jsonl.manifest.json').write_bytes(manifest.read_bytes())
        args=[native,Path(deps).resolve(strict=True) if deps else '', '--track',track,'--seed',seed,'--variants',variants,'--out',copy]
        env=dict(os.environ)
        if track=='ifeval':
            args+=['--ifeval-root',Path(ifeval_root).resolve(strict=True)]
            if limit is not None:args+=['--limit',limit]
            if nltk_data:env['NLTK_DATA']=str(Path(nltk_data).resolve(strict=True))
        actual=worker(LADDER_SCORE,args,env=env)
    legacy_report_version=(track=='ifeval' and 'report_version' not in expected and actual.get('report_version')=='measurement-report-v0.5.1')
    comparable={k:v for k,v in actual.items() if not (legacy_report_version and k=='report_version')}
    if comparable!=expected or actual['complete'] is not True or actual['dataset_sha256']!=m['dataset_sha256']:raise ValueError('Native report parity or completeness failure')
    if refs!=[ref(x) for x in (raw,manifest,report)] or source!=tree(native/'conceptladder') or artifact!=fingerprint(model,'model'):raise ValueError('Input changed during replay')
    if track=='ifeval' and (actual.get('ifeval_source_hashes')!=m.get('ifeval_source_hashes') or actual['expected_n']!=(limit or 541)):raise ValueError('IFEval source or subset mismatch')
    with reg.db:reg.remember(artifact)
    return save(store,'conceptladder',artifact['sha256'],{'artifact_bytes':artifact['bytes'],'source_sha256':source,'raw_sha256':refs[0]['sha256'],'manifest_sha256':refs[1]['sha256'],'report_sha256':refs[2]['sha256'],'seed':seed,'variants':variants,'result':actual,**({'track':'ifeval','limit':limit,'official_full_dataset':limit in (None,541),'legacy_report_version_added':legacy_report_version,'ifeval_source_hashes':actual['ifeval_source_hashes']} if track=='ifeval' else {}),'scope':'Historical native regrading; current artifact matches declared path and size only. Historical weight identity and execution settings are not independently established.'})


HEALER_COMPOSE="""
import json,runpy,sys
sys.path.insert(0,sys.argv[2])
h=runpy.run_path(sys.argv[1])
base,donor,out,growth,*names=sys.argv[3:]
print(json.dumps(h['compose'](base,donor,out,names,float(growth))))
"""

def healer_compose(reg,store,base,donor,out,native,gguf_root,tensors,max_growth,execute=False):
    if execute is not True:raise ValueError('Explicit --execute required')
    if not math.isfinite(max_growth) or not 0<max_growth<=1:raise ValueError('Invalid size growth limit')
    if not tensors or len(set(tensors))!=len(tensors) or any(not x for x in tensors):raise ValueError('Require unique tensor names')
    base=Path(base).resolve(strict=True);donor=Path(donor).resolve(strict=True);out=Path(out).resolve()
    if out.suffix.lower()!='.gguf' or out.exists() or out.with_suffix('.manifest.json').exists() or out in (base,donor):raise ValueError('Require a new GGUF output and manifest path')
    before=[fingerprint(p,'model') for p in (base,donor)]
    if before[0]['sha256']==before[1]['sha256']:raise ValueError('Distinct base and donor required')
    script=ref(native);library=tree(Path(gguf_root)/'gguf')
    result=worker(HEALER_COMPOSE,[Path(native).resolve(),Path(gguf_root).resolve(),base,donor,out,max_growth,*tensors],300)
    manifest=out.with_suffix('.manifest.json')
    if result!=read_json(manifest) or result['tensors']!=sorted(tensors):raise ValueError('Native composition manifest mismatch')
    if before!=[fingerprint(p,'model') for p in (base,donor)] or script!=ref(native) or library!=tree(Path(gguf_root)/'gguf'):raise ValueError('Inputs changed during composition; output retained for inspection')
    receipt=healer_import(reg,store,manifest,native,gguf_root,max_growth)
    if [receipt['inputs'][k]['sha256'] for k in ('base','donor')]!=[r['sha256'] for r in before] or receipt['native_sha256']!=script['sha256'] or receipt['gguf_source_sha256']!=library:raise ValueError('Composition receipt mismatch')
    return receipt

Q4FIT_BUILD="""
import contextlib,io,json,runpy,sys
from pathlib import Path
native,gguf,model,out,quantizer,default,threads,imatrix,*changes=sys.argv[1:]
sys.path.insert(0,gguf)
sys.path.insert(0,str(Path(native).parent))
h=runpy.run_path(native)
with contextlib.redirect_stdout(sys.stderr):
 ts=h['inventory'](model)
 if not any(t['eligible'] for t in ts) or any(t['source'] not in ('F32','F16','BF16') for t in ts):raise ValueError('Build requires a floating-point GGUF source, not a requantization')
 h['build'](model,out,quantizer,default,changes,imatrix or None,int(threads),False)
print(json.dumps(json.loads(Path(out).with_suffix('.manifest.json').read_text())))
"""

def q4fit_build(reg,store,model,out,native,gguf_root,quantizer,default,changes,threads,imatrix=None,execute=False):
    import os
    import weight
    if execute is not True:raise ValueError('Explicit --execute required for weight construction')
    if default not in ('Q3_K','Q4_K','Q5_K','Q6_K','Q8_0','F16','BF16'):raise ValueError('Unsupported native precision')
    if type(threads) is not int or not 1<=threads<=128:raise ValueError('Invalid thread count')
    model=Path(model).resolve(strict=True);native=Path(native).resolve(strict=True);gguf_root=Path(gguf_root).resolve(strict=True);quantizer=Path(quantizer).resolve(strict=True);out=Path(out).resolve()
    if native.name!='q4fit.py' or quantizer.name!='llama-quantize' or not os.access(quantizer,os.X_OK):raise ValueError('Expected native q4fit.py and executable llama-quantize')
    if out.suffix.lower()!='.gguf' or out==model:raise ValueError('Expected distinct GGUF output')
    if out.parent.exists():raise ValueError('Use a new output directory to protect recipe and log files')
    source=fingerprint(model,'model');script=ref(native);lib=tree(gguf_root/'gguf');binary=ref(quantizer,'runtime')
    matrix=Path(imatrix).resolve(strict=True) if imatrix else None
    matrix_ref=ref(matrix) if matrix else None
    out.parent.mkdir(parents=True,exist_ok=False)
    manifest=worker(Q4FIT_BUILD,[native,gguf_root,model,out,quantizer,default,threads,matrix or '',*changes],1800)
    path=out.with_suffix('.manifest.json')
    if manifest!=read_json(path) or manifest['source_sha256']!=source['sha256'] or manifest['default']!=default:raise ValueError('Native build manifest mismatch')
    if source!=fingerprint(model,'model') or script!=ref(native) or lib!=tree(gguf_root/'gguf') or binary!=ref(quantizer,'runtime') or (matrix is not None and matrix_ref!=ref(matrix)):raise ValueError('Build input changed; output retained for inspection')
    report=weight.recipe(path);row=report['records'][0]
    result=weight.ingest(store,report)
    with reg.db:
        reg.remember(source)
        reg.remember({'role':'model','path':str(out),'sha256':row['artifact_sha256'],'bytes':row['artifact_bytes']})
    return save(store,'q4fit',row['artifact_sha256'],{'artifact_bytes':row['artifact_bytes'],'reference_sha256':source['sha256'],'weight_report_id':result['report_id'],'native_sha256':script['sha256'],'gguf_source_sha256':lib,'quantizer_sha256':binary['sha256'],'imatrix_sha256':matrix_ref['sha256'] if matrix_ref else None,'default':default,'overrides':manifest['overrides'],'scope':'Native Q4Fit build completed and output recipe verified; no inference or model quality claim.'})

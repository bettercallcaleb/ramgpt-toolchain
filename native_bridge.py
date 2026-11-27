import json
import math
import re
import tempfile
from pathlib import Path
from registry import digest,read_json
import evidence
import native_tools as nt

SCHEMA='ramgpt-native-report-v1'

def save(store,tool,subject,body):
    store.db.execute('CREATE TABLE IF NOT EXISTS native_reports(id TEXT PRIMARY KEY REFERENCES blobs(id), tool TEXT NOT NULL)')
    report={'schema':SCHEMA,'tool':tool,'subject':subject,**body,'deployment_authorized':False}
    with store.db:
        rid=store.put_blob(report)
        store.db.execute('INSERT OR IGNORE INTO native_reports VALUES (?,?)',(rid,tool))
    return {'report_id':rid,**report}

def find(store,tool=None):
    store.db.execute('CREATE TABLE IF NOT EXISTS native_reports(id TEXT PRIMARY KEY REFERENCES blobs(id), tool TEXT NOT NULL)')
    rows=store.db.execute('SELECT id,tool FROM native_reports'+(' WHERE tool=?' if tool else '')+' ORDER BY id',(tool,) if tool else ())
    result=[]
    for rid,t in rows:
        r=store.get(rid)
        if r['schema']!=SCHEMA or r['tool']!=t:raise ValueError('Native report index mismatch')
        result.append({'report_id':rid,**r})
    return result

EXL3_PREFLIGHT="""
import contextlib,io,json,runpy,sys
from pathlib import Path
native,repo,backend,root,bits=sys.argv[1:]
h=runpy.run_path(native)
h['CFG'].update(workspace_root=str(Path(root)/'work'),exllamav3_repo=backend)
with contextlib.redirect_stdout(io.StringIO()):
 r=h['preflight'](repo,{'targets':[float(bits)],'allow_large':False})
print(json.dumps(r))
"""

def exl3_preflight(store,repo,native,backend,bits=4.0):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repo) or '..' in repo:raise ValueError('Expected owner/model')
    if not math.isfinite(bits) or not 2<=bits<=8:raise ValueError('Invalid target bitrate')
    native=Path(native).resolve(strict=True);backend=Path(backend).resolve()
    if native.name!='fm.py':raise ValueError('Expected native First Mover fm.py')
    if (native.parent/'config.json').exists():raise ValueError('Supply a clean native fm.py without local config overrides')
    source=nt.ref(native);arch=backend/'exllamav3/architecture'
    arch_id=nt.tree(arch) if arch.is_dir() else None
    with tempfile.TemporaryDirectory() as tmp:
        r=nt.worker(EXL3_PREFLIGHT,[native,repo,backend,tmp,bits],180)
    if source!=nt.ref(native) or arch_id!=(nt.tree(arch) if arch.is_dir() else None):raise ValueError('Native inputs changed')
    if r['repo_id']!=repo or r['targets']!=[bits]:raise ValueError('Native preflight identity mismatch')
    if r['sweet_spot'] not in ('GREEN','AMBER','BLOCKED','OVERRIDE'):raise ValueError('Unrecognized native gate')
    ready=r['sweet_spot'] in ('GREEN','AMBER') and r['exllamav3_arch_supported'] is True
    return save(store,'exl3-first-mover',{'repo_id':repo,'source_revision':r['source_revision']},{'native_sha256':source['sha256'],'architecture_source_sha256':arch_id,'result':r,'conversion_preflight_passed':ready,'scope':'Native metadata/config preflight only; no weights downloaded or converted; no GPU validation. Architecture detection is native source inspection, not a runtime guarantee.'})

HERETIC_REPLAY="""
import json,sys,hashlib
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,sys.argv[1])
from hereticbench.grading import grade_dual
from hereticbench.common import load_jsonl
root,ds,ap,bp,track=sys.argv[1:]
items=load_jsonl(ds); a=load_jsonl(ap); b=load_jsonl(bp)
def index(rows):
 out={r['id']:r for r in rows}
 if len(out)!=len(rows):raise ValueError('Duplicate task')
 return out
D,A,B=map(index,(items,a,b))
if not D or set(D)!=set(A) or set(D)!=set(B):raise ValueError('Incomplete paired dataset')
ds_sha=hashlib.sha256(Path(ds).read_bytes()).hexdigest()
groups=defaultdict(list);tracks=defaultdict(list)
allowed={'retention','instruction_v3'} if track=='retention' else {'epistemic'}
for k,d in D.items():
 if d['track'] not in allowed:raise ValueError('Wrong track')
 pair=[]
 for r in (A[k],B[k]):
  if r['status']!='ok' or r['dataset_sha256']!=ds_sha or any(r.get(f)!=d.get(f) for f in ('track','category','messages','grading')):raise ValueError('Dataset binding mismatch')
  s,f,_=grade_dual(r['content'],d['grading'])
  if s not in (0,1) or f not in (0,1) or (s,f)!=(r['semantic_score'],r['format_score']):raise ValueError('Native grade parity failure')
  pair.append((s,f))
 groups[d['category']].append(pair)
 tracks[d['track']].append(pair)
def summary(rows):
 return {'n':len(rows),'base_semantic':sum(x[0][0] for x in rows),'variant_semantic':sum(x[1][0] for x in rows),'base_format':sum(x[0][1] for x in rows),'variant_format':sum(x[1][1] for x in rows),'base_only':sum(x[0][0]>x[1][0] for x in rows),'variant_only':sum(x[1][0]>x[0][0] for x in rows)}
for rows in (a,b):
 for k in ('label','server_model_path','server_build_info','reasoning_effort'):
  if len({r.get(k) for r in rows})!=1:raise ValueError('Mixed run identity')
print(json.dumps({'overall':summary([r for g in groups.values() for r in g]),'categories':{k:summary(v) for k,v in sorted(groups.items())},'tracks':{k:summary(v) for k,v in sorted(tracks.items())},'native_grade_parity':2*len(D)}))
"""

def heretic_import(store,native,dataset,base,variant,track):
    if track not in ('retention','epistemic'):raise ValueError('Supported tracks: retention, epistemic')
    native=Path(native).resolve(strict=True)
    paths=[Path(x).resolve(strict=True) for x in (dataset,base,variant)]
    if len(set(paths))!=3:raise ValueError('Distinct dataset and run files required')
    refs=[nt.ref(p) for p in paths];source=nt.tree(native/'hereticbench')
    r=nt.worker(HERETIC_REPLAY,[native,*paths,track])
    if refs!=[nt.ref(p) for p in paths] or source!=nt.tree(native/'hereticbench'):raise ValueError('Replay inputs changed')
    return save(store,'hereticbench',{'track':track,'dataset_sha256':refs[0]['sha256'],'base_raw_sha256':refs[1]['sha256'],'variant_raw_sha256':refs[2]['sha256']},{'native_source_sha256':source,'result':r,'scope':'Complete paired historical native semantic and format regrading; model weights not authenticated, no model artifact hash asserted, no causal claim or composite score.'})


EXL3_MANIFEST="""
import json,runpy,sys
from pathlib import Path
h=runpy.run_path(sys.argv[1])
print(json.dumps(h['manifest_dir'](Path(sys.argv[2]))))
"""

def exl3_import(store,folder,manifest,native):
    folder=Path(folder).resolve(strict=True);manifest=Path(manifest).resolve(strict=True);native=Path(native).resolve(strict=True)
    if not folder.is_dir() or native.name!='fm.py' or (native.parent/'config.json').exists():raise ValueError('Expected artifact directory and clean native fm.py')
    if any(p.is_symlink() for p in folder.rglob('*')):raise ValueError('EXL3 artifact symlinks are unsupported')
    source=nt.ref(native);mref=nt.ref(manifest);m=read_json(manifest)
    cfg=read_json(folder/'config.json')
    if cfg.get('quantization_config',{}).get('quant_method')!='exl3':raise ValueError('Expected EXL3 quantization config')
    bits=m['bits']
    if type(bits) not in (int,float) or not math.isfinite(bits) or bits<=0:raise ValueError('Invalid declared bitrate')
    files=nt.worker(EXL3_MANIFEST,[native,folder],300)
    if files!=m['files'] or 'config.json' not in files or not any(k.endswith('.safetensors') for k in files):raise ValueError('Native artifact manifest mismatch')
    for k,v in files.items():
        if Path(k).is_absolute() or '..' in Path(k).parts or type(v['size']) is not int or v['size']<0:raise ValueError('Invalid native file descriptor')
        evidence.checked_id(v['sha256'])
    if nt.ref(native)!=source or nt.ref(manifest)!=mref:raise ValueError('Manifest or producer changed')
    identity=digest({'format':'exl3','files':files})
    return save(store,'exl3-first-mover',{'format':'exl3','artifact_id':identity,'repo_id':m['repo_id'],'source_revision':m['source_revision']},{'native_sha256':source['sha256'],'manifest_sha256':mref['sha256'],'file_count':len(files),'artifact_bytes':sum(v['size'] for v in files.values()),'declared_bits':bits,'files':files,'scope':'Native directory manifest verified against current files, excluding README.md per producer protocol; not a GGUF SHA-256, not an independent runtime or historical conversion verification.'})

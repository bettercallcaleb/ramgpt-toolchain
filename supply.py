import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
import adaptive
import evaluation
import install_native
import workspace
from registry import digest,read_json

SCHEMA='ramgpt-native-supply-v1'
PACKS={
 'q4fit':('q4fit','35656748662f7954bb267b5a5f7a1eb411a4356a','dc6b65a8917bd3648746ca44adb4b724616753d415a36ee6658f47a8f20ad79b',{'native':'q4fit.py','evidence_script':'q4evidence.py'}),
 'quantbench':('ramgpt-quantbench','f9da19f1d05977907db99c7d92c3da71e9f32341','bd60bcfb0d6d15fb9e2e81069ac3f9ddd6205c634d709a395e02d8add989aaaf',{'native':'artifact-bench/run-artifact-comparison.py'}),
 'llama-autotune':('llama-autotune','59419e1a812d4c6cb82e485b85ac45e3a5bff5d6','c586a0fd6b0a7132e3798a4c3ba2752c94367a4a060466f5b46d22b2e5ae6c86',{'autotune_source':'.'}),
 'exl3-first-mover':('exl3-first-mover','5214474d0b3a393049d45b22b23aedff323669fb','bbe1380b33a35c82dafe596e5d46c930e6210e374cb8a3d5d628101c5204dfdd',{'native':'fm.py'}),
 'hereticbench':('HereticBench','9c1f5bd01ba869ce269b5e785a2d6f51702a77d2','d2b837c944e9ab877e3d39fd8362a84b80bfe1bac68629c1a20c8d53648283b0',{'native':'.'})
}
MAX_COMPRESSED=32*1024*1024
MAX_EXTRACTED=128*1024*1024
MAX_ENTRIES=10000

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def download(url,dest,expected,local=None,limit=MAX_COMPRESSED):
    h=hashlib.sha256();size=0
    src=Path(local).open('rb') if local is not None else urllib.request.urlopen(url,timeout=30)
    try:
        with dest.open('xb') as f:
            while chunk:=src.read(65536):
                size+=len(chunk)
                if size>limit:raise ValueError('Native archive exceeds download cap')
                h.update(chunk);f.write(chunk)
        if h.hexdigest()!=expected:raise ValueError('Pinned native archive digest mismatch: '+dest.name)
    finally:src.close()
    return size

def unpack(archive,dest):
    dest.mkdir()
    seen=set();prefix=None;total=0;count=0
    with tarfile.open(archive,'r:gz') as t:
        for m in t:
            count+=1;total+=m.size
            if count>MAX_ENTRIES or total>MAX_EXTRACTED:raise ValueError('Native archive extraction cap exceeded')
            p=PurePosixPath(m.name)
            if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in m.name or not (m.isfile() or m.isdir()):raise ValueError('Unsafe native archive entry')
            if prefix is None:prefix=p.parts[0]
            if p.parts[0]!=prefix:raise ValueError('Expected one native archive root')
            rel=p.parts[1:]
            if not rel:
                if not m.isdir():raise ValueError('Expected archive root directory')
                continue
            q=dest.joinpath(*rel)
            if q in seen:raise ValueError('Duplicate native archive path')
            seen.add(q)
            if m.isdir():q.mkdir(parents=True,exist_ok=True)
            else:
                q.parent.mkdir(parents=True,exist_ok=True)
                with t.extractfile(m) as source,q.open('xb') as f:shutil.copyfileobj(source,f)
    if not seen:raise ValueError('Empty native archive')

def files(root):
    result={}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Unexpected symlink in supplied tools')
        if p.is_file():
            if p.suffix=='.pyc' and '__pycache__' in p.parts:continue
            result[p.relative_to(root).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
        elif not p.is_dir():raise ValueError('Unexpected special file in supplied tools')
    return result

def plan():
    return {'schema':SCHEMA,'packages':{name:{'repository':'bettercallcaleb/'+repo,'revision':rev,'archive_sha256':h,'url':f'https://codeload.github.com/bettercallcaleb/{repo}/tar.gz/{rev}'} for name,(repo,rev,h,_) in PACKS.items()},'evaluator':{'url':install_native.URL,'sha256':adaptive.PIN},'external_requirements':['llama.cpp executable and GGUF Python source','ConceptLadder v0.5 source (public v0.4 incompatible)','GGUF-Healer local script','ExLlamaV3 backend and Python environment'],'scope':'Pinned archives, not automatic installation of Python/GPU dependencies or model weights.'}

def install(out,base_config=None,llama_root=None,archive_dir=None,evaluator=None,execute=False,runtime=None,quantizer=None):
    root=Path(out).expanduser().resolve()
    if root.exists():raise ValueError('Require fresh supply directory')
    body=plan()
    if not execute:return {'status':'supply_planned','out':str(root),**body}
    config=workspace.load(base_config)
    root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.ramgpt-supply-',dir=root.parent) as tmp:
        stage=Path(tmp)/'bundle';stage.mkdir()
        sources=stage/'sources';sources.mkdir()
        archives=stage/'archives';archives.mkdir()
        receipts={}
        for name,(repo,rev,h,mapping) in PACKS.items():
            archive=archives/(repo+'.tar.gz')
            local=Path(archive_dir)/archive.name if archive_dir else None
            url=body['packages'][name]['url']
            size=download(url,archive,h,local)
            dest=sources/name;unpack(archive,dest)
            for field,rel in mapping.items():
                actual=dest/rel
                if not actual.exists():raise ValueError('Pinned native layout mismatch: '+name+'.'+field)
                config.setdefault(name,{})[field]=(Path('sources')/name/rel).as_posix()
            receipts[name]={'archive':archive.relative_to(stage).as_posix(),'archive_bytes':size,'files':files(dest)}
        local=evaluator or (str(Path(archive_dir)/'budgetfit.pyz') if archive_dir else None)
        install_native.fetch(stage/'budgetfit.pyz',local)
        for name in ('budgetfit','trustmebro'):config.setdefault(name,{})['evaluator']='budgetfit.pyz'
        if llama_root:
            ll=Path(llama_root).expanduser().resolve(strict=True)
            for tool,key,p in [('llama-autotune','runtime',ll/'build/bin/llama-server'),('q4fit','quantizer',ll/'build/bin/llama-quantize'),('q4fit','gguf_root',ll/'gguf-py'),('gguf-healer','gguf_root',ll/'gguf-py')]:
                if key in ('runtime','quantizer') and not (p.is_file() and os.access(p,os.X_OK)):continue
                if key=='gguf_root' and not (p/'gguf').is_dir():continue
                config.setdefault(tool,{}).setdefault(key,str(p))
        for tool,key,value in [('llama-autotune','runtime',runtime),('q4fit','quantizer',quantizer)]:
            if value:
                p=Path(value).expanduser().resolve(strict=True)
                if not p.is_file() or not os.access(p,os.X_OK):raise ValueError('Expected executable: '+str(p))
                config.setdefault(tool,{})[key]=str(p)
        evaluation.write_new(stage/'tools.json',{'tools':config})
        lock={**body,'receipts':receipts,'config_sha256':sha(stage/'tools.json')}
        evaluation.write_new(stage/'supply-lock.json',{'lock_id':digest(lock),**lock})
        if root.exists():raise ValueError('Supply output already exists')
        stage.rename(root)
    return verify(root)

def verify(root):
    root=Path(root).expanduser().resolve(strict=True)
    if any((root/x).is_symlink() for x in ('supply-lock.json','tools.json','budgetfit.pyz','archives','sources')):raise ValueError('Unexpected supply symlink')
    lock=read_json(root/'supply-lock.json')
    body={k:v for k,v in lock.items() if k!='lock_id'}
    if lock['lock_id']!=digest(body) or any(body.get(k)!=v for k,v in plan().items()) or set(body['receipts'])!=set(PACKS):raise ValueError('Supply lock does not match pinned catalog')
    if sha(root/'tools.json')!=body['config_sha256'] or sha(root/'budgetfit.pyz')!=adaptive.PIN:raise ValueError('Supply configuration or evaluator changed')
    with tempfile.TemporaryDirectory() as tmp:
        for name,(repo,rev,h,mapping) in PACKS.items():
            r=body['receipts'][name];archive=root/'archives'/(repo+'.tar.gz')
            if archive.is_symlink() or r['archive']!=archive.relative_to(root).as_posix() or sha(archive)!=h or archive.stat().st_size!=r['archive_bytes']:raise ValueError('Supplied archive changed: '+name)
            dest=Path(tmp)/name
            unpack(archive,dest)
            if (root/'sources'/name).is_symlink() or files(dest)!=r['files'] or files(root/'sources'/name)!=r['files']:raise ValueError('Supplied source changed: '+name)
    cfg=workspace.load(root/'tools.json')
    for name,(_,_,_,mapping) in PACKS.items():
        for field,rel in mapping.items():
            if cfg.get(name,{}).get(field)!=str((root/'sources'/name/rel).resolve()):raise ValueError('Supply config points outside pinned source: '+name+'.'+field)
    for name in ('budgetfit','trustmebro'):
        if cfg.get(name,{}).get('evaluator')!=str(root/'budgetfit.pyz'):raise ValueError('Supply evaluator path mismatch')
    return {'status':'supply_verified','lock_id':lock['lock_id'],'config':str(root/'tools.json'),'packages':len(PACKS),'evaluator_verified':True,'tools':workspace.catalog(cfg)['tools'],'external_requirements':body['external_requirements'],'new_inference_requests':0,'scope':'Archive and extracted source integrity verified against pinned catalog; external dependencies not certified.'}

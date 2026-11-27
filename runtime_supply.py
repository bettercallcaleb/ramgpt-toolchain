import hashlib
import os
import platform
import stat
import tarfile
import tempfile
from pathlib import Path,PurePosixPath
import evaluation
import install_native
import runtime_opt
from registry import digest,read_json

PROFILE='cuda12.8'
TAG='b12810'
REV='ca123ba76311290ec6ab041dd2058fed8d9f03d2'
BINARY='llama-b12810-bin-ubuntu-cuda-12.8-x64.tar.gz'
SOURCE='llama-source.tar.gz'
ARCHIVES={
 BINARY:{'url':'https://github.com/ggml-org/llama.cpp/releases/download/'+TAG+'/'+BINARY,'sha256':'896094e79fac3bbb820cda906f0fbd0819ee3abb98fcf707b90d66e043efdc55','bytes':227286099},
 SOURCE:{'url':'https://codeload.github.com/ggml-org/llama.cpp/tar.gz/'+REV,'sha256':'2ec9fa358203214431fcef77944a349c99496d0f6e4d39825ab2f4a05069fa8b','bytes':39863454}
}
PATHS={'runtime':'bin/llama-server','quantizer':'bin/llama-quantize','gguf_root':'gguf-py'}
MAX_UNPACKED=2*2**30

def plan():
    return {'schema':'ramgpt-runtime-supply-v1','profile':PROFILE,'tag':TAG,'revision':REV,'archives':ARCHIVES,'paths':PATHS,'requirements':['Linux x86_64, glibc 2.39 or newer','Compatible NVIDIA driver and system CUDA 12 shared libraries','Python GPU extra and NumPy for Q4Fit'],'download_bytes':sum(x['bytes'] for x in ARCHIVES.values())}

def host_check():
    if platform.system()!='Linux' or platform.machine()!='x86_64':raise ValueError('Runtime pack requires Linux x86_64')
    libc,version=platform.libc_ver()
    if libc!='glibc' or tuple(map(int,version.split('.')[:2]))<(2,39):raise ValueError('Runtime pack requires glibc 2.39 or newer')

def manifest(archive,source=False,dest=None):
    entries={};root=None;total=0;count=0
    with tarfile.open(archive,'r:gz') as t:
        for m in t:
            count+=1;total+=m.size
            if count>100000 or total>MAX_UNPACKED:raise ValueError('Runtime archive exceeds extraction cap')
            p=PurePosixPath(m.name)
            if not p.parts or p.is_absolute() or '..' in p.parts or '\\' in m.name:raise ValueError('Unsafe runtime archive path')
            if root is None:root=p.parts[0]
            if root!=p.parts[0]:raise ValueError('Expected one archive root')
            rel=p.parts[1:]
            if not rel:continue
            if source:
                if rel[0]!='gguf-py':continue
                rel=rel[1:]
                if not rel:continue
            elif len(rel)!=1:raise ValueError('Expected flat binary archive')
            key='/'.join(rel)
            if key in entries:raise ValueError('Duplicate runtime archive path')
            q=dest.joinpath(*rel) if dest is not None else None
            if m.isdir():
                entries[key]={'kind':'directory'}
                if q is not None:q.mkdir(parents=True,exist_ok=True)
            elif m.isfile():
                mode=0o755 if m.mode&0o111 else 0o644
                h=hashlib.sha256()
                if q is not None:q.parent.mkdir(parents=True,exist_ok=True)
                with t.extractfile(m) as f:
                    out=q.open('xb') if q is not None else None
                    try:
                        while chunk:=f.read(1024*1024):
                            h.update(chunk)
                            if out:out.write(chunk)
                    finally:
                        if out:out.close()
                if q is not None:q.chmod(mode)
                entries[key]={'kind':'file','sha256':h.hexdigest(),'bytes':m.size,'mode':mode}
            elif m.issym() and not source:
                target=PurePosixPath(m.linkname)
                if len(target.parts)!=1 or target.is_absolute() or target.parts[0]=='..' or '\\' in m.linkname:raise ValueError('Unsafe runtime symlink')
                entries[key]={'kind':'symlink','target':m.linkname}
            else:raise ValueError('Unsupported runtime archive entry')
    if not entries:raise ValueError('Empty runtime archive selection')
    for name,r in entries.items():
        if r['kind']!='symlink':continue
        seen={name};target=r['target']
        while True:
            if target in seen or target not in entries:raise ValueError('Dangling or cyclic runtime symlink')
            seen.add(target);item=entries[target]
            if item['kind']=='file':break
            if item['kind']!='symlink':raise ValueError('Runtime symlink must resolve to file')
            target=item['target']
        if dest is not None:(dest/name).symlink_to(r['target'])
    return entries

def verify_tree(root,expected):
    actual={}
    for p in sorted(root.rglob('*')):
        k=p.relative_to(root).as_posix()
        if p.is_symlink():actual[k]={'kind':'symlink','target':os.readlink(p)}
        elif p.is_dir():
            if p.name=='__pycache__':continue
            actual[k]={'kind':'directory'}
        elif p.is_file():
            if p.suffix=='.pyc' and '__pycache__' in p.parts:continue
            with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
            actual[k]={'kind':'file','sha256':h,'bytes':p.stat().st_size,'mode':stat.S_IMODE(p.stat().st_mode)}
        else:raise ValueError('Unexpected runtime special file')
    if actual!=expected:raise ValueError('Installed runtime files changed')

def archive_manifest(root,extract=False):
    import supply
    result={}
    for name,desc in ARCHIVES.items():
        p=root/'archives'/name
        if p.is_symlink() or p.stat().st_size!=desc['bytes'] or supply.sha(p)!=desc['sha256']:raise ValueError('Runtime archive digest or size mismatch: '+name)
        dest=root/('gguf-py' if name==SOURCE else 'bin')
        if extract:dest.mkdir()
        if dest.is_symlink() or not dest.is_dir():raise ValueError('Unsafe runtime directory')
        expected=manifest(p,name==SOURCE,dest if extract else None)
        verify_tree(dest,expected)
        result[name]=digest(expected)
    return result

def verify(root):
    root=Path(root).resolve(strict=True)
    for p in ('archives','runtime-lock.json'):
        if (root/p).is_symlink():raise ValueError('Unsafe runtime bundle')
    body=read_json(root/'runtime-lock.json')
    expected={**plan(),'file_manifests':archive_manifest(root)}
    if body!={'lock_id':digest(expected),**expected}:raise ValueError('Runtime lock mismatch')
    return {'status':'runtime_supply_verified','lock_id':body['lock_id'],'profile':PROFILE,'paths':{k:str(root/v) for k,v in PATHS.items()},'requirements':body['requirements'],'scope':'Pinned archive and installed file integrity; system CUDA libraries remain external'}

def install(out,archive_dir=None,execute=False):
    root=Path(out).expanduser().resolve()
    if root.exists():raise ValueError('Require fresh runtime bundle directory')
    if not execute:return {'status':'runtime_supply_preview','out':str(root),**plan()}
    host_check()
    import supply
    root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.runtime-supply-',dir=root.parent) as tmp:
        stage=Path(tmp)/'runtime';stage.mkdir();(stage/'archives').mkdir()
        for name,desc in ARCHIVES.items():
            local=Path(archive_dir)/name if archive_dir else None
            n=supply.download(desc['url'],stage/'archives'/name,desc['sha256'],local,desc['bytes'])
            if n!=desc['bytes']:raise ValueError('Runtime download size mismatch')
        body={**plan(),'file_manifests':archive_manifest(stage,True)}
        evaluation.write_new(stage/'runtime-lock.json',{'lock_id':digest(body),**body})
        install_native.runtime_check(stage/PATHS['runtime'])
        runtime_opt.dependencies(stage/'bin/libggml-cuda.so')
        runtime_opt.dependencies(stage/PATHS['quantizer'])
        if root.exists():raise ValueError('Runtime output already exists')
        stage.rename(root)
    return verify(root)

import hashlib
import os
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from registry import fingerprint
import evidence

MAX_BYTES=128*2**30

def target(cache,sha):
    evidence.checked_id(sha)
    return Path(cache).expanduser().resolve()/sha/'model.gguf'

def lookup(cache,sha):
    p=target(cache,sha)
    if p.is_symlink() or p.parent.is_symlink():raise ValueError('Unexpected model cache symlink')
    r=fingerprint(p,'model')
    if r['sha256']!=sha:raise ValueError('Cached model digest mismatch')
    return r

def identity(repo,revision,name,sha,size):
    evidence.checked_id(sha)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*',repo) or '..' in repo:raise ValueError('Expected Hugging Face owner/repository')
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Require immutable 40-character repository revision')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.gguf',name,re.I) or '..' in name or re.search(r'-\d{5}-of-\d{5}\.gguf$',name,re.I):raise ValueError('Require one unsharded GGUF filename')
    if type(size) is not int or not 24<=size<=MAX_BYTES:raise ValueError('Invalid explicit download size')
    return {'repo':repo,'revision':revision,'filename':name,'sha256':sha,'bytes':size,'url':f'https://huggingface.co/{repo}/resolve/{revision}/{urllib.parse.quote(name)}'}

def place(reg,cache,sha,size,open_source):
    dest=target(cache,sha)
    if dest.exists():
        r=lookup(cache,sha)
        if r['bytes']!=size:raise ValueError('Cached model size mismatch')
        with reg.db:reg.remember(r)
        return {**r,'status':'cache_hit','new_bytes':0}
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.parent.is_symlink():raise ValueError('Unexpected model cache directory symlink')
    if shutil.disk_usage(dest.parent).free<size:raise ValueError('Insufficient free disk space')
    with tempfile.TemporaryDirectory(prefix='.incoming-',dir=dest.parent) as tmp:
        candidate=Path(tmp)/'model.gguf'
        h=hashlib.sha256();n=0
        with open_source() as src,candidate.open('xb') as f:
            while chunk:=src.read(8*1024*1024):
                n+=len(chunk)
                if n>size:raise ValueError('Model exceeds declared byte size')
                h.update(chunk);f.write(chunk)
            f.flush();os.fsync(f.fileno())
        if n!=size or h.hexdigest()!=sha:raise ValueError('Model size or SHA-256 mismatch')
        r=fingerprint(candidate,'model')
        if r['bytes']!=size or r['sha256']!=sha:raise ValueError('Downloaded model verification failed')
        try:os.link(candidate,dest)
        except FileExistsError:pass
    r=lookup(cache,sha)
    with reg.db:reg.remember(r)
    return {**r,'status':'cached_verified','new_bytes':size}

def add(reg,path,cache,execute=False):
    ref=fingerprint(path,'model')
    if not execute:return {**ref,'status':'cache_import_preview','destination':str(target(cache,ref['sha256'])),'new_bytes':0}
    return place(reg,cache,ref['sha256'],ref['bytes'],lambda:Path(ref['path']).open('rb'))

def fetch(reg,repo,revision,filename,sha,size,cache,execute=False):
    r=identity(repo,revision,filename,sha,size)
    if not execute:return {'status':'model_download_preview','source':r,'destination':str(target(cache,sha)),'new_bytes':0}
    return {**place(reg,cache,sha,size,lambda:urllib.request.urlopen(r['url'],timeout=60)),'source':r}

def resolve(value,cache):
    if re.fullmatch('[0-9a-f]{64}',value) and not Path(value).is_file():return lookup(cache,value)['path']
    return value

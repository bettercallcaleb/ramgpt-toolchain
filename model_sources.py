import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
import evaluation
import model_cache
from registry import digest,read_json

SCHEMA='ramgpt-model-source-v1'
METADATA_LIMIT=16*1024*1024
DEFAULT_LIMIT=8*2**30

def metadata(repo,revision='main'):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*',repo) or '..' in repo:raise ValueError('Expected Hugging Face owner/repository')
    if not isinstance(revision,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.\-/]{0,199}',revision) or '..' in revision:raise ValueError('Invalid repository ref')
    url=f'https://huggingface.co/api/models/{repo}/revision/{urllib.parse.quote(revision,safe="")}?blobs=true'
    with urllib.request.urlopen(url,timeout=30) as f:
        data=f.read(METADATA_LIMIT+1)
    if len(data)>METADATA_LIMIT:raise ValueError('Model metadata exceeds size cap')
    d=json.loads(data)
    if d.get('private') or d.get('gated'):raise ValueError('Only public ungated model repositories are supported')
    sha=d.get('sha')
    if not isinstance(sha,str) or not re.fullmatch('[0-9a-f]{40}',sha):raise ValueError('Missing immutable repository revision')
    if re.fullmatch('[0-9a-f]{40}',revision) and revision!=sha:raise ValueError('Repository revision mismatch')
    if not isinstance(d.get('id'),str) or d['id'].lower()!=repo.lower():raise ValueError('Repository identity mismatch')
    rows=[];seen=set()
    for s in d.get('siblings',[]):
        name=s.get('rfilename');lfs=s.get('lfs')
        if not isinstance(name,str) or not isinstance(lfs,dict):continue
        if not name.lower().endswith('.gguf') or '/' in name:continue
        if re.search(r'-\d{5}-of-\d{5}\.gguf$',name,re.I):continue
        if s.get('size')!=lfs.get('size'):raise ValueError('Inconsistent model metadata size: '+name)
        try:r=model_cache.identity(repo,sha,name,lfs.get('sha256'),lfs.get('size'))
        except (TypeError,ValueError):continue
        if name in seen:raise ValueError('Duplicate model filename')
        seen.add(name);rows.append(r)
    return {'repo':repo,'requested_ref':revision,'revision':sha,'files':sorted(rows,key=lambda r:(r['bytes'],r['filename'])),'scope':'Public repository metadata; model bytes not downloaded or verified'}

def packet(source):
    expected=model_cache.identity(source['repo'],source['revision'],source['filename'],source['sha256'],source['bytes'])
    if expected!=source:raise ValueError('Invalid model source descriptor')
    body={'schema':SCHEMA,'source':source}
    return {'lock_id':digest(body),**body}

def read_lock(path):
    d=read_json(path)
    if d!=packet(d['source']):raise ValueError('Model source lock integrity failure')
    return d

def fetch(reg,repo,filename,revision,sha,size,cache,execute=False,lock=None,lock_out=None,max_bytes=None):
    if lock:
        if any(x is not None for x in (repo,filename,revision,sha,size)):raise ValueError('Use --lock alone or repository and filename')
        p=read_lock(lock)
        limit=model_cache.MAX_BYTES if max_bytes is None else max_bytes
    elif repo and filename:
        if (sha is None)!=(size is None):raise ValueError('Specify SHA-256 and byte count together')
        if sha is not None:
            p=packet(model_cache.identity(repo,revision or '',filename,sha,size))
            limit=model_cache.MAX_BYTES if max_bytes is None else max_bytes
        else:
            d=metadata(repo,revision or 'main')
            matches=[r for r in d['files'] if r['filename']==filename]
            if len(matches)!=1:raise ValueError('No eligible single-file GGUF; use model-list to inspect available files')
            p=packet(matches[0])
            limit=DEFAULT_LIMIT if max_bytes is None else max_bytes
    else:raise ValueError('Specify repository and filename or --lock')
    if type(limit) is not int or not 24<=limit<=model_cache.MAX_BYTES:raise ValueError('Invalid download cap')
    src=p['source']
    if src['bytes']>limit:raise ValueError('Model exceeds download cap; set --max-size explicitly')
    if lock_out:
        path=Path(lock_out).expanduser()
        if path.exists():
            if read_lock(path)!=p:raise ValueError('Existing source lock differs')
        else:evaluation.write_new(path,p)
    r=model_cache.fetch(reg,src['repo'],src['revision'],src['filename'],src['sha256'],src['bytes'],cache,execute)
    return {**r,'source_lock':p,'download_cap_bytes':limit,'lock_path':str(Path(lock_out).expanduser()) if lock_out else None}

def cached(path,cache):
    p=read_lock(path)['source']
    r=model_cache.lookup(cache,p['sha256'])
    if r['bytes']!=p['bytes']:raise ValueError('Model cache and source lock size mismatch')
    return r['path']

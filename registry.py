import hashlib
import json
import os
import re
import sqlite3
import stat
import struct
from datetime import datetime, timezone
from pathlib import Path

SCHEMA='ramgpt-registry-v1'
CONTRACT='ramgpt-experiment-contract-v1'
ROLES=('model','runtime','dataset','template','evaluator','configuration','runtime_dependency')

def now():
    return datetime.now(timezone.utc).isoformat()

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def unique_pairs(pairs):
    result={}
    for k,v in pairs:
        if k in result:raise ValueError('Duplicate JSON key: '+k)
        result[k]=v
    return result

def reject_constant(value):
    raise ValueError('Non-finite JSON value: '+value)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),object_pairs_hook=unique_pairs,parse_constant=reject_constant)

def file_stamp(s):
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)

def fingerprint(path,role):
    if role not in ROLES:raise ValueError('Unsupported input role')
    p=Path(path).expanduser().resolve(strict=True)
    h=hashlib.sha256()
    fd=os.open(p,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):raise ValueError('Not a regular file')
        if before.st_size<=0:raise ValueError('Empty input file')
        header=stream.read(24)
        if role=='model':
            if len(header)<24 or header[:4]!=b'GGUF' or struct.unpack('<I',header[4:8])[0] not in (2,3):
                raise ValueError('Expected little-endian GGUF v2/v3')
            if re.search(r'-\d{5}-of-\d{5}\.gguf$',p.name,re.I):
                raise ValueError('Sharded GGUF is not supported by this prototype')
            if struct.unpack('<Q',header[8:16])[0]==0:raise ValueError('GGUF has no declared tensors')
        h.update(header)
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        after=os.fstat(stream.fileno())
        if file_stamp(before)!=file_stamp(after) or file_stamp(after)!=file_stamp(p.stat()):
            raise ValueError('Input changed during hashing: '+p.name)
    return {'role':role,'path':str(p),'sha256':h.hexdigest(),'bytes':before.st_size}

class Registry:
    def __init__(self,path):
        p=Path(path).expanduser()
        p.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(p,timeout=15)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS artifacts(sha256 TEXT PRIMARY KEY,bytes INTEGER NOT NULL,first_seen TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS observations(sha256 TEXT NOT NULL REFERENCES artifacts(sha256),role TEXT NOT NULL,path TEXT NOT NULL,verified_at TEXT NOT NULL,PRIMARY KEY(sha256,role,path));
        CREATE TABLE IF NOT EXISTS contracts(id TEXT PRIMARY KEY,body TEXT NOT NULL,created_at TEXT NOT NULL);
        """)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)',('schema',SCHEMA))
            if self.db.execute('SELECT value FROM metadata WHERE key=?',('schema',)).fetchone()[0]!=SCHEMA:
                raise ValueError('Unsupported registry schema')
    def close(self):
        self.db.close()
    def remember(self,r):
        self.db.execute('INSERT OR IGNORE INTO artifacts VALUES (?,?,?)',(r['sha256'],r['bytes'],now()))
        size=self.db.execute('SELECT bytes FROM artifacts WHERE sha256=?',(r['sha256'],)).fetchone()[0]
        if size!=r['bytes']:raise ValueError('Conflicting artifact size')
        self.db.execute('INSERT INTO observations VALUES (?,?,?,?) ON CONFLICT(sha256,role,path) DO UPDATE SET verified_at=excluded.verified_at',
                        (r['sha256'],r['role'],r['path'],now()))
    def register(self,path,role):
        r=fingerprint(path,role)
        with self.db:self.remember(r)
        return r
    def get(self,h):
        if not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h):raise ValueError('Invalid SHA-256')
        row=self.db.execute('SELECT bytes,first_seen FROM artifacts WHERE sha256=?',(h,)).fetchone()
        if row is None:raise ValueError('Unknown artifact')
        return {'sha256':h,'bytes':row[0],'first_seen':row[1],'observations':[dict(zip(('role','path','verified_at'),v)) for v in self.db.execute('SELECT role,path,verified_at FROM observations WHERE sha256=? ORDER BY role,path',(h,))]}
    def freeze(self,spec):
        if not isinstance(spec,dict) or set(spec)!={'inputs','execution','workload'}:raise ValueError('Expected inputs, execution, workload')
        inputs=spec['inputs']
        if not isinstance(inputs,list) or not inputs:raise ValueError('Missing inputs')
        if any(not isinstance(x,dict) or set(x)!={'role','path'} or not isinstance(x['path'],str) for x in inputs):
            raise ValueError('Expected role/path input objects')
        roles=[x['role'] for x in inputs]
        if any(roles.count(r)!=1 for r in ('model','runtime','dataset','template','evaluator')):raise ValueError('Require exactly one model, runtime, dataset, template and evaluator')
        execution=spec['execution'];workload=spec['workload']
        if not isinstance(execution,dict) or not {'runtime_args','gpu_uuids','sampling','reasoning_policy'}<=execution.keys():
            raise ValueError('Missing execution settings')
        if not isinstance(execution['runtime_args'],list) or not all(isinstance(x,str) for x in execution['runtime_args']):
            raise ValueError('Runtime arguments must be strings')
        devices=execution['gpu_uuids']
        if not isinstance(devices,list) or len(devices)!=len(set(devices)) or not all(isinstance(x,str) and re.fullmatch('GPU-[0-9a-fA-F-]{36}',x) for x in devices):
            raise ValueError('Invalid GPU UUIDs')
        if not isinstance(execution['sampling'],dict) or execution['reasoning_policy'] not in ('off','on','unknown'):
            raise ValueError('Invalid sampling or reasoning policy')
        if not isinstance(workload,dict) or not {'protocol','seed','budgets'}<=workload.keys() or not isinstance(workload['protocol'],str) or not workload['protocol'].strip() or type(workload['seed']) is not int:
            raise ValueError('Missing workload protocol or seed')
        caps=workload['budgets']
        if not isinstance(caps,list) or not caps or any(type(x) is not int or x<=0 for x in caps) or len(set(caps))!=len(caps):
            raise ValueError('Invalid workload budgets')
        records=[fingerprint(x['path'],x['role']) for x in inputs]
        records.sort(key=lambda x:(x['role'],x['path']))
        if len({(x['role'],x['path']) for x in records})!=len(records):raise ValueError('Duplicate input')
        body={'schema':CONTRACT,'inputs':records,'execution':execution,'workload':workload,'status':'planned_not_executed'}
        encoded=canonical(body).decode();cid=digest(body)
        with self.db:
            for r in records:self.remember(r)
            self.db.execute('INSERT OR IGNORE INTO contracts VALUES (?,?,?)',(cid,encoded,now()))
            if self.db.execute('SELECT body FROM contracts WHERE id=?',(cid,)).fetchone()[0]!=encoded:raise ValueError('Contract conflict')
        return {'contract_id':cid,**body}
    def contract(self,cid):
        row=self.db.execute('SELECT body FROM contracts WHERE id=?',(cid,)).fetchone()
        if row is None:raise ValueError('Unknown contract')
        body=json.loads(row[0])
        if digest(body)!=cid or body.get('schema')!=CONTRACT:raise ValueError('Contract integrity failure')
        return {'contract_id':cid,**body}
    def verify(self,cid):
        plan=self.contract(cid)
        for expected in plan['inputs']:
            actual=fingerprint(expected['path'],expected['role'])
            if actual!=expected:raise ValueError('Frozen input mismatch: '+expected['role'])
        return {'contract_id':cid,'status':'inputs_verified','inputs':len(plan['inputs']),'execution_verified':False}

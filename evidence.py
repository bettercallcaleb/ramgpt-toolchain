import fcntl
import hashlib
import json
import sqlite3
from pathlib import Path
from registry import canonical,digest,fingerprint,read_json,now
import inference
import evaluation

SCHEMA='ramgpt-evidence-store-v1'
SOURCE='ramgpt-source-evidence-v1'
DERIVED='ramgpt-derived-evidence-v1'

def checked_id(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected lowercase SHA-256 identity')
    return value

def adapter_hash():
    return fingerprint(Path(evaluation.__file__).with_name('evaluator_worker.py'),'configuration')['sha256']

def snapshot(reg,cid,folder):
    plan=reg.contract(cid)
    reg.verify(cid)
    refs={r['role']:r for r in plan['inputs'] if r['role'] in ('dataset','model','evaluator','runtime','configuration')}
    rows=inference.schedule(plan)
    if plan["workload"].get("evaluator")=="BudgetFit 0.6.0 Router":
        import adaptive
        adaptive.validate_schedule(plan,rows)
    else:evaluation.validate_schedule(plan,rows)
    p=Path(folder)
    with (p/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta={'schema':inference.RUN,'contract_id':cid,'schedule_digest':digest(rows),'planned':len(rows)}
        if read_json(p/'run.json')!=meta:raise ValueError('Run contract mismatch')
        sessions=inference.capture_sessions(p)
        completed,identity,_=inference.validate_sessions(sessions,rows,cid)
        if len(completed)!=len(rows):raise ValueError('Run incomplete')
        manifest=read_json(refs['dataset']['path'])
        reg.verify(cid)
        return {'schema':SOURCE,'contract':plan,'run':meta,'schedule':rows,'manifest':manifest,'sessions':sessions,
                'artifact_sha256':refs['model']['sha256'],'response_identity':identity,
                'scope':'historical committed execution; no claim of fresh measurement'}

def validate_source(source):
    if source['schema']!=SOURCE:raise ValueError('Unknown source schema')
    plan=source['contract'];cid=plan['contract_id']
    if digest({k:v for k,v in plan.items() if k!='contract_id'})!=cid:raise ValueError('Source contract hash mismatch')
    refs={r['role']:r for r in plan['inputs'] if r['role'] in ('model','runtime','dataset','evaluator','configuration')}
    if source['artifact_sha256']!=refs['model']['sha256']:raise ValueError('Source artifact mismatch')
    rows=source['schedule'];meta=source['run']
    if meta!={'schema':inference.RUN,'contract_id':cid,'schedule_digest':digest(rows),'planned':len(rows)}:
        raise ValueError('Source schedule mismatch')
    if len(rows)!=plan['workload']['request_count']:raise ValueError('Source request count mismatch')
    completed,identity,_=inference.validate_sessions(source['sessions'],rows,cid)
    if len(completed)!=len(rows) or not completed or identity!=source['response_identity']:raise ValueError('Incomplete source evidence')
    return completed

def native_report(source,archive):
    if source['contract']['workload'].get('evaluator')=='BudgetFit 0.6.0 Router':
        import adaptive
        return adaptive.native_report(source,archive)
    rows=validate_source(source);plan=source['contract'];work=plan['workload']
    refs={r['role']:r for r in plan['inputs'] if r['role'] in ('dataset','model','evaluator','runtime')}
    if fingerprint(archive,'evaluator')['sha256']!=refs['evaluator']['sha256']:raise ValueError('Evaluator archive mismatch')
    result=evaluation.worker(archive,{'action':'score','manifest':source['manifest'],'rows':rows,'budgets':work['budgets'],
                                  'strategies':work['strategies'],'reserve':work.get('reserve')})
    return {'schema':evaluation.SCHEMA,'contract_id':plan['contract_id'],'artifact_sha256':refs['model']['sha256'],
            'runtime_sha256':refs['runtime']['sha256'],'dataset_file_sha256':refs['dataset']['sha256'],
            'evaluator_sha256':refs['evaluator']['sha256'],'adapter_sha256':adapter_hash(),
            'schedule_digest':source['run']['schedule_digest'],'session_seals':[digest(s['seal']) for s in source['sessions']],
            'response_identity':source['response_identity'],'requests':len(rows),**result}

def derived(source,report):
    return {'schema':DERIVED,'source_id':digest(source),'report':report}

class Store:
    def __init__(self,path):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(p,timeout=15)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS blobs(id TEXT PRIMARY KEY,body BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS sources(id TEXT PRIMARY KEY REFERENCES blobs(id),contract_id TEXT NOT NULL,artifact_sha256 TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS evaluations(id TEXT PRIMARY KEY REFERENCES blobs(id),source_id TEXT NOT NULL REFERENCES sources(id),evaluator_sha256 TEXT NOT NULL,adapter_sha256 TEXT NOT NULL,created_at TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS source_contract ON sources(contract_id);
        CREATE INDEX IF NOT EXISTS source_artifact ON sources(artifact_sha256);
        """)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO meta VALUES (?,?)',('schema',SCHEMA))
            if self.db.execute('SELECT value FROM meta WHERE key=?',('schema',)).fetchone()[0]!=SCHEMA:raise ValueError('Unsupported evidence database')
    def close(self):self.db.close()
    def put_blob(self,body):
        raw=canonical(body);bid=hashlib.sha256(raw).hexdigest()
        self.db.execute('INSERT OR IGNORE INTO blobs VALUES (?,?)',(bid,raw))
        if self.db.execute('SELECT body FROM blobs WHERE id=?',(bid,)).fetchone()[0]!=raw:raise ValueError('Content collision or corrupt blob')
        return bid
    def get(self,bid):
        checked_id(bid)
        r=self.db.execute('SELECT body FROM blobs WHERE id=?',(bid,)).fetchone()
        if r is None:raise ValueError('Unknown evidence ID')
        raw=r[0]
        if hashlib.sha256(raw).hexdigest()!=bid:raise ValueError('Evidence blob hash mismatch')
        body=json.loads(raw)
        if canonical(body)!=raw:raise ValueError('Noncanonical evidence')
        return body
    def ingest(self,reg,cid,folder,report_path=None):
        source=snapshot(reg,cid,folder)
        archive=next(r['path'] for r in source['contract']['inputs'] if r['role']=='evaluator')
        if source["contract"].get("workload",{}).get("evaluator")=="BudgetFit 0.6.0 Router":
            import adaptive
            adaptive.check_binding(self,source["contract"])
            if self.db.execute("SELECT run_path FROM router_decisions WHERE id=?",(source["contract"]["workload"]["router_decision_id"],)).fetchone()!=(str(Path(folder).resolve()),):raise ValueError("Router run path mismatch")
        if 'joint_decision_id' in source['contract'].get('workload',{}):
            import joint
            joint.check_binding(self,source['contract'],folder)
        report=native_report(source,archive)
        if report_path is not None:
            prior=read_json(report_path)
            if {k:v for k,v in prior.items() if k!='created_at'}!=report:raise ValueError('Supplied report differs from independently replayed native evaluation')
        reg.verify(cid)
        d=derived(source,report)
        with self.db:
            sid=self.put_blob(source);eid=self.put_blob(d)
            self.db.execute('INSERT OR IGNORE INTO sources VALUES (?,?,?,?)',(sid,cid,source['artifact_sha256'],now()))
            self.db.execute('INSERT OR IGNORE INTO evaluations VALUES (?,?,?,?,?)',(eid,sid,report['evaluator_sha256'],report['adapter_sha256'],now()))
            if self.db.execute('SELECT contract_id,artifact_sha256 FROM sources WHERE id=?',(sid,)).fetchone()!=(cid,source['artifact_sha256']):raise ValueError('Source index mismatch')
            if self.db.execute('SELECT source_id,evaluator_sha256,adapter_sha256 FROM evaluations WHERE id=?',(eid,)).fetchone()!=(sid,report['evaluator_sha256'],report['adapter_sha256']):raise ValueError('Evaluation index mismatch')
        return {'status':'archived','source_id':sid,'evaluation_id':eid,'contract_id':cid,'artifact_sha256':source['artifact_sha256'],'requests':report['requests'],'new_inference_requests':0}
    def find(self,artifact=None,contract=None):
        clauses=[];args=[]
        if artifact is not None:clauses.append('s.artifact_sha256=?');args.append(checked_id(artifact))
        if contract is not None:clauses.append('s.contract_id=?');args.append(checked_id(contract))
        query='SELECT s.id,s.contract_id,s.artifact_sha256,e.id,e.evaluator_sha256,e.adapter_sha256 FROM sources s JOIN evaluations e ON e.source_id=s.id'
        if clauses:query+=' WHERE '+' AND '.join(clauses)
        result=[]
        for sid,cid,aid,eid,ev,ad in self.db.execute(query+' ORDER BY s.id,e.id',args):
            s=self.get(sid);d=self.get(eid)
            if (s['contract']['contract_id'],s['artifact_sha256'],d['source_id'],d['report']['evaluator_sha256'],d['report']['adapter_sha256'])!=(cid,aid,sid,ev,ad):raise ValueError('Evidence index mismatch')
            result.append({'source_id':sid,'contract_id':cid,'artifact_sha256':aid,'evaluation_id':eid,'evaluator_sha256':ev,'adapter_sha256':ad,'requests':d['report']['requests']})
        return result
    def replay(self,reg,cid,source_id=None):
        reg.verify(cid);plan=reg.contract(cid)
        candidates=self.find(contract=cid)
        if source_id is not None:candidates=[x for x in candidates if x['source_id']==checked_id(source_id)]
        evaluator=next(r for r in plan['inputs'] if r['role']=='evaluator')
        expected_adapter=adapter_hash()
        if plan.get('workload',{}).get('evaluator')=='BudgetFit 0.6.0 Router':
            import adaptive
            expected_adapter=adaptive.adapter_hash()
            adaptive.check_binding(self,plan)
        candidates=[x for x in candidates if x['evaluator_sha256']==evaluator['sha256'] and x['adapter_sha256']==expected_adapter]
        if not candidates:raise ValueError('No exact-contract evaluation available')
        if len(candidates)!=1:raise ValueError('Multiple historical sources; specify source_id explicitly')
        hit=candidates[0];source=self.get(hit['source_id']);d=self.get(hit['evaluation_id'])
        if source['contract']!=plan:raise ValueError('Frozen contract mismatch')
        validate_source(source)
        if d!=derived(source,native_report(source,evaluator['path'])):raise ValueError('Derived report replay mismatch')
        reg.verify(cid)
        return {'status':'historical_replay','new_inference_requests':0,'source_id':hit['source_id'],'evaluation_id':hit['evaluation_id'],
                'note':'Historical observations only; timing and GPU evidence are not fresh. Not a deployment validation.',
                'report':d['report']}

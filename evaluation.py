import fcntl
import json
import subprocess
import sys
from pathlib import Path
from registry import fingerprint,read_json,digest,now
import inference
import runtime

SCHEMA='ramgpt-evaluation-v1'

def worker(archive,data):
    p=Path(archive).resolve(strict=True)
    if p.suffix!='.pyz':raise ValueError('Expected trusted BudgetFit 0.6.0 zip application')
    script=Path(__file__).with_name('evaluator_worker.py').resolve()
    before=fingerprint(p,'evaluator')
    script_before=fingerprint(script,'configuration')
    r=subprocess.run([sys.executable,'-I',str(script),str(p)],input=json.dumps(data,allow_nan=False),text=True,capture_output=True,timeout=180)
    if r.returncode:raise ValueError('Evaluator failed: '+r.stderr[-4000:])
    if fingerprint(p,'evaluator')!=before or fingerprint(script,'configuration')!=script_before:raise ValueError('Evaluator or adapter changed')
    return json.loads(r.stdout)

def write_new(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def prepare(dataset,archive,budgets,strategies,reserve,out):
    result=worker(archive,{'action':'prepare','manifest':read_json(dataset),'budgets':budgets,'strategies':strategies,'reserve':reserve})
    write_new(out,result)
    return {'status':'schedule_prepared_not_frozen','requests':len(result['requests']),'schedule':str(out)}

def validate_schedule(plan,rows):
    refs={r['role']:r for r in plan['inputs'] if r['role'] in ('dataset','evaluator')}
    for role in refs:
        if fingerprint(refs[role]['path'],role)!=refs[role]:raise ValueError('Frozen evaluator input changed')
    workload=plan['workload']
    expected=worker(refs['evaluator']['path'],{'action':'prepare','manifest':read_json(refs['dataset']['path']),'budgets':workload['budgets'],'strategies':workload['strategies'],'reserve':workload.get('reserve')})['requests']
    for role in refs:
        if fingerprint(refs[role]['path'],role)!=refs[role]:raise ValueError('Frozen evaluator input changed')
    def key(r):return (r['task_id'],r['max_tokens'],r['strategy'])
    lookup={key(r):r['prompt'] for r in expected}
    if len(rows)!=len(lookup) or len({key(r) for r in rows})!=len(rows) or any(lookup.get(key(r))!=r['prompt'] for r in rows):
        raise ValueError('Schedule differs from native evaluator prompt matrix')
    return True

def score(reg,cid,folder,out):
    plan=reg.contract(cid)
    reg.verify(cid)
    if plan['workload'].get('evaluator')!='BudgetFit 0.6.0':raise ValueError('Unsupported frozen evaluator')
    refs={r['role']:r for r in plan['inputs'] if r['role'] in ('dataset','evaluator','model','runtime')}
    rows=inference.schedule(plan)
    p=Path(folder)
    with (p/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta={'schema':inference.RUN,'contract_id':cid,'schedule_digest':digest(rows),'planned':len(rows)}
        if read_json(p/'run.json')!=meta:raise ValueError('Run contract mismatch')
        committed,identity,num=inference.read_committed(p,rows,cid)
        if len(committed)!=len(rows):raise ValueError('Run incomplete')
        data={'action':'score','manifest':read_json(refs['dataset']['path']),'rows':committed,
              'budgets':plan['workload']['budgets'],'strategies':plan['workload']['strategies'],'reserve':plan['workload'].get('reserve')}
        for role in ('dataset','evaluator'):
            if fingerprint(refs[role]['path'],role)!=refs[role]:raise ValueError('Frozen evaluator input changed')
        result=worker(refs['evaluator']['path'],data)
        for role in ('dataset','evaluator'):
            if fingerprint(refs[role]['path'],role)!=refs[role]:raise ValueError('Frozen evaluator input changed')
        seals=[digest(read_json(x/'commit.json')) for x in sorted(p.glob('session-*'))]
        report={'schema':SCHEMA,'contract_id':cid,'artifact_sha256':refs['model']['sha256'],'runtime_sha256':refs['runtime']['sha256'],
                'dataset_file_sha256':refs['dataset']['sha256'],'evaluator_sha256':refs['evaluator']['sha256'],
                'adapter_sha256':fingerprint(Path(__file__).with_name('evaluator_worker.py'),'configuration')['sha256'],
                'schedule_digest':meta['schedule_digest'],'session_seals':seals,'response_identity':identity,
                'requests':len(committed),'created_at':now(),**result}
        write_new(out,report)
    return {'status':'scored','requests':len(committed),'profiles':[r['budgetfit'] for r in result['profiles']],'report':str(out)}

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA='ramgpt-toolchain-evidence-v0.1'

def load(path):
    return json.loads(Path(path).read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('Invalid SHA-256')
    return value

def record(producer,kind,subject,reference,scope,metrics,sources,status):
    return {'producer':producer,'kind':kind,'artifact_sha256':identity(subject) if subject else None,
            'reference_sha256':identity(reference) if reference else None,'scope':scope,'metrics':metrics,
            'source_sha256':{Path(p).name:digest(p) for p in sources},'source_status':status}

def quantbench(path):
    d=load(path)
    if d['schema']!='ramgpt-quantbench-verified-open-artifact-result' or d['status']!='PASS':raise ValueError('Unsupported QuantBench result')
    return [record('quantbench','fidelity',r['artifact_sha256'],d['reference_sha256'],
                   {'benchmark':d['benchmark_id'],'contract':d['execution_contract'],'model':d['model']},
                   {'mean_kl':r['mean_kl_bf16_to_artifact'],'bytes':r['artifact_bytes']},[path],d['status']) for r in d['rankings']]

def q4fit(path):
    d=load(path)
    if d['schema']!='ramgpt-experiment-v1':raise ValueError('Unsupported Q4Fit evidence')
    return [record('q4fit','knowledge_depth',d['artifacts']['candidate']['sha256'],d['artifacts']['reference']['sha256'],
                   d['evaluation'],d['results'],[path],d['status'])]

def budgetfit(folder):
    p=Path(folder);report=p/'router_report.json';plan=p/'router_plan.json';tasks=p/'validation_tasks.json'
    d=load(report);z=load(plan);t=load(tasks)
    if d['status'] not in ('validation passed','validation failed; no router recommendation'):raise ValueError('Incomplete router report')
    if d['model']!=z['model'] or d['version']!=z['version']:raise ValueError('Router plan mismatch')
    v=d['validation']
    if len(t)!=v['router']['n'] or len(t)!=v['fixed']['n'] or not v['router']['complete'] or not v['fixed']['complete']:raise ValueError('Incomplete validation')
    if bool(d['deployment'])!=(d['status']=='validation passed'):raise ValueError('Deployment mismatch')
    return [record('budgetfit','inference_optimization',None,None,
                   {'model_alias':d['model'],'engine':z['engine'],'validation_hash':z['validation_hash'],'calibration_hash':z['calibration_hash'],'sampling':z['sampling'],'thresholds':z['thresholds'],'identity_note':'Historical report lacks artifact hash bound to run.'},
                   v,[report,plan,tasks],d['status'])]

def trustmebro(folder):
    p=Path(folder);plan=p/'plan.json';raw=p/'raw.json'
    d=load(plan);rows=load(raw)
    if len(rows)!=d['planned_requests'] or [(r['id'],r['budget'],r['arm']) for r in rows]!=[tuple(x) for x in d['schedule']]:raise ValueError('Incomplete or misaligned schedule')
    return [record('trustmebro','behavior_trace',d['artifact']['sha256'],None,
                   {'model':d['model'],'manifest_sha256':d['manifest_sha256'],'sampling':d['sampling'],'reasoning_policy':d['reasoning_policy'],'runtime_sha256':d['engine_artifact']['sha256']},
                   {'requests':len(rows)},[plan,raw],'completed source trace; scores not independently replayed')]

def assemble(records):
    groups={};unbound=[]
    for r in records:
        h=r['artifact_sha256']
        if h is None:unbound.append(r)
        else:groups.setdefault(h,[]).append(r)
    return {'schema':SCHEMA,'artifact_groups':groups,'unbound_evidence':unbound,
            'cross_tool_matches':[h for h,g in groups.items() if len({r['producer'] for r in g})>1],
            'note':'Matching hashes link artifacts only, not equivalent datasets, runtimes or scores. No composite ranking.'}

def main():
    p=argparse.ArgumentParser()
    for name in ('quantbench','q4fit','budgetfit','trustmebro'):p.add_argument('--'+name,action='append',default=[])
    p.add_argument('--out',required=True)
    a=p.parse_args()
    rows=[]
    for name,fn in [('quantbench',quantbench),('q4fit',q4fit),('budgetfit',budgetfit),('trustmebro',trustmebro)]:
        for path in getattr(a,name):rows.extend(fn(path))
    report=assemble(rows)
    Path(a.out).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('records',len(rows),'artifact groups',len(report['artifact_groups']),'unbound',len(report['unbound_evidence']),'cross-tool matches',len(report['cross_tool_matches']))

if __name__=='__main__':main()

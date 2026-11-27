import fcntl
import math
import os
import re
from pathlib import Path
from registry import digest,read_json,now,fingerprint
import runtime

SCHEDULE='ramgpt-request-schedule-v1'
RUN='ramgpt-inference-run-v1'

def schedule(plan):
    refs=[r for r in plan['inputs'] if r['role']=='configuration']
    if len(refs)!=1:raise ValueError('Require exactly one frozen schedule as configuration input')
    ref=refs[0]
    if fingerprint(ref['path'],'configuration')!=ref:raise ValueError('Schedule bytes changed')
    d=read_json(ref['path'])
    if set(d)!={'schema','requests'} or d['schema']!=SCHEDULE:raise ValueError('Unsupported schedule')
    if plan['workload']['protocol']!='toolchain-inference-v1':raise ValueError('Unsupported workload')
    _,_,_,alias,_=runtime.validate(plan)
    sampling=plan['execution']['sampling']
    rows=d['requests']
    if not isinstance(rows,list) or not rows:raise ValueError('Empty schedule')
    keys=set()
    dataset=next(r for r in plan['inputs'] if r['role']=='dataset')
    if fingerprint(dataset['path'],'dataset')!=dataset:raise ValueError('Dataset bytes changed')
    tasks=read_json(dataset['path'])['tasks']
    ids={t['id'] for t in tasks}
    if len(ids)!=len(tasks):raise ValueError('Duplicate dataset IDs')
    if plan['workload'].get('task_count')!=len(tasks):raise ValueError('Dataset count mismatch')
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'request_id','task_id','strategy','prompt','max_tokens'}:raise ValueError('Invalid request fields')
        if not all(isinstance(row[k],str) and row[k].strip() for k in ('request_id','task_id','strategy','prompt')):raise ValueError('Invalid request text')
        if row['request_id'] in keys:raise ValueError('Duplicate request ID')
        keys.add(row['request_id'])
        if row['task_id'] not in ids or row['strategy'] not in plan['workload'].get('strategies',[]):raise ValueError('Unknown task or strategy')
        if type(row['max_tokens']) is not int or row['max_tokens'] not in plan['workload']['budgets']:raise ValueError('Unfrozen token budget')
    if len(rows)!=plan['workload'].get('request_count'):raise ValueError('Schedule count mismatch')
    return [{**r,'body':{'model':alias,'messages':[{'role':'user','content':r['prompt']}],'max_tokens':r['max_tokens'],**sampling,'seed':plan['workload']['seed'],'cache_prompt':False}} for r in rows]

def check_response(raw,expected,identity):
    if not isinstance(raw,dict) or raw.get('model')!=expected['body']['model']:raise ValueError('Response model mismatch')
    for key,value in identity.items():
        if raw.get(key)!=value:raise ValueError('Backend identity changed: '+key)
    c=raw.get('choices')
    if not isinstance(c,list) or len(c)!=1:raise ValueError('Expected one response choice')
    m=c[0]['message'];finish=c[0].get('finish_reason')
    if finish not in ('stop','length') or not isinstance(m.get('content'),str):raise ValueError('Unsupported response content or termination')
    if m.get('tool_calls') or m.get('reasoning_content') or m.get('reasoning') or any(x in m['content'] for x in ('<think>','</think>')):raise ValueError('Unexpected reasoning or tool call')
    u=raw.get('usage',{})
    for key in ('prompt_tokens','completion_tokens','total_tokens'):
        if type(u.get(key)) is not int or u[key]<0:raise ValueError('Invalid usage: '+key)
    if u['completion_tokens']>expected['max_tokens'] or u['total_tokens']!=u['prompt_tokens']+u['completion_tokens']:raise ValueError('Inconsistent usage')
    return {'finish_reason':finish,'prompt_tokens':u['prompt_tokens'],'completion_tokens':u['completion_tokens'],'total_tokens':u['total_tokens']}

def validate_sessions(sessions,rows,cid,smoke_policy=None):
    completed=[];identity=None
    for i,session in enumerate(sessions):
        if session['name']!=f'session-{i:06d}':raise ValueError('Invalid session ordering')
        seal=session['seal'];receipt=session['receipt'];data=session['responses']
        if seal!={'contract_id':cid,'receipt_digest':digest(receipt),'responses_digest':digest(data),'start':len(completed),'count':len(data)}:raise ValueError('Session seal mismatch')
        if receipt.get('contract_id')!=cid or receipt.get('status')!='runtime_session_passed' or not all(receipt.get(x) is True for x in ('input_precheck','input_postcheck','smoke_pass','template_probe_match','cleanup_complete')):raise ValueError('Invalid runtime receipt')
        if smoke_policy=='transport-v2' and (receipt.get('smoke_policy')!='transport-v2' or type(receipt.get('literal_instruction_pass')) is not bool):raise ValueError('Missing transport smoke diagnostic')
        if smoke_policy=='literal-v1' and receipt.get('smoke_policy','literal-v1')!='literal-v1':raise ValueError('Unexpected smoke policy')
        ident=receipt['response_identity']
        if identity is not None and ident!=identity:raise ValueError('Session backend identity mismatch')
        identity=ident
        if not data or len(completed)+len(data)>len(rows):raise ValueError('Invalid response count')
        if receipt.get('workload_result')!={'start':len(completed),'count':len(data)}:raise ValueError('Session workload mismatch')
        for row in data:
            expected=rows[len(completed)]
            if row['request']!=expected or row.get('ordinal')!=len(completed) or row.get('contract_id')!=cid:raise ValueError('Saved request mismatch')
            if row['measurement']!=check_response(row['raw'],expected,identity):raise ValueError('Saved measurement mismatch')
            if not isinstance(row['elapsed_s'],(int,float)) or not math.isfinite(row['elapsed_s']) or row['elapsed_s']<0:raise ValueError('Invalid elapsed time')
            completed.append(row)
    return completed,identity,len(sessions)


def capture_sessions(out):
    result=[]
    for i,p in enumerate(sorted(out.glob('session-*'))):
        if p.name!=f'session-{i:06d}' or not p.is_dir():raise ValueError('Invalid session ordering')
        if not (p/'commit.json').is_file():raise ValueError('Uncommitted session; inspect preserved evidence before any retry')
        result.append({'name':p.name,'seal':read_json(p/'commit.json'),
                       'receipt':read_json(p/'runtime_receipt.json'),'responses':read_json(p/'responses.json')})
    return result

def read_committed(out,rows,cid):
    return validate_sessions(capture_sessions(out),rows,cid)

def run(reg,cid,out,max_new=None,timeout=120,evidence_db=None,measurement_authorized=False):
    if max_new is not None and (type(max_new) is not int or max_new<1):raise ValueError('Invalid max-new')
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('Invalid timeout')
    plan=reg.contract(cid)
    w=plan.get('workload',{})
    if 'measurement_cohort_key' in w and not ('joint_decision_id' in w or 'router_decision_id' in w) and (measurement_authorized is not True or max_new is not None):raise ValueError('Use measurement-run for bounded measurement contracts')
    if w.get('bounded_workflow') and (measurement_authorized is not True or max_new is not None):raise ValueError('Use workflow-finish for bounded holdout contracts')
    rows=schedule(plan)
    if plan.get('workload',{}).get('evaluator')=='BudgetFit 0.6.0':
        import evaluation
        evaluation.validate_schedule(plan,rows)
    elif plan.get('workload',{}).get('evaluator')=='BudgetFit 0.6.0 Router':
        import adaptive
        adaptive.validate_schedule(plan,rows)
        if evidence_db is None:raise ValueError('Router requires evidence database')
        adaptive.authorize(evidence_db,plan,out)
    if 'joint_decision_id' in plan.get('workload',{}):
        import joint
        if evidence_db is None:raise ValueError('Joint holdout requires evidence database')
        joint.authorize(evidence_db,plan,out)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        meta={'schema':RUN,'contract_id':cid,'schedule_digest':digest(rows),'planned':len(rows)}
        path=out/'run.json'
        if path.exists():
            if read_json(path)!=meta:raise ValueError('Run contract mismatch')
        else:
            if list(out.glob('session-*')):raise ValueError('Missing run metadata')
            runtime.atomic(path,meta)
        completed,identity,num=read_committed(out,rows,cid)
        start=len(completed)
        if start<len(rows):
            end=min(len(rows),start+max_new) if max_new is not None else len(rows)
            session=out/f'session-{num:06d}'
            result=[]
            def work(send,observed):
                if identity is not None and observed!=identity:raise ValueError('Backend identity changed before resume')
                if not observed.get('system_fingerprint'):raise ValueError('Backend fingerprint required for managed run')
                for i in range(start,end):
                    row=rows[i]
                    runtime.atomic(session/'pending.json',{'ordinal':i,'request':row,'started_at':now()})
                    try:
                        raw,elapsed=send(row['body'],timeout)
                        runtime.atomic(session/'last_raw.json',raw)
                        measurement=check_response(raw,row,observed)
                        result.append({'contract_id':cid,'ordinal':i,'request':row,'raw':raw,'measurement':measurement,'elapsed_s':elapsed})
                        runtime.atomic(session/'responses.json',result)
                        (session/'pending.json').unlink()
                    except BaseException as e:
                        runtime.atomic(session/'error.json',{'ordinal':i,'error':str(e),'at':now()})
                        raise
                return {'start':start,'count':len(result)}
            receipt=runtime.smoke(reg,cid,session,work=work)
            runtime.atomic(session/'commit.json',{'contract_id':cid,'receipt_digest':digest(receipt),'responses_digest':digest(result),'start':start,'count':len(result)})
            completed,identity,num=read_committed(out,rows,cid)
        else:reg.verify(cid)
        summary={'schema':RUN,'contract_id':cid,'status':'complete' if len(completed)==len(rows) else 'paused',
                 'planned':len(rows),'committed':len(completed),'new_requests':len(completed)-start,'sessions':num,
                 'completion_tokens':sum(r['measurement']['completion_tokens'] for r in completed),'response_identity':identity}
        runtime.atomic(out/'summary.json',summary)
        return summary

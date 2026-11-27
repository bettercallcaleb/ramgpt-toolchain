import fcntl
from pathlib import Path
from registry import digest,read_json
import evidence
import evaluation
import inference
import joint
import measurement_plan as mp
import passport
import planner
import runtime

SCHEMA='ramgpt-measurement-workflow-v1'

def init(store):
    store.db.execute('CREATE TABLE IF NOT EXISTS measurement_workflows(plan_id TEXT PRIMARY KEY REFERENCES measurement_plans(id), request_id TEXT NOT NULL REFERENCES blobs(id), result_id TEXT REFERENCES blobs(id))')
    store.db.execute('CREATE TABLE IF NOT EXISTS workflow_started(contract_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL REFERENCES measurement_workflows(plan_id))')

def calibration(reg,store,p):
    eids=[];lookup={}
    for arm,started,archived in mp.preflight(reg,store,p):
        if not started or len(archived)!=1:raise ValueError('Complete and archive every calibration arm first')
        hit=archived[0]
        if digest(evidence.snapshot(reg,arm['contract_id'],arm['run_path']))!=hit['source_id']:raise ValueError('Calibration archive mismatch')
        eids.append(hit['evaluation_id']);lookup[arm['kv']]=hit['evaluation_id']
    return eids,lookup

def holdout_cost(n,base,chosen):
    return {'workload_requests':2*n,'smoke_requests':2,'total_completion_requests':2*n+2,'workload_completion_token_cap':n*(base[0]+chosen[0]),'smoke_completion_token_cap':128,'total_completion_token_cap':n*(base[0]+chosen[0])+128,'server_sessions':2}

def select(reg,store,pid,policy,anchor_kv,fixed,seed,max_requests,max_tokens):
    joint.policy(policy)
    if type(seed) is not int or not 0<=seed<2**31:raise ValueError('Invalid holdout seed')
    if type(max_requests) is not int or not 1<=max_requests<=10000 or type(max_tokens) is not int or not 1<=max_tokens<=10**8:raise ValueError('Invalid holdout caps')
    p=mp.load(reg,store,pid);root=Path(p['root'])
    req={'schema':SCHEMA,'plan_id':pid,'policy':policy,'anchor_kv':anchor_kv,'anchor_fixed':fixed,'validation_seed':seed,'max_requests':max_requests,'max_completion_tokens':max_tokens}
    with (root/'.measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        init(store)
        prior=store.db.execute('SELECT request_id,result_id FROM measurement_workflows WHERE plan_id=?',(pid,)).fetchone()
        if prior:
            if prior[0]!=digest(req):raise ValueError('Workflow already frozen; no alternate policy or holdout')
            if not prior[1]:raise ValueError('Interrupted workflow preparation; inspect local state')
            result=store.get(prior[1])
            if result['plan_id']!=pid:raise ValueError('Workflow result mismatch')
            return {'workflow_id':prior[1],**result}
        eids,lookup=calibration(reg,store,p)
        if anchor_kv not in lookup:raise ValueError('Anchor KV not measured')
        preview=joint.select(store,eids,lookup[anchor_kv],fixed,policy)
        chosen=preview['selected'];base=preview['baseline']
        failed,_,_=planner.validation_history(store,p['artifact_sha256'])
        reason=None
        if chosen is None:reason='no_eligible_candidate'
        elif chosen['candidate_id']==base['candidate_id']:reason='baseline_selected'
        elif chosen['candidate_id'] in failed:reason='previous_heldout_failure'
        elif any(x['evaluation_id'] in eids for x in preview['excluded_evaluations']):reason='incomparable_calibration'
        else:
            key=joint.OBJECTIVES[policy['objective']]
            a=chosen['memory'][key] if policy['objective']=='memory' else chosen['metrics'][key]
            b=base['memory'][key] if policy['objective']=='memory' and base['memory'] else base['metrics'].get(key)
            if b is None or a>=b:reason='no_calibration_objective_improvement'
        c=None
        if reason is None:
            c=holdout_cost(base['metrics']['n'],base['fixed'],chosen['fixed'])
            mp.check_cost(c,max_requests,max_tokens)
        with store.db:
            rid=store.put_blob(req)
            store.db.execute('INSERT INTO measurement_workflows(plan_id,request_id) VALUES (?,?)',(pid,rid))
        d=joint.freeze(store,eids,lookup[anchor_kv],fixed,policy,seed)
        if d['result']!=preview:raise ValueError('Selection changed while freezing')
        h=None
        if reason is None:
            h=joint.prepare(reg,store,d['decision_id'],root/'holdout',bounded_workflow=True)
            if len(h['arms'])!=2:raise ValueError('Unexpected holdout arms')
        result={'schema':SCHEMA,'plan_id':pid,'request_id':rid,'decision_id':d['decision_id'],'holdout_id':h['holdout_id'] if h else None,'status':'holdout_frozen' if h else 'no_holdout','reason':reason,'cost':c,'selected_candidate_id':chosen['candidate_id'] if chosen else None,'deployment_authorized':False}
        with store.db:
            wid=store.put_blob(result)
            store.db.execute('UPDATE measurement_workflows SET result_id=? WHERE plan_id=?',(wid,pid))
        evaluation.write_new(root/'workflow.json',{'workflow_id':wid,**result})
        return {'workflow_id':wid,**result}

def load(reg,store,pid):
    p=mp.load(reg,store,pid);init(store)
    row=store.db.execute('SELECT request_id,result_id FROM measurement_workflows WHERE plan_id=?',(pid,)).fetchone()
    if not row or not row[1]:raise ValueError('No completed workflow preparation')
    req=store.get(row[0]);r=store.get(row[1]);root=Path(p['root'])
    if req['plan_id']!=pid or r['plan_id']!=pid or r['request_id']!=row[0] or read_json(root/'workflow.json')!={'workflow_id':row[1],**r}:raise ValueError('Workflow binding mismatch')
    d=joint.decision(store,r['decision_id'])
    if d['policy']!=req['policy'] or d['validation_seed']!=req['validation_seed'] or d['anchor_fixed']!=req['anchor_fixed']:raise ValueError('Frozen selection mismatch')
    eids,lookup=calibration(reg,store,p)
    if set(eids)!=set(d['evaluation_ids']) or lookup[req['anchor_kv']]!=d['anchor_eid']:raise ValueError('Calibration binding mismatch')
    h=None
    if r['holdout_id']:
        hid=store.db.execute('SELECT holdout_id FROM joint_decisions WHERE id=?',(r['decision_id'],)).fetchone()[0]
        if hid!=r['holdout_id']:raise ValueError('Holdout binding mismatch')
        h=store.get(hid);folder=root/'holdout'
        if read_json(folder/'holdout.json')!={'holdout_id':hid,**h}:raise ValueError('Holdout file mismatch')
        if h['decision_id']!=r['decision_id'] or {a['arm'] for a in h['arms']}!={'baseline','candidate'} or len(h['arms'])!=2:raise ValueError('Invalid holdout arms')
        totals=[]
        for a in h['arms']:
            cid=a['contract_id'];plan=reg.contract(cid);reg.verify(cid)
            if plan!=read_json(folder/(a['arm']+'-contract.json')) or a['run_path']!=str(folder/(a['arm']+'-run')):raise ValueError('Holdout contract mismatch')
            if plan['workload'].get('bounded_workflow') is not True:raise ValueError('Missing bounded workflow marker')
            joint.check_binding(store,plan,a['run_path'])
            rows=inference.schedule(plan);evaluation.validate_schedule(plan,rows)
            totals.append(rows)
        c=holdout_cost(len(totals[0]),[totals[0][0]['max_tokens']],[totals[1][0]['max_tokens']])
        if len(totals[0])!=len(totals[1]) or c!=r['cost'] or sum(x['max_tokens'] for rows in totals for x in rows)!=c['workload_completion_token_cap']:raise ValueError('Holdout cost mismatch')
        mp.check_cost(c,req['max_requests'],req['max_completion_tokens'])
    elif r['status']!='no_holdout':raise ValueError('Missing holdout')
    return p,r,h

def pending(reg,store,pid,h):
    rows=[]
    for a in h['arms']:
        cid=a['contract_id'];out=Path(a['run_path'])
        started=store.db.execute('SELECT plan_id FROM workflow_started WHERE contract_id=?',(cid,)).fetchone()
        if started and started!=(pid,):raise ValueError('Holdout start binding mismatch')
        archived=store.find(contract=cid)
        if out.exists():
            if out.resolve()!=out or not started:raise ValueError('Unregistered holdout run')
            schedule=inference.schedule(reg.contract(cid))
            complete,_,n=inference.validate_sessions(inference.capture_sessions(out),schedule,cid)
            if n!=1 or len(complete)!=len(schedule):raise ValueError('Partial holdout; inspect manually, no retry')
            if len(archived)>1:raise ValueError('Ambiguous holdout archive')
        elif started or archived:raise ValueError('Previously started holdout missing; no retry')
        rows.append((a,bool(started),archived))
    return rows

def finish(reg,store,pid,execute=False):
    if execute is not True:raise ValueError('Explicit --execute required for workflow finish')
    p,r,h=load(reg,store,pid);root=Path(p['root'])
    with (root/'.measurement.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p,r,h=load(reg,store,pid)
        store.db.execute('CREATE TABLE IF NOT EXISTS workflow_completed(plan_id TEXT PRIMARY KEY REFERENCES measurement_workflows(plan_id), result_id TEXT NOT NULL REFERENCES blobs(id), passport_blob_id TEXT NOT NULL REFERENCES blobs(id))')
        prior=store.db.execute('SELECT result_id,passport_blob_id FROM workflow_completed WHERE plan_id=?',(pid,)).fetchone()
        if prior:
            result=store.get(prior[0]);packet=store.get(prior[1]);passport.verify(packet)
            if result['plan_id']!=pid or result['passport_id']!=packet['passport_id'] or read_json(root/'model.passport.json')!=packet or read_json(root/'workflow-final.json')!={'result_id':prior[0],**result}:raise ValueError('Final workflow export mismatch')
            return {'result_id':prior[0],**result,'new_inference_requests':0,'new_smoke_requests':0}
        new=0;smokes=0;v=None
        if h:
            todo=pending(reg,store,pid,h)
            db=store.db.execute('PRAGMA database_list').fetchone()[2]
            eids=[]
            for a,started,archived in todo:
                cid=a['contract_id'];out=a['run_path']
                if not started:
                    with store.db:store.db.execute('INSERT INTO workflow_started VALUES (?,?)',(cid,pid))
                    summary=inference.run(reg,cid,out,evidence_db=db,measurement_authorized=True)
                    if summary['status']!='complete' or summary['sessions']!=1:raise ValueError('Expected one complete holdout session')
                    new+=summary['new_requests'];smokes+=1
                if not archived:hit=store.ingest(reg,cid,out)
                else:
                    hit=archived[0]
                    if digest(evidence.snapshot(reg,cid,out))!=hit['source_id']:raise ValueError('Holdout archive mismatch')
                eids.append(hit['evaluation_id'])
            v=joint.validate(store,r['decision_id'],eids)
        packet=passport.build(reg,store,p['artifact_sha256']);passport.verify(packet)
        result={'schema':SCHEMA,'plan_id':pid,'decision_id':r['decision_id'],'holdout_id':r['holdout_id'],'validation_id':v['validation_id'] if v else None,'status':v['status'] if v else 'no_holdout','reason':r['reason'],'failures':v['failures'] if v else [],'passport_id':packet['passport_id'],'deployment_authorized':False}
        for path,body in ((root/'model.passport.json',packet),(root/'workflow-final.json',{'result_id':digest(result),**result})):
            if path.exists():
                if read_json(path)!=body:raise ValueError('Conflicting workflow export')
            else:evaluation.write_new(path,body)
        with store.db:
            bid=store.put_blob(packet);rid=store.put_blob(result)
            store.db.execute('INSERT INTO workflow_completed VALUES (?,?,?)',(pid,rid,bid))
        return {'result_id':rid,**result,'new_inference_requests':new,'new_smoke_requests':smokes}

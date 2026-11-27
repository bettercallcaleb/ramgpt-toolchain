import json
import sys

sys.path.insert(0,sys.argv[1])
import budgetfit as b
import router as r
import calibration as c
import measurement as m

if b.VERSION!='0.6.0' or b.ENGINE!='e362d6bed40cdab94c07f489e17f779eec2124a6' or not all(str(x.__file__).startswith(sys.argv[1]+'/') for x in (b,r,c,m)):
    raise ValueError('Unexpected native engine')

def train(tasks,configs,rows,t):
    policy,fixed,profiles=r.train(tasks,[tuple(x) for x in configs],rows,t['min_accuracy'],t['tolerance'],t['min_bucket'],t['min_savings'])
    return {'policy':policy,'fixed':fixed,'profiles':profiles}

def expected(tasks,selection):
    policy=selection['policy']
    fixed=c.config_key(selection['fixed'])
    out=[]
    for task in tasks:
        for cap,strategy,reserve in sorted({fixed,r.choose(policy,task['prompt'])}):
            prompt,_=b.prompt(task,strategy,cap,reserve)
            out.append({'request_id':f'router-{len(out):06d}','task_id':task['id'],'strategy':strategy,'prompt':prompt,'max_tokens':cap})
    return out

def observed(tasks,selection,rows,t):
    fixed=c.config_key(selection['fixed'])
    keys={(x['id'],*c.config_key(x)) for x in rows}
    needed={(x['id'],*cfg) for x in tasks for cfg in {fixed,r.choose(selection['policy'],x['prompt'])}}
    if len(keys)!=len(rows) or keys!=needed:raise ValueError('Invalid sparse pair coverage')
    v=r.evaluate(tasks,rows,selection['policy'],fixed)
    return {'validation':v,'passed':r.gate(v,selection['policy'],t['min_accuracy'],t['tolerance'],t['max_validation_drop'],t['min_savings'],t['min_bucket'])}

def main(d):
    a=d['action']
    if a=='historical':
        s=train(d['calibration']['tasks'],d['configs'],d['cal_rows'],d['thresholds'])
        return {**s,**observed(d['tasks'],s,d['rows'],d['thresholds'])}
    manifest=d['manifest'];m.verify(manifest);tasks=manifest['tasks']
    if a=='train':return train(tasks,d['configs'],d['rows'],d['thresholds'])
    if a=='generate':raise ValueError('Unexpected action')
    selection=d['selection']
    requests=expected(tasks,selection)
    if a=='prepare':return {'schema':'ramgpt-request-schedule-v1','requests':requests}
    if a!='score':raise ValueError('Unknown router action')
    lookup={x['id']:x for x in tasks}
    reserve=d['reserve'];rows=[]
    if len(d['rows'])!=len(requests):raise ValueError('Incomplete sparse run')
    for row,req in zip(d['rows'],requests):
        q=row['request']
        if any(q[k]!=v for k,v in req.items()):raise ValueError('Sparse request mismatch')
        task=lookup[q['task_id']]
        prompt,mode=b.prompt(task,q['strategy'],q['max_tokens'],reserve if q['strategy']=='allocated' else 0)
        if prompt!=q['prompt'] or q['body']['messages']!=[{'role':'user','content':prompt}]:raise ValueError('Native prompt mismatch')
        rows.append({'id':task['id'],'budget':q['max_tokens'],'strategy':q['strategy'],'reserve':reserve if q['strategy']=='allocated' else 0,'elapsed_s':row['elapsed_s'],**b.grade(task,row['raw'],mode,q['max_tokens'])})
    return {**observed(tasks,selection,rows,d['thresholds']),'rows':rows}

d=json.load(sys.stdin)
if d['action']=='generate':
    result=m.generate(d['seed'],d['variants'],d['suite'],d['ladder_variants'])
else:
    result=main(d)
json.dump(result,sys.stdout,allow_nan=False)

import collections
import json
import sys

sys.path.insert(0,sys.argv[1])
import budgetfit as b
import measurement as m
import recovery
import calibration

def tasks_for(manifest):
    m.verify(manifest)
    return {t['id']:t for t in manifest['tasks']}

def configs(budgets,strategies,reserve):
    if not budgets or len(set(budgets))!=len(budgets) or any(type(x) is not int or x<2 for x in budgets):raise ValueError('Invalid budgets')
    if not strategies or len(set(strategies))!=len(strategies) or any(x not in b.STRATEGIES for x in strategies):raise ValueError('Invalid strategies')
    if 'allocated' in strategies and (type(reserve) is not int or not 0<reserve<min(budgets)):raise ValueError('Invalid frozen reserve')
    return [(cap,strategy,reserve if strategy=='allocated' else 0) for cap in budgets for strategy in strategies]

def build(manifest,budgets,strategies,reserve):
    tasks=tasks_for(manifest)
    rows=[]
    for cap,strategy,r in configs(budgets,strategies,reserve):
        for tid,t in tasks.items():
            prompt,_=b.prompt(t,strategy,cap,r)
            rows.append({'request_id':f'{strategy}-{cap}-{len(rows):06d}','task_id':tid,'strategy':strategy,'prompt':prompt,'max_tokens':cap})
    return {'schema':'ramgpt-request-schedule-v1','requests':rows}

def evaluate(data):
    manifest=data['manifest'];tasks=tasks_for(manifest)
    cs=configs(data['budgets'],data['strategies'],data.get('reserve'))
    expected={(tid,cap,strategy) for tid in tasks for cap,strategy,_ in cs}
    reserve={(cap,strategy):r for cap,strategy,r in cs}
    seen=set()
    groups=collections.defaultdict(list)
    details=[]
    for row in data['rows']:
        q=row['request'];tid=q['task_id'];cap=q['max_tokens'];strategy=q['strategy']
        key=(tid,cap,strategy)
        if key not in expected or key in seen:raise ValueError('Unexpected or duplicate evaluation key')
        seen.add(key)
        t=tasks[tid]
        prompt,mode=b.prompt(t,strategy,cap,reserve[cap,strategy])
        if q['prompt']!=prompt or q['body']['messages']!=[{'role':'user','content':prompt}]:raise ValueError('Native prompt mismatch: '+tid)
        grade=b.grade(t,row['raw'],mode,cap)
        c=row['raw']['choices'][0]
        native=m.grade(t,c['message']['content'],mode,c['finish_reason'])
        if grade['grade']!=native:raise ValueError('Native grading mismatch')
        detail={'id':tid,'track':t['track'],'family':t['family'],'domain':t['domain'],'budget':cap,'strategy':strategy,'reserve':reserve[cap,strategy],
                'mode':mode,'request_id':q['request_id'],'ordinal':row['ordinal'],'elapsed_s':row['elapsed_s'],**grade}
        if 'level' in t:detail.update(level=t['level'],root=t['root'])
        details.append(detail)
        groups[cap,strategy].append({**detail,'raw':row['raw']})
    if seen!=expected:raise ValueError('Incomplete evaluation matrix')
    profiles=[]
    for cap,strategy,r in cs:
        rows=groups[cap,strategy]
        mode=rows[0]['mode']
        axes=[]
        for track in sorted({t['track'] for t in tasks.values()}):
            subset=[x for x in rows if x['track']==track]
            by={}
            for dimension in ('family','domain','level'):
                if dimension=='level' and track!='ladder':continue
                by[dimension]={str(k):m.report([x for x in subset if x[dimension]==k],sum(1 for x in subset if x[dimension]==k)) for k in sorted({x[dimension] for x in subset})}
            axes.append({'track':track,'measurement':m.report(subset,len(subset)),'recovery':recovery.report([x['recovery'] for x in subset]),'breakdown':by})
        profiles.append({'budgetfit':calibration.profile((cap,strategy,r),rows,len(tasks)),
                         'mode':mode,'prompt_contract':'native TrustMeBro' if strategy in ('blind','native_strict') else 'BudgetFit-modified TrustMeBro',
                         'tracks':axes})
    return {'engine':{'budgetfit':b.VERSION,'trustmebro_protocol':m.VERSION,'recovery':recovery.VERSION,'vendored_commit':b.ENGINE},
            'manifest_sha256':b.core.digest(manifest),'profiles':profiles,'rows':details,
            'scope':'observed task matrix only; no deployment recommendation; modified prompts are not native benchmark runs'}

if b.VERSION!='0.6.0' or m.VERSION!='trustmebro-v0.2' or recovery.VERSION!='answer-recovery-v0.4' or b.ENGINE!='e362d6bed40cdab94c07f489e17f779eec2124a6':
    raise ValueError('Unsupported evaluator versions')
if not str(b.__file__).startswith(sys.argv[1]+'/'):raise ValueError('Evaluator not loaded from frozen archive')
data=json.load(sys.stdin)
if data['action']=='prepare':result=build(data['manifest'],data['budgets'],data['strategies'],data.get('reserve'))
elif data['action']=='score':result=evaluate(data)
else:raise ValueError('Unknown evaluator operation')
json.dump(result,sys.stdout,allow_nan=False)

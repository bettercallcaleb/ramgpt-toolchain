import argparse
import json
from pathlib import Path
from registry import read_json
import evaluation

FIELDS=('semantic_correct','joint_success_recovered','transition','unknown_reason','method','labels')

def audit(archive,folder,reserve):
    p=Path(folder)
    plan=read_json(p/'plan.json')
    manifest=read_json(p/'tasks.json')
    raw=read_json(p/'raw.json')
    gold=read_json(p/'graded.json')
    if len(raw)!=len(gold) or len(raw)!=plan['planned_requests']:
        raise ValueError('Incomplete historical source')
    if [(x['id'],x['budget'],x['arm']) for x in raw]!=[tuple(x) for x in plan['schedule']]:
        raise ValueError('Historical schedule mismatch')
    rows=[{'ordinal':n,'request':{'request_id':str(n),'task_id':r['id'],'strategy':r['arm'],
          'prompt':r['prompt'],'max_tokens':r['budget'],'body':{'messages':[{'role':'user','content':r['prompt']}]}},
          'raw':r['raw'],'elapsed_s':r['elapsed_s']} for n,r in enumerate(raw)]
    result=evaluation.worker(archive,{'action':'score','manifest':manifest,'rows':rows,'budgets':plan['budgets'],
                                     'strategies':plan['arms'],'reserve':reserve})
    if result['manifest_sha256']!=plan['manifest_sha256'] or len(result['rows'])!=len(gold):
        raise ValueError('Historical manifest or count mismatch')
    for a,z in zip(result['rows'],gold):
        key=(a['id'],a['budget'],a['strategy'])
        if key!=(z['id'],z['budget'],z['arm']) or a['grade']!=z['grade']:
            raise ValueError('Historical grade mismatch: '+str(key))
        for field in FIELDS:
            if a['recovery'][field]!=z['recovery'][field]:
                raise ValueError('Historical recovery mismatch: '+str((key,field)))
        for field in ('finish_reason','completion_tokens','elapsed_s'):
            if a[field]!=z[field]:raise ValueError('Historical measurement mismatch: '+str((key,field)))
    return {'model':plan['model'],'requests':len(rows),'status':'parity_passed','recovery_fields':list(FIELDS)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluator',required=True)
    p.add_argument('--reserve',type=int,required=True)
    p.add_argument('folders',nargs='+')
    args=p.parse_args()
    results=[audit(args.evaluator,path,args.reserve) for path in args.folders]
    print(json.dumps({'requests':sum(x['requests'] for x in results),'sources':results},indent=2))

if __name__=='__main__':main()

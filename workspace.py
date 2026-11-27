import os
import re
from pathlib import Path
from registry import read_json,fingerprint
import evidence
import native_bridge
import native_tools
import weight

TOOLS={
 'q4fit':({'native':'file','quantizer':'executable','gguf_root':'directory'},['build','weight-import recipe','weight-import q4fit']),
 'quantbench':({'native':'file'},['weight-import quantbench','weight-select']),
 'conceptladder':({'native':'directory'},['ladder-import']),
 'trustmebro':({'evaluator':'file'},['prepare','score']),
 'budgetfit':({'evaluator':'file'},['select','router-prepare']),
 'llama-autotune':({'autotune_source':'directory'},['runtime-prepare','runtime-compare']),
 'gguf-healer':({'native':'file','gguf_root':'directory'},['healer-compose','healer-import']),
 'exl3-first-mover':({'native':'file','backend':'directory'},['exl3-preflight','exl3-import']),
 'hereticbench':({'native':'directory'},['heretic-import'])
}

def load(path):
    if path is None:return {}
    p=Path(path).expanduser().resolve(strict=True);doc=read_json(p)
    if not isinstance(doc,dict) or set(doc)!={'tools'} or not isinstance(doc['tools'],dict):raise ValueError('Expected tools configuration object')
    out={}
    for name,values in doc['tools'].items():
        if name not in TOOLS or not isinstance(values,dict) or set(values)-TOOLS[name][0].keys():raise ValueError('Unknown tool or configuration field: '+name)
        out[name]={}
        for k,v in values.items():
            if not isinstance(v,str) or not v.strip():raise ValueError('Expected nonempty path')
            q=Path(v).expanduser()
            if not q.is_absolute():q=p.parent/q
            out[name][k]=str(q.resolve())
    return out

def catalog(config):
    rows=[]
    for name,(fields,commands) in TOOLS.items():
        paths={}
        for k,kind in fields.items():
            val=config.get(name,{}).get(k)
            p=Path(val) if val else None
            ok=bool(p and (p.is_dir() if kind=='directory' else p.is_file() and (kind!='executable' or os.access(p,os.X_OK))))
            paths[k]={'path':val,'present':ok}
        rows.append({'tool':name,'commands':commands,'paths':paths,'configured':all(x['present'] for x in paths.values())})
    return {'tools':rows,'scope':'Local path presence only; native compatibility and execution readiness are checked by each adapter.'}

def apply(args,config):
    action=args.action
    mapping={'build':'q4fit','healer-compose':'gguf-healer','healer-import':'gguf-healer','ladder-import':'conceptladder','exl3-preflight':'exl3-first-mover','exl3-import':'exl3-first-mover','heretic-import':'hereticbench','runtime-prepare':'llama-autotune','router-parity':'budgetfit','prepare':'budgetfit'}
    tool=mapping.get(action)
    if action=='weight-import' and args.kind!='recipe':tool=args.kind
    if tool is None:return
    keys={'build':('native','quantizer','gguf_root'),'healer-compose':('native','gguf_root'),'healer-import':('native','gguf_root'),'ladder-import':('native',),'exl3-preflight':('native','backend'),'exl3-import':('native',),'heretic-import':('native',),'weight-import':('native',),'runtime-prepare':('autotune_source',),'router-parity':('evaluator',),'prepare':('evaluator',)}[action]
    for k in keys:
        if getattr(args,k,None) is None:setattr(args,k,config.get(tool,{}).get(k))
        if not getattr(args,k,None):raise ValueError('Supply --'+k.replace('_','-')+' or configure '+tool+'.'+k)

def model_info(reg,store,subject):
    if subject.startswith('exl3:'):
        aid=evidence.checked_id(subject[5:])
        rows=[r for r in native_bridge.find(store,'exl3-first-mover') if r['subject'].get('artifact_id')==aid]
        if not rows:raise ValueError('Unknown EXL3 artifact identity')
        sizes={r['artifact_bytes'] for r in rows}
        if len(sizes)!=1:raise ValueError('Conflicting EXL3 size')
        return {'format':'exl3','artifact_id':aid,'bytes':sizes.pop(),'producer_reports':[{'report_id':r['report_id'],'repo_id':r['subject']['repo_id'],'source_revision':r['subject']['source_revision'],'declared_bits':r['declared_bits']} for r in rows],'runtime_evidence':[],'scope':'EXL3 manifest identity; native runtime integration pending'}
    aid=subject if re.fullmatch('[0-9a-f]{64}',subject) else fingerprint(subject,'model')['sha256']
    view=weight.artifact_view(store,aid)
    receipts=native_tools.find(store,aid)
    local=reg.db.execute('SELECT bytes FROM artifacts WHERE sha256=?',(aid,)).fetchone()
    sizes={v for v in (view['artifact_bytes'],local[0] if local else None) if v is not None}
    for r in receipts:
        if r['tool']=='gguf-healer':sizes.add(r['inputs']['output']['bytes'])
        elif 'artifact_bytes' in r:sizes.add(r['artifact_bytes'])
    if len(sizes)!=1:raise ValueError('Unknown model or conflicting byte size; register the model first')
    related=weight.find(store,reference=aid)
    children=[r for r in related if r['kind']=='quantization_recipe']
    repaired=[r for r in native_tools.find(store) if r['tool']=='gguf-healer' and r['inputs']['base']['sha256']==aid]
    return {'format':'gguf','sha256':aid,'bytes':sizes.pop(),'registered':local is not None,'weight_evidence':view['weight_evidence'],'managed_task_evidence':view['managed_task_evidence'],'native_receipts':receipts,'derived_artifacts':[{'sha256':r['artifact_sha256'],'tool':r['producer'],'kind':r['kind'],'report_id':r['report_id']} for r in children]+[{'sha256':r['artifact_sha256'],'tool':'gguf-healer','kind':'tensor_transplant','receipt_id':r['receipt_id']} for r in repaired],'comparison_artifacts':[{'sha256':r['artifact_sha256'],'tool':r['producer'],'kind':r['kind'],'report_id':r['report_id']} for r in related if r['kind']!='quantization_recipe'],'missing_managed_axes':view['missing_axes'],'scope':'Separate evidence axes; unbound historical pair reports are not attached to model identities.'}


def models(reg,store):
    rows={}
    def add(aid,size,kind):
        r=rows.setdefault(('gguf',aid),{'format':'gguf','sha256':aid,'bytes':size,'sources':set()})
        if r['bytes']!=size:raise ValueError('Conflicting model size')
        r['sources'].add(kind)
    for aid,size in reg.db.execute("SELECT DISTINCT a.sha256,a.bytes FROM artifacts a JOIN observations o ON a.sha256=o.sha256 WHERE o.role='model'"):
        add(aid,size,'registry')
    for r in weight.find(store):
        add(r['artifact_sha256'],r['artifact_bytes'],r['producer'])
        if r.get('reference_bytes') is not None:add(r['reference_sha256'],r['reference_bytes'],'reference')
    for r in native_tools.find(store):
        if r['tool']=='gguf-healer':
            for k,v in r['inputs'].items():add(v['sha256'],v['bytes'],'gguf-healer:'+k)
        elif 'artifact_bytes' in r:add(r['artifact_sha256'],r['artifact_bytes'],r['tool'])
    for r in store.find():
        source=store.get(r['source_id'])
        ref=next(x for x in source['contract']['inputs'] if x['role']=='model')
        add(ref['sha256'],ref['bytes'],'managed_evaluation')
    for r in native_bridge.find(store,'exl3-first-mover'):
        aid=r['subject'].get('artifact_id')
        if aid is None:continue
        item=rows.setdefault(('exl3',aid),{'format':'exl3','artifact_id':aid,'bytes':r['artifact_bytes'],'sources':set()})
        if item['bytes']!=r['artifact_bytes']:raise ValueError('Conflicting EXL3 artifact size')
        item['sources'].add('exl3-first-mover')
    return {'models':[{**r,'sources':sorted(r['sources'])} for _,r in sorted(rows.items())],'scope':'Known local or historical model identities; not a claim that files are present, compatible or validated.'}

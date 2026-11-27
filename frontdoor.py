from pathlib import Path
import re
from registry import read_json
import evidence
import evaluation
import inference
import measurement_plan
import native_tools
import native_bridge
import cross_format

def format_of(value,requested=None):
    p=Path(value).expanduser()
    inferred='exl3' if p.is_dir() else 'gguf' if p.is_file() and p.suffix.lower()=='.gguf' else None
    if inferred is None or (requested is not None and inferred!=requested):raise ValueError('Expected matching GGUF file or local model directory')
    return inferred

def configure(args,config,tool,keys):
    for k in keys:
        if getattr(args,k,None) is None:setattr(args,k,config.get(tool,{}).get(k))
        if not getattr(args,k,None):raise ValueError('Supply --'+k.replace('_','-')+' or configure '+tool+'.'+k)

def build(reg,store,args,config):
    kind=format_of(args.model,args.format)
    if kind=='exl3':
        if args.gguf_root or args.quantizer or args.default is not None or args.set or args.imatrix or args.threads is not None:raise ValueError('GGUF quantizer options are not EXL3 options')
        configure(args,config,'exl3-first-mover',('native','backend','python'))
        if not args.gpu_uuid:raise ValueError('EXL3 build requires --gpu-uuid')
        calibration={k:getattr(args,k) for k in ('cal_rows','cal_cols') if getattr(args,k,None) is not None}
        return native_bridge.exl3_build(store,args.model,args.out,args.native,args.backend,args.python,args.gpu_uuid,args.bits if args.bits is not None else 4.0,args.execute,**calibration)
    if any(x is not None for x in (args.backend,args.python,args.gpu_uuid,args.bits,getattr(args,"cal_rows",None),getattr(args,"cal_cols",None))):raise ValueError('EXL3 options are not GGUF quantizer options')
    configure(args,config,'q4fit',('native','quantizer','gguf_root'))
    return native_tools.q4fit_build(reg,store,args.model,args.out,args.native,args.gguf_root,args.quantizer,args.default or 'Q4_K',args.set,args.threads if args.threads is not None else 12,args.imatrix,args.execute)

def run(reg,store,args,config):
    value=args.id
    if re.fullmatch('[0-9a-f]{64}',value):
        if args.format or any(getattr(args,k,None) is not None for k in ('spec','dataset','seed','budgets','max_requests','max_completion_tokens','gpu_uuid','native','backend','python','evaluator','kv','context','strategies','reserve','variants')):raise ValueError('Model preparation options cannot modify a frozen run')
        row=None
        if store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='measurement_plans'").fetchone():
            row=store.db.execute('SELECT id FROM measurement_plans WHERE id=?',(value,)).fetchone()
        if row:
            if args.max_new is not None or args.out is not None or args.archive:raise ValueError('Measurement plan already binds output and archives')
            return measurement_plan.run(reg,store,value,args.execute)
        if not args.out:raise ValueError('Contract run requires --out')
        plan=reg.contract(value)
        if args.archive and (args.max_new is not None or plan['workload'].get('evaluator') not in ('BudgetFit 0.6.0','BudgetFit 0.6.0 Router')):raise ValueError('Archive requires a full native evaluation')
        result=inference.run(reg,value,args.out,args.max_new,evidence_db=args.evidence_db)
        if args.archive:result={**result,'archive':store.ingest(reg,value,args.out)}
        return result
    kind=format_of(value,args.format)
    if args.max_new is not None or args.archive:raise ValueError('Model run already binds and archives a full session')
    if not args.out or not args.budgets or args.max_requests is None or args.max_completion_tokens is None:raise ValueError('Model run requires --out, --budgets and both cost caps')
    strategies=args.strategies or ['allocated'];reserve=args.reserve if args.reserve is not None else 64
    if kind=='exl3':
        if args.spec or args.seed is not None or args.kv or args.variants is not None:raise ValueError('EXL3 requires a shared dataset, not GGUF preparation options')
        if not args.dataset or not args.gpu_uuid:raise ValueError('EXL3 run requires --dataset and --gpu-uuid')
        configure(args,config,'exl3-first-mover',('native','backend','python'))
        configure(args,config,'budgetfit',('evaluator',))
        return cross_format.run(store,value,args.dataset,args.evaluator,args.native,args.backend,args.python,args.gpu_uuid,args.out,args.budgets,strategies,reserve,args.context if args.context is not None else 4096,args.max_requests,args.max_completion_tokens,args.execute)
    if any(getattr(args,k,None) is not None for k in ('dataset','gpu_uuid','native','backend','python','evaluator','context')):raise ValueError('GGUF run uses model-specific --spec for runtime, GPU, context and template')
    if not args.spec or args.seed is None:raise ValueError('GGUF model run requires --spec and --seed')
    spec=read_json(args.spec)
    models=[Path(x['path']).expanduser().resolve(strict=True) for x in spec['inputs'] if x['role']=='model']
    if models!=[Path(value).expanduser().resolve(strict=True)]:raise ValueError('Base spec model path does not match requested model')
    plan=measurement_plan.prepare(reg,store,args.spec,args.out,args.seed,args.variants if args.variants is not None else 1,args.budgets,strategies,reserve,args.kv or ['f16'],args.max_requests,args.max_completion_tokens)
    if args.execute:return {'plan':plan,'result':measurement_plan.run(reg,store,plan['plan_id'],True)}
    return plan

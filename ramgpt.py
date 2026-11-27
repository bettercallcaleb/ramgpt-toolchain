import argparse
import json
import sys
from pathlib import Path
from registry import Registry,read_json
from release_info import VERSION

def main():
    p=argparse.ArgumentParser(prog='ramgpt',description='RAMGPT Toolchain: local artifact registry and frozen experiment contracts')
    p.add_argument('--version',action='version',version=VERSION)
    p.add_argument('--db',default='state/registry.sqlite')
    p.add_argument('--evidence-db',default='state/evidence.sqlite')
    p.add_argument('--tools-config')
    sub=p.add_subparsers(dest='action',required=True)
    r=sub.add_parser('register');r.add_argument('path');r.add_argument('--role',default='model')
    a=sub.add_parser('artifact');a.add_argument('sha256')
    f=sub.add_parser('freeze');f.add_argument('spec')
    c=sub.add_parser('contract');c.add_argument('id')
    v=sub.add_parser('verify');v.add_argument('id')
    m=sub.add_parser('runtime-smoke');m.add_argument('id');m.add_argument('--out',required=True)
    n=sub.add_parser('run');n.add_argument('id');n.add_argument('--out');n.add_argument('--max-new',type=int);n.add_argument('--format',choices=['gguf','exl3']);n.add_argument('--spec');n.add_argument('--dataset');n.add_argument('--seed',type=int);n.add_argument('--variants',type=int);n.add_argument('--budgets',type=int,nargs='+');n.add_argument('--strategies',nargs='+');n.add_argument('--reserve',type=int);n.add_argument('--kv',nargs='+');n.add_argument('--context',type=int);n.add_argument('--max-requests',type=int);n.add_argument('--max-completion-tokens',type=int);n.add_argument('--native');n.add_argument('--backend');n.add_argument('--python');n.add_argument('--evaluator');n.add_argument('--gpu-uuid');n.add_argument('--execute',action='store_true');n.add_argument('--archive',action='store_true')
    e=sub.add_parser('score');e.add_argument('id');e.add_argument('--run',required=True);e.add_argument('--out',required=True)
    q=sub.add_parser('prepare');q.add_argument('--dataset',required=True);q.add_argument('--evaluator');q.add_argument('--budgets',type=int,nargs='+',required=True);q.add_argument('--strategies',nargs='+',required=True);q.add_argument('--reserve',type=int);q.add_argument('--out',required=True)
    z=sub.add_parser('archive-run');z.add_argument('id');z.add_argument('--run',required=True);z.add_argument('--report')
    z=sub.add_parser('evidence-find');z.add_argument('--artifact');z.add_argument('--contract')
    z=sub.add_parser('evidence-get');z.add_argument('id');z.add_argument('--out',required=True)
    z=sub.add_parser('replay');z.add_argument('id');z.add_argument('--source');z.add_argument('--out',required=True)
    z=sub.add_parser('select');z.add_argument('evaluation_id');z.add_argument('--policy',required=True);z.add_argument('--validation-seed',type=int)
    z=sub.add_parser('validate-selection');z.add_argument('decision_id');z.add_argument('evaluation_id')
    z=sub.add_parser('router-prepare');z.add_argument('evaluation_id');z.add_argument('--validation-seed',required=True,type=int);z.add_argument('--thresholds',required=True);z.add_argument('--out',required=True)
    z=sub.add_parser('router-parity');z.add_argument('--evaluator');z.add_argument('--folder',required=True)
    z=sub.add_parser('runtime-prepare');z.add_argument('--spec',required=True);z.add_argument('--autotune-source');z.add_argument('--out',required=True);z.add_argument('--seed',type=int,default=611)
    z=sub.add_parser('runtime-compare');z.add_argument('--plan',required=True);z.add_argument('evaluation_ids',nargs=2)
    z=sub.add_parser('weight-import');z.add_argument('kind',choices=['quantbench','q4fit','recipe']);z.add_argument('path');z.add_argument('--native')
    z=sub.add_parser('weight-find');z.add_argument('--artifact');z.add_argument('--reference')
    z=sub.add_parser('weight-view');z.add_argument('artifact')
    z=sub.add_parser('weight-select');z.add_argument('report_id');z.add_argument('--max-kl',type=float,required=True);z.add_argument('--max-bytes',type=int,required=True)
    z=sub.add_parser('joint-select');z.add_argument('--policy',required=True);z.add_argument('--anchor',required=True);z.add_argument('--budget',type=int,required=True);z.add_argument('--strategy',required=True);z.add_argument('--reserve',type=int);z.add_argument('--validation-seed',type=int,required=True);z.add_argument('evaluation_ids',nargs='+')
    z=sub.add_parser('joint-prepare');z.add_argument('decision_id');z.add_argument('--out',required=True)
    z=sub.add_parser('joint-validate');z.add_argument('decision_id');z.add_argument('evaluation_ids',nargs=2)
    z=sub.add_parser('passport-export',aliases=['passport']);z.add_argument('artifact');z.add_argument('--out',required=True);z.add_argument('--consolidated',action='store_true');z.add_argument('--native')
    z=sub.add_parser('passport-verify');z.add_argument('path');z.add_argument('--expected-id');z.add_argument('--model');z.add_argument('--local',action='store_true')
    z=sub.add_parser('doctor');z.add_argument('--evaluator');z.add_argument('--runtime');z.add_argument('--model');z.add_argument('--gpu',action='store_true')
    z=sub.add_parser('optimize');z.add_argument('model');z.add_argument('--vram',required=True);z.add_argument('--objective',choices=['balanced','memory','latency','tokens'],default='balanced');z.add_argument('--min-accuracy',type=float,default=.85);z.add_argument('--min-tasks',type=int,default=8);z.add_argument('--headroom',default='1GiB');z.add_argument('--context',type=int);z.add_argument('--max-sample-gap',type=float,default=.25);z.add_argument('--fidelity-report');z.add_argument('--max-kl',type=float);z.add_argument('--out')
    z=sub.add_parser('measurement-prepare');z.add_argument('--spec',required=True);z.add_argument('--out',required=True);z.add_argument('--seed',type=int,required=True);z.add_argument('--variants',type=int,default=1);z.add_argument('--budgets',type=int,nargs='+',required=True);z.add_argument('--strategies',nargs='+',default=['allocated']);z.add_argument('--reserve',type=int,default=64);z.add_argument('--kv',nargs='+',default=['f16','q8_0']);z.add_argument('--max-requests',type=int,required=True);z.add_argument('--max-completion-tokens',type=int,required=True)
    z=sub.add_parser('measurement-run');z.add_argument('plan_id');z.add_argument('--execute',action='store_true')
    z=sub.add_parser('workflow-select');z.add_argument('plan_id');z.add_argument('--policy',required=True);z.add_argument('--anchor-kv',required=True);z.add_argument('--budget',type=int,required=True);z.add_argument('--strategy',required=True);z.add_argument('--reserve',type=int,required=True);z.add_argument('--validation-seed',type=int,required=True);z.add_argument('--max-requests',type=int,required=True);z.add_argument('--max-completion-tokens',type=int,required=True)
    z=sub.add_parser('workflow-finish');z.add_argument('plan_id');z.add_argument('--execute',action='store_true')
    z=sub.add_parser('healer-compose');z.add_argument('base');z.add_argument('donor');z.add_argument('--out',required=True);z.add_argument('--native');z.add_argument('--gguf-root');z.add_argument('--tensor',action='append',required=True);z.add_argument('--max-growth',type=float,required=True);z.add_argument('--execute',action='store_true')
    z=sub.add_parser('healer-import');z.add_argument('manifest');z.add_argument('--native');z.add_argument('--gguf-root');z.add_argument('--max-growth',type=float,required=True)
    z=sub.add_parser('ladder-import');z.add_argument('model');z.add_argument('raw');z.add_argument('--native');z.add_argument('--seed',type=int,default=211);z.add_argument('--variants',type=int,default=1);z.add_argument('--track',choices=['knowledge','ifeval'],default='knowledge');z.add_argument('--ifeval-root');z.add_argument('--limit',type=int);z.add_argument('--deps');z.add_argument('--nltk-data')
    z=sub.add_parser('tool-find');z.add_argument('--artifact')
    z=sub.add_parser('exl3-preflight');z.add_argument('repo');z.add_argument('--native');z.add_argument('--backend');z.add_argument('--bits',type=float,default=4.0)
    z=sub.add_parser('exl3-build');z.add_argument('source');z.add_argument('--out',required=True);z.add_argument('--native');z.add_argument('--backend');z.add_argument('--python');z.add_argument('--gpu-uuid',required=True);z.add_argument('--bits',type=float,default=4.0);z.add_argument('--execute',action='store_true')
    z=sub.add_parser('exl3-run');z.add_argument('folder');z.add_argument('--dataset',required=True);z.add_argument('--evaluator');z.add_argument('--native');z.add_argument('--backend');z.add_argument('--python');z.add_argument('--gpu-uuid',required=True);z.add_argument('--out',required=True);z.add_argument('--budgets',type=int,nargs='+',required=True);z.add_argument('--strategies',nargs='+',default=['allocated']);z.add_argument('--reserve',type=int,default=64);z.add_argument('--context',type=int,default=4096);z.add_argument('--max-requests',type=int,required=True);z.add_argument('--max-completion-tokens',type=int,required=True);z.add_argument('--execute',action='store_true')
    z=sub.add_parser('cross-compare');z.add_argument('gguf_evaluation');z.add_argument('exl3_report')
    z=sub.add_parser('exl3-import');z.add_argument('folder');z.add_argument('manifest');z.add_argument('--native')
    z=sub.add_parser('heretic-import');z.add_argument('--native');z.add_argument('--dataset',required=True);z.add_argument('--base',required=True);z.add_argument('--variant',required=True);z.add_argument('--track',choices=['retention','epistemic'],required=True)
    z=sub.add_parser('tool-reports');z.add_argument('--tool',choices=['exl3-first-mover','hereticbench'])
    z=sub.add_parser('recommend');z.add_argument('--candidate',action='append');z.add_argument('--vram',required=True);z.add_argument('--objective',choices=['balanced','memory','latency','tokens'],default='balanced');z.add_argument('--min-accuracy',type=float,default=.85);z.add_argument('--min-tasks',type=int,default=8);z.add_argument('--headroom',default='1GiB');z.add_argument('--context',type=int);z.add_argument('--max-sample-gap',type=float,default=.25);z.add_argument('--fidelity-report');z.add_argument('--max-kl',type=float);z.add_argument('--max-model-size');z.add_argument('--manifest');z.add_argument('--gpu-name');z.add_argument('--out')
    z=sub.add_parser('tool-sync');z.add_argument('source_evidence_db')
    z=sub.add_parser('tools')
    z=sub.add_parser('model-info');z.add_argument('subject',nargs='?')
    z=sub.add_parser('build');z.add_argument('model');z.add_argument('--out',required=True);z.add_argument('--native');z.add_argument('--gguf-root');z.add_argument('--quantizer');z.add_argument('--default');z.add_argument('--set',action='append',default=[]);z.add_argument('--threads',type=int);z.add_argument('--imatrix');z.add_argument('--format',choices=['gguf','exl3']);z.add_argument('--backend');z.add_argument('--python');z.add_argument('--gpu-uuid');z.add_argument('--bits',type=float);z.add_argument('--execute',action='store_true')
    args=p.parse_args()
    reg=None
    try:
        import workspace
        cfg=workspace.load(args.tools_config)
        workspace.apply(args,cfg)
        if args.action=='tools':
            print(json.dumps(workspace.catalog(cfg),indent=2,ensure_ascii=False,allow_nan=False))
            return 0
        if args.action=='doctor':
            import doctor
            result=doctor.check(args.evaluator,args.runtime,args.model,args.gpu)
            print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
            return 0 if result['status']=='checks_passed' else 1
        if args.action=='passport-verify':
            import passport,evidence
            store=evidence.Store(args.evidence_db) if args.local else None
            try:
                packet=read_json(args.path)
                import model_passport
                if packet.get('schema')==model_passport.SCHEMA:
                    if args.local:reg=Registry(args.db)
                    result=model_passport.verify_model(packet,args.expected_id,args.model,reg,store)
                elif args.local:
                    reg=Registry(args.db)
                    result=passport.verify_local(reg,store,packet,args.expected_id,args.model)
                else:result=passport.verify_model(packet,args.expected_id,args.model)
            finally:
                if store is not None:store.close()
            print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
            return 0
        reg=Registry(args.db)
        if args.action=='register':result=reg.register(args.path,args.role)
        elif args.action=='artifact':result=reg.get(args.sha256)
        elif args.action=='freeze':result=reg.freeze(read_json(args.spec))
        elif args.action=='contract':result=reg.contract(args.id)
        elif args.action=='verify':result=reg.verify(args.id)
        elif args.action=='runtime-prepare':
            import runtime_opt
            result=runtime_opt.prepare(reg,args.spec,args.autotune_source,args.out,args.seed)
            import evidence
            store=evidence.Store(args.evidence_db)
            try:
                with store.db:plan_id=store.put_blob(result)
            finally:store.close()
            result={**result,'comparison_plan_id':plan_id}
        elif args.action=='router-parity':
            import adaptive
            result=adaptive.historical(args.evaluator,args.folder)
        elif args.action in ('recommend','tool-sync','build','model-info','exl3-run','cross-compare','exl3-build','exl3-preflight','exl3-import','heretic-import','tool-reports','healer-compose','healer-import','ladder-import','tool-find','workflow-select','workflow-finish','measurement-prepare','measurement-run','optimize','passport-export','passport','joint-select','joint-prepare','joint-validate','weight-import','weight-find','weight-view','weight-select','runtime-compare','router-prepare','archive-run','evidence-find','evidence-get','replay','select','validate-selection'):
            import evidence,evaluation
            store=evidence.Store(args.evidence_db)
            try:
                if args.action=='recommend':
                    import portfolio,planner
                    policy=planner.policy(args.vram,args.objective,args.min_accuracy,args.min_tasks,args.headroom,args.context,args.max_sample_gap,args.fidelity_report,args.max_kl)
                    result=portfolio.select(store,reg,policy,args.candidate,planner.bytes_arg(args.max_model_size) if args.max_model_size else None,args.manifest,args.gpu_name)
                    if args.out:evaluation.write_new(args.out,result)
                elif args.action=='tool-sync':
                    result=workspace.sync_native(store,args.source_evidence_db)
                elif args.action=='build':
                    import frontdoor
                    result=frontdoor.build(reg,store,args,cfg)
                elif args.action=='model-info':
                    result=workspace.model_info(reg,store,args.subject) if args.subject else workspace.models(reg,store)
                elif args.action=='exl3-preflight':
                    import native_bridge
                    result=native_bridge.exl3_preflight(store,args.repo,args.native,args.backend,args.bits)
                elif args.action=='exl3-run':
                    import cross_format
                    result=cross_format.run(store,args.folder,args.dataset,args.evaluator,args.native,args.backend,args.python,args.gpu_uuid,args.out,args.budgets,args.strategies,args.reserve,args.context,args.max_requests,args.max_completion_tokens,args.execute)
                elif args.action=='cross-compare':
                    import cross_format
                    result=cross_format.compare(store,args.gguf_evaluation,args.exl3_report)
                elif args.action=='exl3-build':
                    import native_bridge
                    result=native_bridge.exl3_build(store,args.source,args.out,args.native,args.backend,args.python,args.gpu_uuid,args.bits,args.execute)
                elif args.action=='exl3-import':
                    import native_bridge
                    result=native_bridge.exl3_import(store,args.folder,args.manifest,args.native)
                elif args.action=='heretic-import':
                    import native_bridge
                    result=native_bridge.heretic_import(store,args.native,args.dataset,args.base,args.variant,args.track)
                elif args.action=='tool-reports':
                    import native_bridge
                    result=native_bridge.find(store,args.tool)
                elif args.action=='healer-compose':
                    import native_tools
                    result=native_tools.healer_compose(reg,store,args.base,args.donor,args.out,args.native,args.gguf_root,args.tensor,args.max_growth,args.execute)
                elif args.action=='healer-import':
                    import native_tools
                    result=native_tools.healer_import(reg,store,args.manifest,args.native,args.gguf_root,args.max_growth)
                elif args.action=='ladder-import':
                    import native_tools
                    result=native_tools.ladder_import(reg,store,args.model,args.raw,args.native,args.seed,args.variants,args.track,args.ifeval_root,args.limit,args.deps,args.nltk_data)
                elif args.action=='tool-find':
                    import native_tools
                    result=native_tools.find(store,args.artifact)
                elif args.action=='workflow-select':
                    import workflow
                    result=workflow.select(reg,store,args.plan_id,read_json(args.policy),args.anchor_kv,[args.budget,args.strategy,args.reserve],args.validation_seed,args.max_requests,args.max_completion_tokens)
                elif args.action=='workflow-finish':
                    import workflow
                    result=workflow.finish(reg,store,args.plan_id,args.execute)
                elif args.action=='measurement-prepare':
                    import measurement_plan
                    result=measurement_plan.prepare(reg,store,args.spec,args.out,args.seed,args.variants,args.budgets,args.strategies,args.reserve,args.kv,args.max_requests,args.max_completion_tokens)
                elif args.action=='measurement-run':
                    import measurement_plan
                    result=measurement_plan.run(reg,store,args.plan_id,args.execute)
                elif args.action=='optimize':
                    import planner
                    policy=planner.policy(args.vram,args.objective,args.min_accuracy,args.min_tasks,args.headroom,args.context,args.max_sample_gap,args.fidelity_report,args.max_kl)
                    result=planner.build(reg,store,args.model,policy)
                    if args.out:evaluation.write_new(args.out,result)
                elif args.action in ('passport-export','passport'):
                    import passport
                    if args.consolidated or args.action=='passport' or args.artifact.startswith('exl3:'):
                        import model_passport
                        result=model_passport.export(reg,store,args.artifact,args.out,args.native or cfg.get('exl3-first-mover',{}).get('native'))
                    else:result=passport.export(reg,store,args.artifact,args.out)
                elif args.action.startswith('joint-'):
                    import joint
                    if args.action=='joint-select':result=joint.freeze(store,args.evaluation_ids,args.anchor,[args.budget,args.strategy,args.reserve],read_json(args.policy),args.validation_seed)
                    elif args.action=='joint-prepare':result=joint.prepare(reg,store,args.decision_id,args.out)
                    else:result=joint.validate(store,args.decision_id,args.evaluation_ids)
                elif args.action.startswith('weight-'):
                    import weight
                    if args.action=='weight-import':
                        if args.kind!='recipe' and not args.native:raise ValueError('Native producer script required')
                        body=weight.recipe(args.path) if args.kind=='recipe' else getattr(weight,args.kind)(args.path,args.native)
                        result=weight.ingest(store,body)
                    elif args.action=='weight-find':result=weight.find(store,args.artifact,args.reference)
                    elif args.action=='weight-view':result=weight.artifact_view(store,args.artifact)
                    else:result=weight.select_fidelity(store,args.report_id,args.max_kl,args.max_bytes)
                elif args.action=='runtime-compare':
                    import runtime_opt
                    from registry import digest
                    plan=read_json(args.plan)
                    if store.get(digest(plan))!=plan:raise ValueError('Unregistered runtime comparison plan')
                    result=runtime_opt.compare(store,plan,args.evaluation_ids)
                elif args.action=='router-prepare':
                    import adaptive
                    result=adaptive.plan_holdout(reg,store,args.evaluation_id,args.validation_seed,read_json(args.thresholds),args.out)
                elif args.action=='select':
                    import optimize
                    result=optimize.freeze(store,args.evaluation_id,read_json(args.policy),args.validation_seed)
                elif args.action=='validate-selection':
                    import optimize
                    result=optimize.validate(store,args.decision_id,args.evaluation_id)
                elif args.action=='archive-run':result=store.ingest(reg,args.id,args.run,args.report)
                elif args.action=='evidence-find':result=store.find(args.artifact,args.contract)
                elif args.action=='evidence-get':
                    evaluation.write_new(args.out,store.get(args.id));result={'status':'exported','out':args.out}
                else:
                    replay=store.replay(reg,args.id,args.source);evaluation.write_new(args.out,replay);result={k:v for k,v in replay.items() if k!='report'};result['out']=args.out
            finally:store.close()
        elif args.action=='prepare':
            import evaluation
            result=evaluation.prepare(args.dataset,args.evaluator,args.budgets,args.strategies,args.reserve,args.out)
        elif args.action=='score':
            import evaluation
            result=evaluation.score(reg,args.id,args.run,args.out)
        elif args.action=='run':
            import evidence,frontdoor
            store=evidence.Store(args.evidence_db)
            try:result=frontdoor.run(reg,store,args,cfg)
            finally:store.close()
        else:
            import runtime
            if 'measurement_cohort_key' in reg.contract(args.id)['workload']:raise ValueError('Use measurement-run for bounded measurement contracts')
            result=runtime.smoke(reg,args.id,args.out)
        print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    except Exception as e:
        print('RAMGPT:',e,file=sys.stderr)
        return 1
    finally:
        if reg is not None:reg.close()
    return 0

if __name__=='__main__':raise SystemExit(main())

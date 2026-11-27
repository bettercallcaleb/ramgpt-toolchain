import argparse
import json
import sys
import time
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--backend',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--context',type=int,required=True)
    p.add_argument('--schedule',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--render-only',action='store_true')
    a=p.parse_args()
    sys.path.insert(0,a.backend)
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from exllamav3 import Generator,Job,GreedySampler,model_init
    import torch
    schedule=json.loads(Path(a.schedule).read_text())
    tok_cfg=json.loads((Path(a.model)/'tokenizer_config.json').read_text())
    cfg_json=json.loads((Path(a.model)/'config.json').read_text())
    if cfg_json.get('model_type')!='qwen3':raise ValueError('Initial EXL3 text adapter supports Qwen3 chat templates only')
    template=tok_cfg.get('chat_template')
    external=Path(a.model)/'chat_template.jinja'
    if external.exists():
        ext=external.read_text()
        if template is not None and template.strip()!=ext.strip():raise ValueError('Conflicting chat templates')
        template=ext
    if not isinstance(template,str) or not template:raise ValueError('Missing text chat template')
    env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
    compiled=env.from_string(template)
    def render(q):
        text=compiled.render(messages=[{'role':'user','content':q['prompt']}],add_generation_prompt=True,enable_thinking=False,tools=None,bos_token=tok_cfg.get('bos_token') or '',eos_token=tok_cfg.get('eos_token') or '')
        if not text.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n'):raise ValueError('Expected Qwen3 non-thinking assistant prefix')
        return text
    rendered=[render(q) for q in schedule]
    if a.render_only:
        with Path(a.out).open('x') as f:json.dump({'rendered':rendered,'requests':len(schedule)},f)
        return
    parser=argparse.ArgumentParser()
    model_init.add_args(parser,cache=True,default_cache_size=a.context)
    args=parser.parse_args(['-m',a.model])
    model,config,cache,tokenizer=model_init.init(args)
    generator=Generator(model=model,cache=cache,tokenizer=tokenizer)
    stops=sorted({i for i in [tokenizer.eos_token_id]+list(getattr(config,'eos_token_id_list',[]) or []) if isinstance(i,int) and i>=0})
    if not stops:raise ValueError('No model EOS tokens')
    ids=[tokenizer.encode(s,add_bos=False,encode_special_tokens=True) for s in rendered]
    if any(x.shape[-1]+q['max_tokens']+1>a.context for x,q in zip(ids,schedule)):raise ValueError('Prompt plus budget exceeds native context')
    if torch.cuda.device_count()!=1:raise ValueError('Expected one visible GPU')
    with Path(a.out).open('x') as f:
        for i,(q,tokens) in enumerate(zip(schedule,ids)):
            start=time.perf_counter()
            job=Job(input_ids=tokens,max_new_tokens=q['max_tokens'],stop_conditions=stops,sampler=GreedySampler(),stop_on_loop=False,identifier=i)
            generator.enqueue(job)
            final=None
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    if r.get('stage')=='streaming' and r.get('eos'):
                        if final is not None:raise ValueError('Multiple terminal events')
                        final=r
            torch.cuda.synchronize()
            if final is None or final['eos_reason'] not in ('max_new_tokens','stop_token'):raise ValueError('Unexpected native termination')
            generated=final['new_tokens'];prompt=int(tokens.shape[-1])
            if type(generated) is not int or not 0<=generated<=q['max_tokens']:raise ValueError('Invalid native token count')
            raw={'model':'exl3','choices':[{'message':{'content':final['full_completion']},'finish_reason':'length' if final['eos_reason']=='max_new_tokens' else 'stop'}],'usage':{'prompt_tokens':prompt,'completion_tokens':generated,'total_tokens':prompt+generated}}
            row={'request_id':q['request_id'],'raw':raw,'elapsed_s':time.perf_counter()-start,'native':{k:final[k] for k in ('eos_reason','time_prefill','time_generate','cached_tokens')}}
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()

if __name__=='__main__':main()

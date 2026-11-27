import os
from pathlib import Path
import adaptive
import evaluation
import workspace
from registry import fingerprint

def discover(roots,overrides=(),depth=3):
    if type(depth) is not int or not 0<=depth<=5:raise ValueError('Discovery depth must be 0..5')
    candidates={}
    def add(tool,key,path):
        p=Path(path)
        if not p.exists():return
        kind=workspace.TOOLS[tool][0].get(key,'executable' if key in ('python','runtime') else 'file' if key=='evidence_script' else 'directory')
        if not (p.is_dir() if kind=='directory' else p.is_file() and (kind!='executable' or os.access(p,os.X_OK))):return
        value=os.path.abspath(p) if key=='python' else str(p.resolve())
        candidates.setdefault((tool,key),set()).add(value)
    seen=set()
    for raw in roots:
        root=Path(raw).expanduser().resolve(strict=True)
        if not root.is_dir():raise ValueError('Discovery root must be a directory')
        for path,dirs,files in os.walk(root,followlinks=False):
            p=Path(path);d=len(p.relative_to(root).parts)
            dirs[:]=sorted(x for x in dirs if not x.startswith('.') and x not in ('__pycache__','node_modules','models','state','dist-info','site-packages','work'))
            if d>=depth:dirs.clear()
            if p in seen:continue
            seen.add(p)
            if len(seen)>10000:raise ValueError('Discovery exceeded 10000 directories; specify narrower roots')
            if 'q4fit.py' in files and 'build' not in p.relative_to(root).parts:add('q4fit','native',p/'q4fit.py')
            if 'q4evidence.py' in files and 'build' not in p.relative_to(root).parts:add('q4fit','evidence_script',p/'q4evidence.py')
            if 'run-artifact-comparison.py' in files:add('quantbench','native',p/'run-artifact-comparison.py')
            if (p/'conceptladder'/'measurement.py').is_file():add('conceptladder','native',p)
            if 'budgetfit.pyz' in files and fingerprint(p/'budgetfit.pyz','evaluator')['sha256']==adaptive.PIN:
                for tool in ('budgetfit','trustmebro'):add(tool,'evaluator',p/'budgetfit.pyz')
            if (p/'src'/'llama_autotune').is_dir():add('llama-autotune','autotune_source',p)
            if 'healer.py' in files and ('healer' in p.name.lower() or 'diagnose.py' in files):add('gguf-healer','native',p/'healer.py')
            if 'fm.py' in files and ('first-mover' in p.name.lower() or 'exl3' in p.name.lower()):add('exl3-first-mover','native',p/'fm.py')
            if 'convert.py' in files and (p/'exllamav3'/'architecture').is_dir():
                add('exl3-first-mover','backend',p)
                for base in (p,p.parent):
                    add('exl3-first-mover','python',base/'.venv/bin/python')
            if (p/'hereticbench'/'grading.py').is_file():add('hereticbench','native',p)
            if (p/'gguf-py'/'gguf').is_dir():
                for tool in ('q4fit','gguf-healer'):add(tool,'gguf_root',p/'gguf-py')
            if 'llama-quantize' in files:add('q4fit','quantizer',p/'llama-quantize')
            if 'llama-server' in files:add('llama-autotune','runtime',p/'llama-server')
    explicit={}
    for value in overrides:
        key,sep,path=value.partition('=')
        tool,dot,field=key.partition('.')
        if not sep or not dot or tool not in workspace.TOOLS or field not in workspace.TOOLS[tool][0].keys()|workspace.OPTIONAL.get(tool,set()) or not path:raise ValueError('Expected --set tool.field=PATH')
        if (tool,field) in explicit:raise ValueError('Repeated explicit field')
        p=Path(path).expanduser()
        kind=workspace.TOOLS[tool][0].get(field,'executable' if field in ('python','runtime') else 'file' if field=='evidence_script' else 'directory')
        if not (p.is_dir() if kind=='directory' else p.is_file() and (kind!='executable' or os.access(p,os.X_OK))):raise ValueError('Invalid explicit path type: '+key)
        explicit[tool,field]=os.path.abspath(p) if field=='python' else str(p.resolve())
    config={};ambiguous=[]
    for tool,field in sorted(set(candidates)|set(explicit)):
        choices=candidates.get((tool,field),set())
        if (tool,field) in explicit:value=explicit[tool,field]
        elif len(choices)==1:value=next(iter(choices))
        else:
            ambiguous.append({'field':tool+'.'+field,'candidates':sorted(choices)})
            continue
        config.setdefault(tool,{})[field]=value
    return config,ambiguous

def init(roots,overrides,out,depth):
    if not roots and not overrides:raise ValueError('Specify trusted --root directories or --set paths')
    config,ambiguous=discover(roots,overrides,depth)
    path=Path(out).expanduser()
    evaluation.write_new(path,{'tools':config})
    return {'out':str(path),'ambiguous':ambiguous,**workspace.catalog(config),'scope':'Path discovery only; inspect generated config before use. No tools executed or installed.'}

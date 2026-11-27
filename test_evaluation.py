import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evaluation
from registry import fingerprint

class AdapterTests(unittest.TestCase):
    def test_matrix_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            dataset=p/'tasks.json';dataset.write_text('{}')
            archive=p/'eval.pyz';archive.write_text('fixture')
            plan={'inputs':[fingerprint(dataset,'dataset'),fingerprint(archive,'evaluator')],
                  'workload':{'budgets':[32],'strategies':['aware']}}
            row={'task_id':'t','max_tokens':32,'strategy':'aware','prompt':'Question','request_id':'r'}
            with patch.object(evaluation,'worker',return_value={'requests':[row]}):
                self.assertTrue(evaluation.validate_schedule(plan,[row]))
                with self.assertRaises(ValueError):evaluation.validate_schedule(plan,[{**row,'prompt':'altered'}])
                with self.assertRaises(ValueError):evaluation.validate_schedule(plan,[row,row])
            dataset.write_text('{"changed":true}')
            with self.assertRaises(ValueError):evaluation.validate_schedule(plan,[row])
    def test_write_new(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'out.json'
            evaluation.write_new(p,{'a':1})
            with self.assertRaises(FileExistsError):evaluation.write_new(p,{'a':2})
            self.assertEqual(json.loads(p.read_text()),{'a':1})
    @unittest.skipUnless(os.getenv('BUDGETFIT_TEST_ARCHIVE'),'Set BUDGETFIT_TEST_ARCHIVE to run native integration')
    def test_native_ladder_and_core(self):
        archive=str(Path(os.environ['BUDGETFIT_TEST_ARCHIVE']).resolve())
        code='import sys,json;sys.path.insert(0,sys.argv[1]);import measurement as m;print(json.dumps(m.generate(608,1,"all",1)))'
        m=json.loads(subprocess.check_output([sys.executable,'-I','-c',code,archive],text=True))
        data={'manifest':m,'budgets':[256],'strategies':['aware','native_strict']}
        s=evaluation.worker(archive,{'action':'prepare',**data})
        tasks={t['id']:t for t in m['tasks']}
        rows=[]
        for n,q in enumerate(s['requests']):
            gold=tasks[q['task_id']]['answer']
            answer=json.dumps(gold,separators=(',',':')) if isinstance(gold,dict) else str(gold)
            content=answer if q['strategy']=='native_strict' else 'FINAL: '+answer
            rows.append({'ordinal':n,'request':{**q,'body':{'messages':[{'role':'user','content':q['prompt']}]}},
                         'raw':{'choices':[{'message':{'content':content},'finish_reason':'stop'}],'usage':{'completion_tokens':5}},'elapsed_s':.1})
        d={'action':'score',**data,'rows':rows}
        result=evaluation.worker(archive,d)
        self.assertEqual(len(result['rows']),len(rows))
        self.assertEqual({x['track'] for x in result['profiles'][0]['tracks']},{'core','ladder'})
        self.assertEqual(result['profiles'][0]['prompt_contract'],'BudgetFit-modified TrustMeBro')
        self.assertEqual(result['profiles'][1]['prompt_contract'],'native TrustMeBro')
        bad=copy.deepcopy(d);bad['rows'][0]['request']['prompt']='altered'
        with self.assertRaisesRegex(ValueError,'Native prompt mismatch'):evaluation.worker(archive,bad)
        bad=copy.deepcopy(d);bad['rows'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete evaluation matrix'):evaluation.worker(archive,bad)

if __name__=='__main__':unittest.main()

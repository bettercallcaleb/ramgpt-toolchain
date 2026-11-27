import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import evidence
import native_bridge as nb
import cross_format as c
import workspace
import registry
import adaptive

class CrossFormatTests(unittest.TestCase):
    def test_no_execute_without_flag(self):
        with self.assertRaisesRegex(ValueError,'Explicit --execute'):
            c.run(None,'a','b','c','d','e','f','g','h',[256],['allocated'],64,4096,8,2048)
    def test_native_rows_validation(self):
        q={'request_id':'r','task_id':'t','strategy':'allocated','prompt':'test','max_tokens':128}
        r={'request_id':'r','elapsed_s':.1,'raw':{'model':'exl3','choices':[{'message':{'content':'42'},'finish_reason':'stop'}],'usage':{'prompt_tokens':10,'completion_tokens':2,'total_tokens':12}}}
        with patch('evaluation.worker',return_value={'profiles':[]}) as worker:
            c.score_rows('a',{},[q],[r],[128],['allocated'],64)
            self.assertEqual(worker.call_args.args[1]['rows'][0]['request']['body']['messages'],[{'role':'user','content':'test'}])
            with self.assertRaisesRegex(ValueError,'Incomplete'):c.score_rows('a',{},[q],[],[128],['allocated'],64)
            with self.assertRaisesRegex(ValueError,'order mismatch'):c.score_rows('a',{},[q],[{**r,'request_id':'bad'}],[128],['allocated'],64)
            r['raw']['usage']['completion_tokens']=129
            with self.assertRaisesRegex(ValueError,'usage'):c.score_rows('a',{},[q],[r],[128],['allocated'],64)
    def test_namespace_catalog_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);s=evidence.Store(p/'e.sqlite');reg=registry.Registry(p/'r.sqlite')
            try:
                aid='a'*64
                nb.save(s,'exl3-first-mover',{'format':'exl3','artifact_id':aid,'repo_id':None,'source_revision':None},{'artifact_bytes':100,'declared_bits':4})
                profile={'budgetfit':{'budget':128,'strategy':'allocated','reserve':64,'accuracy':.75}}
                manifest={'tasks':[]};sched=[{'request_id':'r','task_id':'t','strategy':'allocated','prompt':'test','max_tokens':128}]
                body={'kind':'shared-native-evaluation','artifact_bytes':100,'context':4096,'requests':1,'gpu_name':'fixture','profiles':[profile],'manifest_sha256':registry.digest(manifest),'schedule_sha256':registry.digest(sched),'evaluator_sha256':'b'*64,'adapter_sha256':'c'*64}
                r=nb.save(s,'exl3-first-mover',{'format':'exl3','artifact_id':aid},body)
                self.assertEqual(workspace.model_info(reg,s,'exl3:'+aid)['runtime_evidence'][0]['report_id'],r['report_id'])
                self.assertEqual(len(workspace.models(reg,s)['models']),1)
                with s.db:
                    sid=s.put_blob({'manifest':manifest,'schedule':sched,'artifact_sha256':'d'*64})
                    eid=s.put_blob({'report':{'schema':'ramgpt-evaluation-v1','evaluator_sha256':'b'*64,'adapter_sha256':'c'*64,'profiles':[profile]}})
                with patch.object(s,'find',return_value=[{'evaluation_id':eid,'source_id':sid}]):
                    result=c.compare(s,eid,r['report_id'])
                    self.assertEqual(result['profiles'][0]['gguf'],result['profiles'][0]['exl3'])
                    bad=nb.save(s,'exl3-first-mover',{'format':'exl3','artifact_id':aid},{**body,'schedule_sha256':'f'*64})
                    with self.assertRaisesRegex(ValueError,'Different native task'):c.compare(s,eid,bad['report_id'])
            finally:s.close();reg.close()

if __name__=='__main__':unittest.main()

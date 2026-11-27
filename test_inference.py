import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import inference as i
import runtime
from registry import digest

IDENTITY={'model':'m','system_fingerprint':'test-fingerprint'}
CID='a'*64
ROWS=[{'request_id':f'r{n}','task_id':f't{n}','strategy':'aware','prompt':f'Question {n}','max_tokens':32,
       'body':{'model':'m','messages':[{'role':'user','content':f'Question {n}'}],'max_tokens':32}} for n in range(4)]

def response(content='42'):
    return {**IDENTITY,'choices':[{'message':{'content':content},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':5,'completion_tokens':2,'total_tokens':7}}

class FakeRegistry:
    def contract(self,cid):return {'contract_id':cid}
    def verify(self,cid):return {'status':'inputs_verified'}

class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.out=Path(self.tmp.name)/'run'
        self.calls=[]
        self.reg=FakeRegistry()
    def tearDown(self):
        self.tmp.cleanup()
    def session(self,reg,cid,out,timeout=90,work=None):
        out.mkdir()
        def send(body,timeout):
            self.calls.append(body)
            return response(),.1
        result=work(send,IDENTITY)
        receipt={'contract_id':cid,'status':'runtime_session_passed','response_identity':IDENTITY,'workload_result':result,
                 **{k:True for k in ('input_precheck','input_postcheck','smoke_pass','template_probe_match','cleanup_complete')}}
        runtime.atomic(out/'runtime_receipt.json',receipt)
        return receipt
    def test_resume_and_no_duplicate(self):
        with patch.object(i,'schedule',return_value=ROWS),patch.object(runtime,'smoke',side_effect=self.session):
            self.assertEqual(i.run(self.reg,CID,self.out,2)['status'],'paused')
            self.assertEqual(i.run(self.reg,CID,self.out)['committed'],4)
            self.assertEqual(i.run(self.reg,CID,self.out)['new_requests'],0)
        self.assertEqual(self.calls,[x['body'] for x in ROWS])
    def test_tamper_rejected(self):
        with patch.object(i,'schedule',return_value=ROWS),patch.object(runtime,'smoke',side_effect=self.session):
            i.run(self.reg,CID,self.out,2)
            p=self.out/'session-000000/responses.json'
            data=json.loads(p.read_text());data[0]['raw']['choices'][0]['message']['content']='edited'
            runtime.atomic(p,data)
            with self.assertRaisesRegex(ValueError,'seal mismatch'):i.run(self.reg,CID,self.out)
        self.assertEqual(len(self.calls),2)
    def test_uncommitted_never_retries(self):
        def fail(reg,cid,out,timeout=90,work=None):
            out.mkdir()
            def send(body,timeout):
                self.calls.append(body);raise TimeoutError('ambiguous timeout')
            work(send,IDENTITY)
        with patch.object(i,'schedule',return_value=ROWS),patch.object(runtime,'smoke',side_effect=fail):
            with self.assertRaises(TimeoutError):i.run(self.reg,CID,self.out)
            self.assertTrue((self.out/'session-000000/pending.json').exists())
            self.assertTrue((self.out/'session-000000/error.json').exists())
            with self.assertRaisesRegex(ValueError,'Uncommitted session'):i.run(self.reg,CID,self.out)
        self.assertEqual(len(self.calls),1)
    def test_backend_change(self):
        with patch.object(i,'schedule',return_value=ROWS),patch.object(runtime,'smoke',side_effect=self.session):
            i.run(self.reg,CID,self.out,2)
        def changed(reg,cid,out,timeout=90,work=None):
            out.mkdir()
            work(lambda *_:self.fail('must not send'),{**IDENTITY,'system_fingerprint':'different'})
        with patch.object(i,'schedule',return_value=ROWS),patch.object(runtime,'smoke',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'Backend identity changed'):i.run(self.reg,CID,self.out)
    def test_response_checks(self):
        self.assertEqual(i.check_response(response(),ROWS[0],IDENTITY)['completion_tokens'],2)
        for field,value in [('completion_tokens',33),('total_tokens',10),('prompt_tokens',True)]:
            raw=response();raw['usage'][field]=value
            with self.assertRaises(ValueError):i.check_response(raw,ROWS[0],IDENTITY)
        for key,value in [('reasoning_content','hidden'),('tool_calls',[{}]),('content','<think>hi</think>42')]:
            raw=response();raw['choices'][0]['message'][key]=value
            with self.assertRaises(ValueError):i.check_response(raw,ROWS[0],IDENTITY)
        raw=response();raw['choices'][0]['finish_reason']='length'
        self.assertEqual(i.check_response(raw,ROWS[0],IDENTITY)['finish_reason'],'length')

if __name__=='__main__':unittest.main()

import copy
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import runtime

UUID='GPU-00000000-0000-4000-8000-000000000001'

class RuntimeTests(unittest.TestCase):
    def test_lease_conflict_and_release(self):
        test_uuid="GPU-"+str(uuid.uuid4())
        with runtime.lease(test_uuid):
            with self.assertRaises(ValueError):
                with runtime.lease(test_uuid):pass
        with runtime.lease(test_uuid):pass
    def test_owned_listener(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1',0));s.listen()
            port=s.getsockname()[1]
            self.assertTrue(runtime.owns_listener(os.getpid(),port))
            self.assertFalse(runtime.owns_listener(99999999,port))
        self.assertFalse(runtime.owns_listener(os.getpid(),port))
    def test_usage(self):
        raw={'choices':[{'finish_reason':'stop','message':{'content':'42'}}],'usage':{'completion_tokens':2}}
        self.assertEqual(runtime.check_usage(raw,64)['completion_tokens'],2)
        for key,value in [('content','43'),('reasoning_content','hidden'),('tool_calls',[{}])]:
            bad=copy.deepcopy(raw);bad['choices'][0]['message'][key]=value
            with self.assertRaises(ValueError):runtime.check_usage(bad,64)
        bad=copy.deepcopy(raw);bad['usage']['completion_tokens']=65
        with self.assertRaises(ValueError):runtime.check_usage(bad,64)
    def test_stop_owned(self):
        p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
        try:
            runtime.stop_owned(p);self.assertIsNotNone(p.poll())
        finally:
            if p.poll() is None:p.kill();p.wait()
    def test_validate_unsafe_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'llama-server';exe.write_text('fixture');exe.chmod(0o700)
            model=str(Path(tmp)/'model.gguf')
            args=['-m',model,'--alias','model','--host','127.0.0.1','--port','18387','-np','1','--reasoning','off','--jinja','--no-webui']
            plan={'inputs':[{'role':'model','path':model},{'role':'runtime','path':str(exe)},{'role':'template','path':'template'}],'execution':{'runtime_args':args,'reasoning_policy':'off','gpu_uuids':[UUID]}}
            self.assertEqual(runtime.validate(plan)[2],18387)
            for extra in [['--host','0.0.0.0'],['--model-url','https://example.org'],['--chat-template','other']]:
                bad=copy.deepcopy(plan);bad['execution']['runtime_args']+=extra
                with self.assertRaises(ValueError):runtime.validate(bad)
            bad=copy.deepcopy(plan);bad['execution']['sampling']={'model':'other'}
            with self.assertRaises(ValueError):runtime.validate(bad)
            bad=copy.deepcopy(plan);bad['execution']['sampling']={'temperature':0}
            with self.assertRaises(ValueError):runtime.validate(bad)
            bad=copy.deepcopy(plan);bad['execution']['runtime_args'][1]='different.gguf'
            with self.assertRaises(ValueError):runtime.validate(bad)

if __name__=='__main__':unittest.main()

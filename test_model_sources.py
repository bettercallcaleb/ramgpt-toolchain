import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import model_sources as s
from registry import Registry

class ModelSourcesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.reg=Registry(self.root/'reg.sqlite')
        self.data=b'GGUF'+struct.pack('<IQQ',3,1,0)+b'fixture'
        self.sha=hashlib.sha256(self.data).hexdigest()
        self.revision='a'*40;self.cache=self.root/'cache'
        self.meta={'id':'owner/model','sha':self.revision,'private':False,'gated':False,'siblings':[{'rfilename':'model.gguf','size':len(self.data),'lfs':{'size':len(self.data),'sha256':self.sha}},{'rfilename':'model-00001-of-00002.gguf','size':len(self.data),'lfs':{'size':len(self.data),'sha256':self.sha}},{'rfilename':'README.md'}]}
    def tearDown(self):
        self.reg.close();self.tmp.cleanup()
    def response(self):
        return io.BytesIO(json.dumps(self.meta).encode())
    def test_metadata_pins_and_excludes_shards(self):
        with patch.object(s.urllib.request,'urlopen',return_value=self.response()) as get:
            r=s.metadata('owner/model')
        self.assertEqual(r['revision'],self.revision)
        self.assertEqual(len(r['files']),1)
        self.assertIn('/resolve/'+self.revision+'/',r['files'][0]['url'])
        self.assertIn('/revision/main?',get.call_args.args[0])
        with patch.object(s.urllib.request,'urlopen',return_value=self.response()):
            with self.assertRaisesRegex(ValueError,'revision mismatch'):s.metadata('owner/model','b'*40)
    def test_lock_preview_download_cache_and_tune_resolution(self):
        lock=self.root/'model.lock.json'
        with patch.object(s.urllib.request,'urlopen',return_value=self.response()) as get:
            r=s.fetch(self.reg,'owner/model','model.gguf',None,None,None,self.cache,lock_out=lock)
            self.assertEqual(r['new_bytes'],0)
            self.assertFalse(self.cache.exists())
            get.assert_called_once()
        with patch.object(s.urllib.request,'urlopen',return_value=io.BytesIO(self.data)) as get:
            r=s.fetch(self.reg,None,None,None,None,None,self.cache,True,lock=lock)
            self.assertEqual(r['new_bytes'],len(self.data))
            self.assertEqual(s.cached(lock,self.cache),r['path'])
            self.assertEqual(s.fetch(self.reg,None,None,None,None,None,self.cache,True,lock=lock)['new_bytes'],0)
            get.assert_called_once()
        d=json.loads(lock.read_text());d['source']['bytes']+=1;lock.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'integrity failure'):s.cached(lock,self.cache)
    def test_cap_and_partial_explicit_metadata(self):
        lock=self.root/'lock.json'
        with patch.object(s.urllib.request,'urlopen',return_value=self.response()):
            with self.assertRaisesRegex(ValueError,'download cap'):s.fetch(self.reg,'owner/model','model.gguf',None,None,None,self.cache,True,lock_out=lock,max_bytes=24)
        self.assertFalse(lock.exists())
        self.assertFalse(self.cache.exists())
        with patch.object(s.urllib.request,'urlopen') as get:
            with self.assertRaisesRegex(ValueError,'together'):s.fetch(self.reg,'owner/model','model.gguf',None,self.sha,None,self.cache)
            r=s.fetch(self.reg,'owner/model','model.gguf',self.revision,self.sha,len(self.data),self.cache)
            self.assertEqual(r['status'],'model_download_preview')
            get.assert_not_called()
    def test_private_gated_and_bad_repository(self):
        for key in ('private','gated'):
            self.meta[key]=True
            with patch.object(s.urllib.request,'urlopen',return_value=self.response()):
                with self.assertRaisesRegex(ValueError,'public ungated'):s.metadata('owner/model')
            self.meta[key]=False
        self.meta['id']='other/model'
        with patch.object(s.urllib.request,'urlopen',return_value=self.response()):
            with self.assertRaisesRegex(ValueError,'identity mismatch'):s.metadata('owner/model')

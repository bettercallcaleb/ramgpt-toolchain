import hashlib
import io
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import model_cache as m
from registry import Registry

class ModelCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.reg=Registry(self.root/'reg.sqlite')
        self.data=b'GGUF'+struct.pack('<IQQ',3,1,0)+b'fixture'
        self.sha=hashlib.sha256(self.data).hexdigest()
        self.cache=self.root/'cache'
    def tearDown(self):
        self.reg.close();self.tmp.cleanup()
    def test_local_dedup_and_tamper(self):
        p=self.root/'input.gguf';p.write_bytes(self.data)
        self.assertEqual(m.add(self.reg,p,self.cache)['new_bytes'],0)
        self.assertFalse(self.cache.exists())
        r=m.add(self.reg,p,self.cache,True)
        self.assertEqual(r['new_bytes'],len(self.data))
        self.assertEqual(m.add(self.reg,p,self.cache,True)['new_bytes'],0)
        self.assertEqual(m.resolve(self.sha,self.cache),r['path'])
        Path(r['path']).write_bytes(self.data+b'tamper')
        with self.assertRaisesRegex(ValueError,'digest mismatch'):m.lookup(self.cache,self.sha)
    def test_remote_exact_identity_and_cache(self):
        args=(self.reg,'owner/model','a'*40,'model.gguf',self.sha,len(self.data),self.cache)
        with patch.object(m.urllib.request,'urlopen',return_value=io.BytesIO(self.data)) as get:
            self.assertEqual(m.fetch(*args)['status'],'model_download_preview')
            get.assert_not_called()
            self.assertEqual(m.fetch(*args,True)['status'],'cached_verified')
            self.assertEqual(m.fetch(*args,True)['status'],'cache_hit')
            get.assert_called_once()
        for repo,rev,name in [('owner/model','main','model.gguf'),('owner/model','a'*40,'../model.gguf'),('owner/model','a'*40,'model-00001-of-00002.gguf')]:
            with self.assertRaises(ValueError):m.identity(repo,rev,name,self.sha,len(self.data))
    def test_hash_size_failures_do_not_publish(self):
        for data in (self.data+b'x',self.data[:-1],self.data[:-1]+b'x'):
            with self.assertRaises(ValueError):m.place(self.reg,self.cache,self.sha,len(self.data),lambda:io.BytesIO(data))
            self.assertFalse(m.target(self.cache,self.sha).exists())

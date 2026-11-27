import os
import unittest
from pathlib import Path
import adaptive
from registry import digest

class RouterTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(adaptive.thresholds(adaptive.DEFAULT),adaptive.DEFAULT)
        for patch in ({'min_bucket':1},{'min_savings':0},{'tolerance':float('nan')}):
            with self.assertRaises(ValueError):adaptive.thresholds({**adaptive.DEFAULT,**patch})

    @unittest.skipUnless(os.environ.get('BUDGETFIT_TEST_ARCHIVE'),'native archive not configured')
    def test_native_sparse(self):
        archive=os.environ['BUDGETFIT_TEST_ARCHIVE']
        m=adaptive.worker(archive,{'action':'generate','seed':610,'variants':1,'suite':'core','ladder_variants':1})
        lengths=[len(t['prompt'].encode()) for t in m['tasks']]
        p={'version':'budgetfit-router-v1','feature':'prompt_utf8_bytes','threshold':sorted(lengths)[3],'min_length':min(lengths),'max_length':max(lengths),'short':[1024,'allocated',64],'long':[256,'allocated',64],'fallback':[1024,'allocated',64]}
        d={'schema':adaptive.SCHEMA,'adapter_sha256':adaptive.adapter_hash(),'manifest_digest':digest(m),'validation_seed':610,'thresholds':adaptive.DEFAULT,'selection':{'policy':p,'fixed':{'budget':1024,'strategy':'allocated','reserve':64}}}
        rows=adaptive.prepare(archive,m,d)['requests']
        self.assertEqual(len(rows),8+sum(x>p['threshold'] for x in lengths))
        self.assertEqual(len({(x['task_id'],x['max_tokens']) for x in rows}),len(rows))
        with self.assertRaises(ValueError):adaptive.prepare(archive,m,{**d,'validation_seed':611})
        with self.assertRaises(ValueError):adaptive.prepare(archive,m,{**d,'adapter_sha256':'bad'})
        with self.assertRaises(ValueError):adaptive.worker(archive,{'action':'score','manifest':m,'selection':d['selection'],'rows':[],'reserve':64,'thresholds':adaptive.DEFAULT})

    def test_archive_pin(self):
        with self.assertRaises(ValueError):adaptive.worker(Path(__file__),{})

if __name__=='__main__':unittest.main()

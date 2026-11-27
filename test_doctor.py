import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import doctor

class DoctorTests(unittest.TestCase):
    def test_default_does_not_query_gpu(self):
        with patch('doctor.subprocess.run') as run:
            d=doctor.check()
            run.assert_not_called()
        self.assertTrue(d['core_supported'])
        self.assertEqual(d['status'],'checks_passed')
        self.assertIsNone(d['gpu_query']['devices'])

    def test_bad_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'bad.pyz';p.write_text('untrusted')
            r=doctor.check(evaluator=p,model=p)
            self.assertEqual(r['status'],'checks_failed')
            self.assertFalse(r['supplied_inputs']['evaluator']['pinned_budgetfit_match'])
            self.assertFalse(r['supplied_inputs']['model']['valid'])

    def test_gpu_query(self):
        from subprocess import CompletedProcess
        with patch('doctor.shutil.which',return_value='/test/nvidia-smi'),patch('doctor.subprocess.run',return_value=CompletedProcess([],0,'Test GPU, 24564, 590.48.01\n','')) as run:
            d=doctor.check(gpu=True)
        self.assertEqual(d['gpu_query']['devices'][0]['total_mib'],24564)
        self.assertEqual(d['status'],'checks_passed')
        run.assert_called_once()

    def test_missing_gpu_fails_explicit_request(self):
        with patch('doctor.shutil.which',return_value=None):
            self.assertEqual(doctor.check(gpu=True)['status'],'checks_failed')

if __name__=='__main__':unittest.main()

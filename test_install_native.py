import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import install_native as n

class NativeSetupTests(unittest.TestCase):
    def test_copy_and_hash_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);src=root/'archive';src.write_bytes(b'fixture')
            dest=root/'budgetfit.pyz'
            with patch.object(n.adaptive,'PIN',hashlib.sha256(b'fixture').hexdigest()):
                self.assertEqual(n.fetch(dest,src),7)
            self.assertEqual(dest.read_bytes(),b'fixture')
            with self.assertRaises(FileExistsError):n.fetch(dest,src)
            self.assertEqual(dest.read_bytes(),b'fixture')
            with self.assertRaisesRegex(ValueError,'SHA-256 mismatch'):n.fetch(root/'bad',src)
            self.assertFalse((root/'bad').exists())

    def test_plan_no_download(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'setup'
            with patch.object(n,'runtime_check',return_value={'sha256':'a'*64}),patch.object(n,'fetch') as fetch:
                p=n.setup('llama-server',root)
            self.assertEqual(p['status'],'setup_planned')
            self.assertFalse(root.exists())
            fetch.assert_not_called()

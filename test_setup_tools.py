import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import setup_tools as s
import workspace

class SetupToolsTests(unittest.TestCase):
    def test_discover_and_disambiguate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for x in ('a','b'):
                q=p/x;q.mkdir();(q/'q4fit.py').write_text('raise RuntimeError("not executed")')
            config,ambiguous=s.discover([p])
            self.assertNotIn('q4fit',config)
            self.assertEqual(ambiguous[0]['field'],'q4fit.native')
            out=p/'tools.json'
            result=s.init([p],[f'q4fit.native={p/"a/q4fit.py"}'],out,3)
            self.assertFalse(result['ambiguous'])
            self.assertEqual(workspace.load(out)['q4fit']['native'],str(p/'a/q4fit.py'))
            with self.assertRaises(FileExistsError):s.init([p],[],out,3)
    def test_pinned_archive_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'budgetfit.pyz').write_bytes(b'fake')
            self.assertEqual(s.discover([p]),({},[]))
            with patch('setup_tools.fingerprint',return_value={'sha256':s.adaptive.PIN}):
                self.assertEqual(set(s.discover([p])[0]),{'budgetfit','trustmebro'})
    def test_bounds_and_invalid_override(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            with self.assertRaises(ValueError):s.discover([p],depth=6)
            with self.assertRaises(ValueError):s.discover([p],[f'q4fit.unknown={p}'])

if __name__=='__main__':unittest.main()

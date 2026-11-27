import hashlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import adaptive
import supply
import workspace

class SupplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def archive(self,name,entries):
        p=self.root/name
        with tarfile.open(p,'w:gz') as t:
            for path,content in entries:
                info=tarfile.TarInfo(path)
                if content is None:info.type=tarfile.SYMTYPE;info.linkname='/etc/passwd';t.addfile(info)
                else:info.size=len(content);t.addfile(info,io.BytesIO(content))
        return p
    def test_unpack_rejects_escape_link_duplicate(self):
        for i,entries in enumerate([[('root/../escape',b'x')],[('root/link',None)],[('root/x',b'a'),('root/x',b'b')]]):
            with self.assertRaises(ValueError):supply.unpack(self.archive(f'{i}.tar.gz',entries),self.root/f'out{i}')
        self.assertFalse((self.root/'escape').exists())
    def test_verified_bundle_relocation_and_tamper(self):
        arc=self.archive('fixture.tar.gz',[('root/q4fit.py',b'pass\n'),('root/q4evidence.py',b'pass\n')])
        evaluator=self.root/'budgetfit.pyz';evaluator.write_bytes(b'fixture evaluator')
        packs={'q4fit':('fixture','a'*40,supply.sha(arc),{'native':'q4fit.py','evidence_script':'q4evidence.py'})}
        bundle=self.root/'bundle'
        with patch.object(supply,'PACKS',packs),patch.object(adaptive,'PIN',supply.sha(evaluator)):
            self.assertEqual(supply.install(bundle,archive_dir=self.root)['status'],'supply_planned')
            self.assertFalse(bundle.exists())
            r=supply.install(bundle,archive_dir=self.root,execute=True)
            self.assertEqual(r['status'],'supply_verified')
            moved=self.root/'moved';bundle.rename(moved)
            cfg=workspace.load(moved)
            self.assertEqual(cfg['q4fit']['native'],str(moved/'sources/q4fit/q4fit.py'))
            (moved/'sources/q4fit/q4fit.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'source changed'):workspace.load(moved)
    def test_bad_archive_no_partial_bundle(self):
        self.archive('fixture.tar.gz',[('root/q4fit.py',b'x')])
        with patch.object(supply,'PACKS',{'q4fit':('fixture','a'*40,'b'*64,{'native':'q4fit.py'})}):
            with self.assertRaisesRegex(ValueError,'digest mismatch'):supply.install(self.root/'bundle',archive_dir=self.root,execute=True)
        self.assertFalse((self.root/'bundle').exists())

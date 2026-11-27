import io
import tarfile
import tempfile
import unittest
from pathlib import Path
import runtime_supply as r

class RuntimeSupplyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def archive(self,entries):
        p=self.root/'pack.tar.gz'
        with tarfile.open(p,'w:gz') as t:
            for name,kind,value in entries:
                m=tarfile.TarInfo(name)
                if kind=='link':
                    m.type=tarfile.SYMTYPE;m.linkname=value;t.addfile(m)
                elif kind=='dir':
                    m.type=tarfile.DIRTYPE;t.addfile(m)
                else:
                    m.mode=0o755;m.size=len(value);t.addfile(m,io.BytesIO(value))
        return p
    def test_valid_library_aliases_and_integrity(self):
        p=self.archive([('root/lib.so','link','lib.so.1'),('root/lib.so.1','link','lib.so.1.0'),('root/lib.so.1.0','file',b'lib'),('root/llama-server','file',b'bin')])
        out=self.root/'bin';out.mkdir()
        m=r.manifest(p,dest=out)
        r.verify_tree(out,m)
        self.assertEqual((out/'lib.so').read_bytes(),b'lib')
        (out/'llama-server').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'files changed'):r.verify_tree(out,m)
    def test_unsafe_runtime_archive(self):
        cases=[
            [('root/../bad','file',b'x')],
            [('root/lib.so','link','/etc/passwd')],
            [('root/lib.so','link','../bad')],
            [('root/lib.so','link','lib2.so'),('root/lib2.so','link','lib.so')],
            [('root/lib.so','link','missing')],
            [('root/bin','file',b'x'),('root/bin','file',b'y')],
            [('root/nested/file','file',b'x')],
        ]
        for entries in cases:
            with self.subTest(entries=entries):
                with self.assertRaises(ValueError):r.manifest(self.archive(entries))
    def test_source_subset(self):
        p=self.archive([('root/gguf-py/gguf','dir',None),('root/gguf-py/gguf/__init__.py','file',b'pass'),('root/other/file','file',b'ignored')])
        out=self.root/'source';out.mkdir()
        m=r.manifest(p,source=True,dest=out)
        self.assertEqual(set(m),{'gguf','gguf/__init__.py'})
        r.verify_tree(out,m)
    def test_preview_does_not_install(self):
        out=self.root/'runtime'
        self.assertEqual(r.install(out)['status'],'runtime_supply_preview')
        self.assertFalse(out.exists())
        self.assertEqual(r.plan()['download_bytes'],sum(x['bytes'] for x in r.ARCHIVES.values()))

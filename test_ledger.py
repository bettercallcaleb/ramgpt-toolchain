import unittest
import ledger

class Tests(unittest.TestCase):
    def test_no_alias_join(self):
        a={'artifact_sha256':'a'*64,'producer':'quantbench'}
        b={'artifact_sha256':None,'producer':'budgetfit','model_alias':'same'}
        z=ledger.assemble([a,b])
        self.assertEqual(len(z['artifact_groups']),1)
        self.assertEqual(len(z['unbound_evidence']),1)
        self.assertEqual(z['cross_tool_matches'],[])
    def test_exact_hash_join(self):
        a={'artifact_sha256':'a'*64,'producer':'quantbench'}
        b={'artifact_sha256':'a'*64,'producer':'trustmebro'}
        self.assertEqual(ledger.assemble([a,b])['cross_tool_matches'],['a'*64])
    def test_bad_hash(self):
        with self.assertRaises(ValueError):ledger.identity('Q4_K_M')

if __name__=='__main__':unittest.main()

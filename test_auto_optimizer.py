import os
import unittest
from unittest.mock import patch
from pathlib import Path
import auto_optimizer as a
import runtime
import registry
import test_workflow as tw

@unittest.skipUnless(os.getenv('BUDGETFIT_TEST_ARCHIVE'),'Pinned native archive required')
class AutoOptimizerTests(unittest.TestCase):
    setUp=tw.WorkflowTests.setUp
    tearDown=tw.WorkflowTests.tearDown
    fake_session=tw.WorkflowTests.fake_session

    def run_auto(self,execute=False,accuracy=0,requests=36,tokens=8448):
        return a.run(self.reg,self.store,self.root/'fixture.gguf',self.spec,self.root/'auto',[256],['f16','q8_0'],719,720,1,64,'1000B','100B','memory',accuracy,requests,tokens,execute)

    def test_global_cap_before_output(self):
        with self.assertRaisesRegex(ValueError,'Request cap exceeded'):self.run_auto(requests=35)
        with self.assertRaisesRegex(ValueError,'Completion token cap exceeded'):self.run_auto(tokens=8447)
        self.assertFalse((self.root/'auto').exists())

    def test_end_to_end_and_idempotence(self):
        with patch.object(runtime,'smoke',side_effect=self.fake_session):
            p=self.run_auto()
            self.assertEqual(p['status'],'auto_plan_frozen')
            self.assertEqual(len(self.calls),0)
            r=self.run_auto(True)
            again=self.run_auto(True)
        self.assertEqual(r['status'],'heldout_pilot_passed')
        self.assertEqual(r['new_inference_requests'],32)
        self.assertEqual(r['new_smoke_requests'],4)
        self.assertEqual(again['new_inference_requests'],0)
        self.assertEqual(again['new_smoke_requests'],0)
        self.assertEqual(len(self.calls),32)
        self.assertFalse(r['deployment_authorized'])
        profile=registry.read_json(Path(r['selected_profile']))
        self.assertEqual(profile['generation_budget'],256)
        self.assertIn('q8_0',profile['runtime_args'])
        with self.assertRaisesRegex(ValueError,'parameters changed'):self.run_auto(accuracy=.5)

    def test_no_eligible_profile(self):
        with patch.object(runtime,'smoke',side_effect=self.fake_session):
            r=self.run_auto(True,accuracy=1)
        self.assertEqual(r['status'],'no_holdout')
        self.assertEqual(r['new_inference_requests'],16)
        self.assertEqual(r['new_smoke_requests'],2)
        self.assertIsNone(r['selected_profile'])
        self.assertFalse((self.root/'auto/selected-profile.json').exists())

    def test_failed_holdout_has_no_profile(self):
        self.fail_holdout=True
        with patch.object(runtime,'smoke',side_effect=self.fake_session):
            r=self.run_auto(True)
        self.assertEqual(r['status'],'heldout_failed')
        self.assertIsNone(r['selected_profile'])
        self.assertIn('objective_not_improved',r['failures'])

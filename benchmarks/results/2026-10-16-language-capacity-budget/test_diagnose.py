import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("diagnose",Path(__file__).with_name("diagnose.py"))
diagnose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnose)


class LocalizationTests(unittest.TestCase):
    def test_qkv_bias_boundaries_are_not_flattened_together(self):
        self.assertEqual(diagnose.qkv_parts([0,3,4,7,8,11],4),dict(query=2,key=2,value=2))
        self.assertEqual(diagnose.qkv_parts([4,5,6,7],4),dict(query=0,key=4,value=0))
        with self.assertRaises(ValueError):
            diagnose.qkv_parts([12],4)

    def test_no_floor_hides_small_moment_errors(self):
        criteria = SimpleNamespace(ATOL=3e-6,RTOL=5e-5)
        result = diagnose.errors([0.],[1e-12],criteria)
        self.assertEqual(result["failed_indices"],[])
        self.assertEqual(result["relative_l2_error"],1.)
        self.assertIsNone(diagnose.errors([1.],[0.],criteria)["relative_l2_error"])
        with self.assertRaises(ValueError):
            diagnose.errors([1.],[],criteria)


if __name__ == "__main__":
    unittest.main()

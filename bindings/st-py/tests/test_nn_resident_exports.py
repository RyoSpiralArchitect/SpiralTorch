"""Resident NN public discovery must agree with the native registered classes."""
import unittest

import spiraltorch as st


RESIDENT_EXPORTS = (
    "ResidentGraphAutograd",
    "GraphForward",
    "GraphGradients",
    "ResidentGraphLearner",
    "GraphGradientBatch",
    "GraphGradientAccumulator",
    "GraphUpdateSnapshot",
    "ResidentLoss",
)


class ResidentExports(unittest.TestCase):
    def test_native_and_public_exports_include_resident_learning(self):
        for module in (st._rs.nn, st.nn):
            for name in RESIDENT_EXPORTS:
                with self.subTest(module=module.__name__, name=name):
                    self.assertIn(name, module.__all__)
                    self.assertIs(getattr(module, name), getattr(st._rs.nn, name))

    def test_star_import_retains_native_identity(self):
        namespace = {}
        exec("from spiraltorch.nn import *", namespace)
        for name in RESIDENT_EXPORTS:
            with self.subTest(name=name):
                self.assertIn(name, namespace)
                self.assertIs(namespace[name], getattr(st._rs.nn, name))

    def test_existing_native_names_and_python_helpers_remain_discoverable(self):
        namespace = {}
        exec("from spiraltorch.nn import *", namespace)
        for name in ("Linear", "Sequential", "Gelu", "Relu", "InferencePlan"):
            with self.subTest(name=name):
                self.assertIs(namespace[name], getattr(st._rs.nn, name))
        for name in ("eval_mode", "save", "load"):
            self.assertIs(namespace[name], getattr(st.nn, name))


if __name__ == "__main__":
    unittest.main()

import copy
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("probe", Path(__file__).with_name("probe_byte_corpus_mixture.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class NativeSnapshotGateTests(unittest.TestCase):
    def setUp(self):
        self.case = dict(name="case", seed=11, geometry=True, geometry_update="train",
                         bias_initialization="matched_poincare_rms", mixture={"router_update":"train"},
                         parameters=[dict(name="p",shape=[2],values=[0.,0.])])
        self.request = dict(train_batches=[None]*128,cases=[self.case])
        self.report = dict(schema="spiraltorch.byte_corpus.result.v7",engine="spiraltorch",
                           adapter="Fabricated test adapter { backend: Metal }",request_sha256="a"*64,
                           cases=[{**self.case,"training":[dict(revision=i,ce=1.) for i in range(1,129)],
                                   "final_parameters":[[0.1,-0.1]]}])

    def validate(self, report):
        probe.validate_native_snapshot(self.request,report,"a"*64)

    def test_complete_shape_is_accepted_without_claiming_execution(self):
        self.validate(self.report)

    def test_partial_and_missing_training_revisions_reject(self):
        for length in (0,37,127):
            report=copy.deepcopy(self.report)
            report["cases"][0]["training"]=report["cases"][0]["training"][:length]
            with self.subTest(length=length), self.assertRaises(ValueError): self.validate(report)
        for revision in (0,True,127,129):
            report=copy.deepcopy(self.report)
            report["cases"][0]["training"][-1]["revision"]=revision
            with self.subTest(revision=revision), self.assertRaises(ValueError): self.validate(report)

    def test_foreign_schema_engine_hash_and_adapter_reject(self):
        for key,value in [("schema","spiraltorch.byte_corpus.result.v6"),("engine","independent_pytorch"),
                          ("request_sha256","b"*64),("adapter",""),("adapter",None)]:
            report=copy.deepcopy(self.report); report[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(report)

    def test_browser_backend_is_not_a_native_snapshot(self):
        report=copy.deepcopy(self.report)
        for adapter in ("AdapterInfo { backend: BrowserWebGpu }", "some nonempty text", "AdapterInfo { backend: Empty }"):
            report["adapter"]=adapter
            with self.subTest(adapter=adapter), self.assertRaises(ValueError): self.validate(report)

    def test_missing_or_misidentified_cases_reject(self):
        for rows in ([],None,[self.report["cases"][0]]*2):
            report=copy.deepcopy(self.report); report["cases"]=rows
            with self.subTest(rows=rows), self.assertRaises(ValueError): self.validate(report)
        for key,value in [("name","wrong"),("seed",23),("mixture",{}),("geometry_update","frozen")]:
            report=copy.deepcopy(self.report); report["cases"][0][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError): self.validate(report)

    def test_tensor_lengths_nonfinite_and_unrepresentable_values_reject(self):
        for values in (None,[],[[0.]],[[0.,0.,0.]],[[0.,float("nan")]],[[0.,float("inf")]],[[0.,1e300]],[[False,0.]]):
            report=copy.deepcopy(self.report); report["cases"][0]["final_parameters"]=values
            with self.subTest(values=values), self.assertRaises(ValueError): self.validate(report)

    def test_nonfinite_losses_and_wrong_recipe_reject(self):
        for value in (float("nan"),float("inf"),None,True):
            report=copy.deepcopy(self.report); report["cases"][0]["training"][0]["ce"]=value
            with self.subTest(value=value), self.assertRaises(ValueError): self.validate(report)
        self.request["train_batches"].pop()
        with self.assertRaises(ValueError): self.validate(self.report)


if __name__ == "__main__":
    unittest.main()

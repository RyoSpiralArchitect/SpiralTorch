"""Admission regressions need no GPU, numpy or torch."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import KEYS, admit, close, f32
from analyze import analyze, summarize
from archive import STAGES, files, safe_path, verify, write


def fixture(kind="wgpu"):
    cases = []
    for rows, count, hidden in sorted(KEYS):
        dims = [3, hidden, 4] if hidden else [3, 4]
        parameters = []
        for a, b in zip(dims, dims[1:]):
            parameters.extend([{"role":"weight", "shape":[a,b], "values":[0.]*(a*b)},
                               {"role":"bias", "shape":[b], "values":[0.5]*b}])
        routes = ["staged", "direct"] if kind == "wgpu" else ["cpu", "mps"]
        cases.append({"rays":rows, "samples":count, "hidden":hidden, "seed":17,
                      "ray_inputs":[[0.,0.,0.,0.,0.,1.,0.,1.]]*rows,"parameters":parameters,
                      "reference":[0.5]*(rows*4), "last_outputs":[[0.5]*(rows*4),[0.5]*(rows*4)],
                      "intervals":[{"block":block,"burst":burst,"route":route,"elapsed_ms":1.,"max_abs_error":0.,
                                    "order":[0,1] if (block+3+rows+hidden)%2==0 else [1,0]}
                                   for burst in [1,4] for block in range(9) for route in routes]})
    report = {"schema":"spiraltorch.nerf_direct_bench.v1" if kind=="wgpu" else "spiraltorch.nerf_torch_bench.v1",
              "status":"passed","warmup":3,"blocks":9,"bursts":[1,4],"cases":cases}
    if kind == "wgpu":
        report.update(guard_cases=12,adapter="IntegratedGpu",kernel="register_2x2",accumulation="sequential",
                      page_errors=[],console_messages=[],browser_adapter_probe={"is_fallback_adapter":False},
                      browser_version="fixture",asset_sha256={})
    else:
        report.update(devices=["cpu","mps"],intra_op_threads=4,inter_op_threads=1,compiled=False,
                      thin_alpha="fourth_order_below_0.01",torch_version="fixture")
    return report


class ContractTests(unittest.TestCase):
    def test_complete(self):
        for kind in ["wgpu", "torch"]:
            self.assertEqual(len(admit(fixture(kind),kind)),12)

    def test_missing_duplicate_cases_and_intervals(self):
        for field in ["cases", "intervals"]:
            for duplicate in [False, True]:
                report=fixture()
                data=report["cases"] if field=="cases" else report["cases"][0]["intervals"]
                data.pop() if not duplicate else data.__setitem__(0,copy.deepcopy(data[1]))
                with self.assertRaises(ValueError): admit(report,"wgpu")

    def test_nonfinite_and_forged_success(self):
        for value in [float("nan"),float("inf"),1e40,True]:
            for field in ["reference", "parameters"]:
                report=fixture()
                data=report["cases"][0]["reference"] if field=="reference" else report["cases"][0]["parameters"][0]["values"]
                data[0]=value
                with self.assertRaises(ValueError): admit(report,"wgpu")
        for field,value in [("status","error"),("guard_cases",11),("adapter","Cpu"),("accumulation","unknown")]:
            report=fixture(); report[field]=value
            with self.assertRaises(ValueError): admit(report,"wgpu")

    def test_timing_and_shapes(self):
        for field,value in [("elapsed_ms",0),("elapsed_ms",float("nan")),("elapsed_ms",True),("block",False),("max_abs_error",1),("order",[0,0])]:
            report=fixture(); report["cases"][0]["intervals"][0][field]=value
            with self.assertRaises(ValueError): admit(report,"wgpu")
        for field in ["reference","last_outputs","ray_inputs","parameters"]:
            report=fixture(); report["cases"][0][field].pop()
            with self.assertRaises(ValueError): admit(report,"wgpu")

    def test_numerical_gate_and_signed_zero(self):
        self.assertNotEqual(f32([0.]),f32([-0.]))
        for a,b in [([0.],[1.]),([1.,2.],[1.]),([] ,[])]:
            with self.assertRaises(ValueError): close(a,b)

    def test_full_summary_and_input_drift(self):
        native, browser, torch = [[fixture(kind) for _ in range(3)] for kind in ["wgpu","wgpu","torch"]]
        report=analyze(native,browser,torch)
        self.assertEqual(report["descriptive_summary"]["native"]["geomean_staged_over_direct"],1)
        self.assertEqual(len(report["cases"]),12)
        with self.assertRaises(ValueError): analyze(native[:2],browser,torch)
        browser[0]["cases"][0]["parameters"][0]["values"][0]=0.1
        with self.assertRaises(ValueError): analyze(native,browser,torch)

    def test_browser_errors_and_torch_policy(self):
        for field,value in [("compiled",True),("devices",["cpu"]),("thin_alpha","expm1")]:
            report=fixture("torch"); report[field]=value
            with self.assertRaises(ValueError): admit(report,"torch")
        browser=[fixture() for _ in range(3)]; browser[0]["page_errors"]=["device lost"]
        with self.assertRaises(ValueError): analyze([fixture()]*3,browser,[fixture("torch")]*3)

    def test_browser_unknown_class_keeps_probe_boundary(self):
        report=fixture()
        report["adapter"]="device_type: Other, backend: BrowserWebGpu"
        self.assertEqual(len(admit(report,"wgpu")),12)
        for value in [None, True, 0]:
            report["browser_adapter_probe"]["is_fallback_adapter"]=value
            with self.assertRaises(ValueError): admit(report,"wgpu")
        report["browser_adapter_probe"]["is_fallback_adapter"]=False
        report["adapter"]="device_type: Other, backend: Metal"
        with self.assertRaises(ValueError): admit(report,"wgpu")

    def test_published_timing_rejects_missing_or_duplicate(self):
        with self.assertRaises(ValueError): summarize([])
        report=analyze([fixture()]*3,[fixture()]*3,[fixture("torch")]*3)
        intervals=report["cases"][0]["intervals"]
        intervals[0]=dict(intervals[1])
        with self.assertRaises(ValueError): summarize(intervals)

    def test_archive_fixity_and_resealed_summary_regressions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=analyze([fixture()]*3,[fixture()]*3,[fixture("torch")]*3)
            write(root/"results.json",result)
            write(root/"validation.json",[{"stage":s,"exit_code":0,"source_unchanged":True} for s in STAGES])
            write(root/"manifest.json",files(root))
            self.assertFalse(verify(root)["numerical_reexecution"])
            (root/"results.json").write_text("{}")
            with self.assertRaises(ValueError): verify(root)
            result["descriptive_summary"]["native"]["geomean_staged_over_direct"]=2.
            write(root/"results.json",result); write(root/"manifest.json",files(root))
            with self.assertRaises(ValueError): verify(root)

    def test_archive_paths_are_relative(self):
        for name in ["../escape", "/tmp/absolute", ""]:
            with self.assertRaises(ValueError): safe_path(Path("."),name)


if __name__ == "__main__":
    unittest.main()

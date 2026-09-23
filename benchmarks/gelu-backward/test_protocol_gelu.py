import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol_gelu as p

TEST_KEYS = {(1,3,1),(1,3,3)}


def fixture(family):
    torch = family == "torch"
    result = dict(schema=p.TORCH if torch else p.SCHEMA,status="passed",blocks=9,warmup=3,bursts=[1,4],cases=[])
    if torch:
        result.update(devices=["cpu","mps"],intra_op_threads=4,inter_op_threads=1,compiled=False,
            observation="one_packed_owning_cpu_copy",operation="aten.gelu_backward.grad_input:tanh",torch_version="synthetic")
    else:
        source = Path(__file__).resolve().parents[2]/"crates/st-backend-wgpu/examples/support"
        result.update(adapter="synthetic non-CPU adapter",domains=dict(cases=[
            dict(rows=r,cols=c,count=n,padding=pad,errors=[[0.,0.]]*(2 if n==1 and pad else 3))
            for r,c,n,pad in sorted(p.DOMAIN_KEYS)],
            legacy_backend_syntax_repaired_only=dict(input=[0.]*15,output_strings=["0"]*15,nonfinite_count=0)),
            legacy_tensor_source=(source/"gelu_legacy_tensor.wgsl").read_text(),
            legacy_backend_source=(source/"gelu_legacy_backend.wgsl").read_text())
    if family == "browser":
        result.update(page_errors=[],console_messages=[],asset_sha256={"synthetic.wasm":"0"*64},browser_version="synthetic",
            browser_adapter_probe=dict(is_fallback_adapter=False),
            gelu_state_artifacts=dict(rows=len(TEST_KEYS),bytes=1,path="synthetic.cases.jsonl",sha256="0"*64))
    for rows,cols,count in sorted(TEST_KEYS):
        z,g,r = p.inputs(rows,cols)
        reference = p.oracle(rows,cols,count)
        names = ("cpu","mps") if torch else p.names(count)
        intervals = [dict(block=b,burst=n,route=d,order=p.order(b+3,rows,cols,count,torch),
            elapsed_ms=float(i+1),max_abs_error=0.,max_scaled_error=0.)
            for b in range(9) for n in (1,4) for i,d in enumerate(names)]
        result["cases"].append(dict(rows=rows,cols=cols,count=count,order_scheme="balanced-cycle-v1",
            input=z,seed=g,residual=r,reference=reference,last_outputs=[reference[:]]*len(names),intervals=intervals))
    return result


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch.object(p,"KEYS",TEST_KEYS)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def reports(self):
        return [[fixture(f)]*3 for f in ("native","browser","torch")]

    def test_complete_study_and_every_position(self):
        result = p.analyze(*self.reports())
        p.validate_summary(result)
        self.assertEqual(len(result["cases"]),2)
        for r,c,n in TEST_KEYS:
            for route in range(3):
                self.assertEqual([sum(p.order(b,r,c,n).index(route)==i for b in range(3,12)) for i in range(3)], [3,3,3])

    def test_missing_duplicate_and_wrong_order_reject(self):
        for mutate in [
            lambda r:r["cases"].pop(),
            lambda r:r["cases"][0]["intervals"].pop(),
            lambda r:r["cases"][0]["intervals"].append(r["cases"][0]["intervals"][0]),
            lambda r:r["cases"][0]["intervals"][0].update(order=[0,0,0]),
            lambda r:r["cases"][0].update(order_scheme="legacy"),
            lambda r:r["domains"]["cases"].pop(),
        ]:
            report = fixture("native")
            mutate(report)
            with self.assertRaises(ValueError):
                p.admit(report,"native")

    def test_bad_values_source_and_runtime_reject(self):
        for mutate in [
            lambda r:r["cases"][0]["last_outputs"][0].__setitem__(0,float("nan")),
            lambda r:r["cases"][0]["seed"].__setitem__(0,123.),
            lambda r:r.update(legacy_tensor_source="// wrong source"),
            lambda r:r.update(adapter="device_type: Cpu"),
        ]:
            report = fixture("native")
            mutate(report)
            with self.assertRaises(ValueError):
                p.admit(report,"native")
        for mutate in [
            lambda r:r.update(console_messages=["unexpected"]),
            lambda r:r["gelu_state_artifacts"].update(rows=0),
            lambda r:r["browser_adapter_probe"].update(is_fallback_adapter=True),
        ]:
            report = fixture("browser")
            mutate(report)
            with self.assertRaises(ValueError):
                p.admit(report,"browser")
        report = fixture("torch")
        report["operation"] = "erf"
        with self.assertRaises(ValueError):
            p.admit(report,"torch")

    def test_summary_derived_from_all_intervals(self):
        result = p.analyze(*self.reports())
        for field,value in [("cases",result["cases"][:-1]),("descriptive_summary",{})]:
            altered = copy.deepcopy(result)
            altered[field] = value
            with self.assertRaises(ValueError):
                p.validate_summary(altered)
        result["cases"][0]["intervals"][0]["elapsed_ms"] *= 100
        result["cases"][0]["intervals"][0]["max_scaled_error"] = 2.
        with self.assertRaises(ValueError):
            p.validate_summary(result)


if __name__ == "__main__":
    unittest.main()

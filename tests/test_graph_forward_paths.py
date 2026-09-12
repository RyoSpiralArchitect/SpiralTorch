"""Admission/reaggregation only, not device-performance evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1] / "tools/bench_graph_forward_paths.py"
spec = importlib.util.spec_from_file_location("paths", path)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
spec = importlib.util.spec_from_file_location("comparison", path.with_name("validate_graph_forward_paths.py"))
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def fixture():
    seed, width, depth = 17, 7, 2
    parameters, stages = [], []
    index = seed
    for block in range(depth):
        for role in ("gain", "weight", "bias"):
            shape = [width, width] if role == "weight" else [width]
            values = []
            for j in range(width*width if role == "weight" else width):
                value = ((index*17 % 23)-11)/1024
                if role == "gain" or role == "weight" and j//width == j%width: value += 1
                values.append(value)
                index += 1
            parameters.append(dict(role=role, shape=shape, values=values))
        stages += [dict(kind="pointwise", parameters=[block*3], steps=[dict(op="multiply", rhs=1)]),
                   dict(kind="linear", weight=block*3+1, bias=block*3+2, gelu=True),
                   dict(kind="pointwise", parameters=[], steps=[dict(op="relu", rhs=None)])]
    samples = []
    for block in range(9):
        order = [(i+block+3+seed)%5 for i in range(5)]
        for route in order:
            samples.append(dict(block=block, route=route, order=order, elapsed_ms=1.,
                                forwards=8 if route>=3 else 1, max_abs_error=0.))
    case = dict(shape=[2,3,7], seed=seed, depth=depth, source_operations=8, gpu_stages=6,
                plan=dict(schema="spiraltorch.nn.inference_plan.v2", input_shape=[2,3,7], parameters=parameters, stages=stages),
                input=[((i+seed)%29)/16-.5 for i in range(42)], reference=[0.]*42,
                last_outputs=[[0.]*42 for _ in range(5)], samples=samples)
    return dict(schema="spiraltorch.graph_forward_bench.v1", status="passed", client="native",
                warmup=3, samples_per_route=9, burst=8,
                routes=["legacy_h2h","scalar_h2h","register_h2h","scalar_burst","register_burst"],
                adapter=dict(backend="Metal",device_type="IntegratedGpu"),cases=[case])


class Contract(unittest.TestCase):
    def admit(self, document):
        with patch.object(bench, "requests", return_value=[(17,[2,3,7],2)]):
            return bench.admit_native(document)

    def test_f32_roundtrip_is_bitwise_not_a_looser_tolerance(self):
        doc = fixture()
        doc["cases"][0]["plan"]["parameters"][0]["values"][0] = 1.0019531
        self.admit(doc)
        value = struct.unpack("<f", struct.pack("<I", 0x3f804001))[0]
        doc["cases"][0]["plan"]["parameters"][0]["values"][0] = value
        with self.assertRaises(ValueError): self.admit(doc)

    def test_matrix_input_and_topology_cannot_drift(self):
        doc = fixture()
        self.admit(doc)
        for mutation in (lambda d: d["cases"].clear(),
                         lambda d: d["cases"][0]["input"].__setitem__(0, 42.),
                         lambda d: d["cases"][0]["plan"]["stages"][1].__setitem__("gelu",False),
                         lambda d: d.__setitem__("warmup",0)):
            bad = copy.deepcopy(doc); mutation(bad)
            with self.assertRaises(ValueError): self.admit(bad)

    def test_order_count_and_nonfinite_timings_are_rejected(self):
        doc = fixture()
        for key,value in (("route",1), ("block",8), ("forwards",True), ("elapsed_ms",0.),
                          ("elapsed_ms",float("nan")), ("max_abs_error",-1.)):
            bad = copy.deepcopy(doc)
            bad["cases"][0]["samples"][0][key] = value
            with self.subTest(key=key,value=value), self.assertRaises(ValueError): self.admit(bad)
        bad = copy.deepcopy(doc); bad["cases"][0]["samples"].pop()
        with self.assertRaises(ValueError): self.admit(bad)

    def test_guarded_outputs_and_bool_data_are_not_valid_fixtures(self):
        for value in (True,float("inf"),float("nan")):
            with self.assertRaises(ValueError): bench.f32_bytes([value])
        bad=fixture(); bad["cases"][0]["last_outputs"][0][0]=float("nan")
        with self.assertRaises(ValueError): self.admit(bad)

    def test_h2h_and_burst_are_normalized_separately(self):
        report = bench.summarize(fixture()["cases"][0]["samples"])
        self.assertEqual(report[0]["median_ms_per_forward"],1.)
        self.assertEqual(report[3]["median_ms_per_forward"],.125)
        self.assertEqual(report[3]["samples"],9)

    def test_browser_lineage_and_captures_are_required(self):
        native = fixture()
        routes = ["scalar_h2h", "register_h2h", "scalar_burst", "register_burst"]
        case = {key:native["cases"][0][key] for key in ("shape", "seed", "depth")}
        case.update(samples=[], last_outputs={route:[0.]*42 for route in routes})
        for block in range(9):
            offset=(block+3+17)%4
            order=routes[offset:]+routes[:offset]
            for route in order:
                case["samples"].append(dict(block=block,route=route,order=order,elapsed_ms=1.,
                    forwards=8 if route.endswith("_burst") else 1,max_abs_error=0.))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"fixture.json"
            source.write_text(json.dumps(native))
            sha = bench.digest(source)
            report = dict(schema="spiraltorch.graph_forward_browser_bench.v1",status="passed",
                          source_fixture_sha256=sha,asset_sha256={"/fixture.json":sha},page_errors=[],
                          warmup=3,samples_per_route=9,burst=8,adapter=dict(backend="BrowserWebGpu",device_type="Other"),
                          routes=routes,cases=[case])
            comparison.admit_client(report,native,source,True)
            for mutate in (lambda d:d["asset_sha256"].clear(),
                           lambda d:d["cases"][0]["last_outputs"].pop("scalar_burst"),
                           lambda d:d["page_errors"].append("device lost"),
                           lambda d:d["cases"][0]["samples"][0].__setitem__("forwards",8)):
                bad=copy.deepcopy(report); mutate(bad)
                with self.assertRaises(ValueError): comparison.admit_client(bad,native,source,True)


if __name__ == "__main__": unittest.main()

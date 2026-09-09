"""Admission tests only: synthetic fixtures are not GPU execution evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("graph_client_validation",
    Path(__file__).resolve().parents[1] / "tools/verify_resident_graph_clients_torch.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ReportPath:
    def __init__(self, value): self.raw = json.dumps(value).encode()
    def read_bytes(self): return self.raw
    def resolve(self): return self
    def __str__(self): return "synthetic-admission-fixture"


def fixtures():
    python = dict(schema="spiraltorch.nn.graph_client_fixture.v1", status="passed", cases=[])
    browser = dict(schema="spiraltorch.nn.graph_browser_fixture.v1", status="passed", cases=[],
                   cpu_only=False, analytic_policy_cases=2, rollback_recovery_cases=2, parameterless_cases=1)
    returned = dict(schema="spiraltorch.nn.graph_handoff_return.v1", status="passed", cases=[])
    for seed in (17, 29):
        for shape in ([2], [3, 2], [2, 129, 2]):
            for policy in ("exact", "module_compatible"):
                for kernel, accumulation in (("scalar", "sequential"), ("register_2x2", "compensated")):
                    plan = dict(schema="spiraltorch.nn.inference_plan.v2", input_shape=shape,
                                stages=[], parameters=[dict(role="gain", shape=[2], values=[.5, .5])])
                    text = json.dumps(plan)
                    sha = validator.digest(text)
                    def states(count):
                        return [dict(submitted_step=i, batch_generation=1, gradient_policy=policy,
                                     input_shape=shape, output_shape=shape, stage_count=5,
                                     parameters=copy.deepcopy(plan["parameters"])) for i in range(1, count+1)]
                    adapter = dict(device_type="synthetic-not-evidence")
                    index = len(python["cases"])
                    python["cases"].append(dict(seed=seed, shape=shape, policy=policy, kernel=kernel,
                        accumulation=accumulation, learning_rate=.03125, plan_json=text, plan_sha256=sha,
                        half_plan_json=text, half_plan_sha256=sha, final_plan_json=text, final_plan_sha256=sha,
                        states=states(4), adapter=adapter))
                    browser["cases"].append(dict(index=index, plan_sha256=sha, final_plan_json=text,
                        final_plan_sha256=sha, states=states(4), resumed_states=states(2), adapter=adapter))
                    returned["cases"].append(dict(index=index, state=states(1)[0], adapter=adapter))
    return python, browser, returned


def seal(python, browser, returned):
    a = ReportPath(python)
    browser["asset_sha256"] = {"/fixture.json": hashlib.sha256(a.raw).hexdigest()}
    b = ReportPath(browser)
    returned.update(fixture_sha256=hashlib.sha256(a.raw).hexdigest(), browser_sha256=hashlib.sha256(b.raw).hexdigest())
    return a, b, ReportPath(returned)


class Admission(unittest.TestCase):
    def reject(self, mutate):
        data = fixtures()
        mutate(*data)
        with self.assertRaises(ValueError): validator.load_inputs(*seal(*data))

    def test_recipe_matrix_and_source_digests_are_retained(self):
        paths = seal(*fixtures())
        data, sources = validator.load_inputs(*paths)
        self.assertEqual(len(data[0]["cases"]), 24)
        self.assertEqual([s["sha256"] for s in sources], [hashlib.sha256(p.raw).hexdigest() for p in paths])

    def test_partial_duplicate_failed_or_cpu_fixture_rejected(self):
        for side in (0, 1, 2):
            self.reject(lambda *data: data[side]["cases"].pop())
            self.reject(lambda *data: data[side].update(status="error"))
        self.reject(lambda p, b, r: p["cases"].__setitem__(0, p["cases"][1]))
        self.reject(lambda p, b, r: b.update(cpu_only=True))
        self.reject(lambda p, b, r: b.update(rollback_recovery_cases=0))
        self.reject(lambda p, b, r: r["cases"][0].update(adapter=dict(device_type="Cpu")))

    def test_resume_counter_policy_parameter_metadata_and_case_order(self):
        for key, value in (("submitted_step", 3), ("gradient_policy", "exact "), ("batch_generation", 2),
                           ("stage_count", 1), ("output_shape", [999])):
            self.reject(lambda p, b, r: b["cases"][0]["resumed_states"][0].update({key: value}))
        self.reject(lambda p, b, r: b["cases"][0].update(index=1))
        self.reject(lambda p, b, r: p["cases"][0].update(learning_rate=.1))
        self.reject(lambda p, b, r: r["cases"][0]["state"]["parameters"][0].update(role="bias"))

    def test_digest_and_weight_bit_drift(self):
        self.reject(lambda p, b, r: p["cases"][0].update(half_plan_json="changed"))
        self.reject(lambda p, b, r: b["cases"][0]["states"][-1]["parameters"][0].update(values=[.25, .5]))
        paths = seal(*fixtures())
        altered = json.loads(paths[1].raw)
        altered["asset_sha256"]["/fixture.json"] = "wrong"
        with self.assertRaises(ValueError): validator.load_inputs(paths[0], ReportPath(altered), paths[2])

    def test_float32_transport_compares_bits_not_json_decimals(self):
        plan = dict(schema="spiraltorch.nn.inference_plan.v2", parameters=[dict(role="gain", shape=[1], values=[.1])])
        text = json.dumps(plan)
        snapshot = [dict(role="gain", shape=[1], values=[.10000000149011612])]
        validator.checked_plan(text, validator.digest(text), parameters=snapshot)
        for wrong in (.10000000894069672, 0.):
            snapshot[0]["values"] = [wrong]
            with self.assertRaises(ValueError): validator.checked_plan(text, validator.digest(text), parameters=snapshot)
        plan["parameters"][0]["values"] = [-0.]
        text = json.dumps(plan)
        snapshot[0]["values"] = [0.]
        with self.assertRaises(ValueError): validator.checked_plan(text, validator.digest(text), parameters=snapshot)


if __name__ == "__main__": unittest.main()

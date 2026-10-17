import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location("replay", Path(__file__).with_name("replay.py"))
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def fixture():
    cases = replay.expected_cases()
    for case in cases:
        if case["kind"] == "linear":
            r,k,c = case["shape"]
            sizes = {"forward":r*c, "input":r*k, "weight":k*c, "bias":c}
        else:
            sizes = dict.fromkeys(("forward","query","key","value"), math.prod(case["shape"]))
            if case["name"] != "seed11_ordinary":
                sizes["pair"] = 4096
        vectors = {key:[0.]*n for key,n in sizes.items()}
        case["reference"] = {"f32":vectors,"f64":vectors}
    return {"schema":"spiraltorch.language_local_vjp.fixture.v1", "pins":replay.PINS, "cases":cases}


class ReplayTests(unittest.TestCase):
    def test_expected_coverage(self):
        cases = replay.expected_cases()
        self.assertEqual(len(cases), 80)
        self.assertEqual(sum(c["kind"]=="linear" for c in cases), 68)
        self.assertEqual(sum(c["kind"]=="attention" for c in cases), 12)

    def test_validation_rejects_schema_and_case_drift(self):
        original = fixture()
        replay.validate_fixture(original)
        for mutate in [lambda f:f.update(schema="wrong"), lambda f:f["cases"].pop(),
                       lambda f:f["cases"].reverse(), lambda f:f["cases"][0].update(batch_index=1),
                       lambda f:f["cases"][0]["reference"]["f64"]["weight"].pop()]:
            bad = copy.deepcopy(original)
            mutate(bad)
            with self.assertRaises(ValueError):
                replay.validate_fixture(bad)

    def test_errors_reject_invalid_vectors(self):
        for a,b in [([],[]), ([0.],[0.,0.]), ([float("nan")],[0.]), ([True],[0.])]:
            with self.assertRaises(ValueError):
                replay.errors(a,b)
        self.assertEqual(replay.errors([3.,4.],[0.,0.])["l2"], 5.)

    def test_key_sum_reduces_batch_and_time_not_heads(self):
        self.assertEqual(replay.key_sum([1.,-1.,2.,-2.,3.,-3.,5.,-5.], [2,2,2,1]), 0.)

    def test_compare_and_fail_closed(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = json.dumps(data).encode()
            (root/"fixture.json").write_bytes(raw)
            native = {"schema":"spiraltorch.language_local_vjp.native.v1", "fixture_sha256":replay.sha(raw), "cases":[]}
            for case in data["cases"]:
                modes = ["native"] if case["kind"]=="attention" else ["sequential","tiled","compensated"]
                native["cases"].append({**{k:case[k] for k in replay.IDENTITY},
                                        "modes":dict.fromkeys(modes, case["reference"]["f32"])})
            args = SimpleNamespace(fixture=root/"fixture.json", native=root/"native.json", output=root/"ok.json")
            args.native.write_text(json.dumps(native))
            replay.compare(args)
            self.assertEqual(len(json.loads(args.output.read_text())["rows"]), 80)
            with self.assertRaises(FileExistsError):
                replay.compare(args)
            args.output = root/"bad.json"
            for field,value in [("schema","wrong"),("fixture_sha256","bad")]:
                bad = {**native,field:value}
                args.native.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    replay.compare(args)
            native["cases"][0]["modes"].pop("compensated")
            args.native.write_text(json.dumps(native))
            with self.assertRaises(ValueError):
                replay.compare(args)


if __name__ == "__main__":
    unittest.main()

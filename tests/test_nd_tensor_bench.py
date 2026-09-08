"""Dependency-light benchmark admission checks; not GPU execution evidence."""

import copy
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "nd_bench",
    Path(__file__).resolve().parents[1] / "tools/bench_nd_tensor_vs_torch.py",
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Admission(unittest.TestCase):
    def test_fixed_bounded_recipes(self):
        values = bench.recipes()
        self.assertEqual(len(values), 6)
        self.assertEqual({v["seed"] for v in values}, {17, 29, 43})
        self.assertEqual(
            {tuple(v["shape"]) for v in values}, {(8, 16, 64), (16, 32, 128)}
        )
        self.assertTrue(all(v["iterations"] == 20 for v in values))

    def test_fixture_binds_all_inputs_and_identity(self):
        config = bench.recipes()[0]
        shape, seed = config["shape"], config["seed"]
        value = dict(
            config,
            schema="spiraltorch.nd_bench.fixture.v1",
            identity={"frozen": True},
            gain=0.75,
            input=[
                ((i * 13 + seed) % 61) / 64.0 - 0.46875 for i in range(math.prod(shape))
            ],
            bias=[(i % 5) / 32.0 - 0.0625 for i in range(shape[2])],
        )
        bench.validate_fixture(value, config, {"frozen": True})
        for key, bad in (
            ("schema", "other"),
            ("shape", [1, 2, 3]),
            ("gain", 1.0),
            ("input", value["input"][:-1]),
            ("bias", [0.0] * shape[2]),
            ("identity", {}),
            ("seed", 29),
            ("iterations", 19),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_fixture(
                    dict(value, **{key: bad}), config, {"frozen": True}
                )

    def test_sample_rejects_bad_timings_guards_shapes_and_captures(self):
        valid = dict(
            shape=[2, 2, 4], elapsed_ms=1.0, values=[0.0] * 16, finite_checked=True
        )
        bench.validate_sample(valid, [2, 3, 4], True, True)
        for key, bad in (
            ("shape", [16]),
            ("elapsed_ms", True),
            ("elapsed_ms", 0),
            ("elapsed_ms", float("nan")),
            ("elapsed_ms", float("inf")),
            ("finite_checked", False),
            ("finite_checked", 1),
            ("values", [0.0] * 15),
            ("values", [float("nan")] * 16),
            ("values", [True] * 16),
            ("values", None),
        ):
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                bench.validate_sample(dict(valid, **{key: bad}), [2, 3, 4], True, True)
        with self.assertRaises(ValueError):
            bench.validate_sample(valid, [2, 3, 4], False, True)
        uncaptured = copy.copy(valid)
        uncaptured["values"] = None
        bench.validate_sample(uncaptured, [2, 3, 4], False, True)


if __name__ == "__main__":
    unittest.main()

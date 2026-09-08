"""Dependency-light admission/regression checks, not GPU execution evidence."""

import copy
import importlib.util
import math
from pathlib import Path
import unittest


def module(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "tools" / filename
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


bench = module("vjp_bench", "bench_pointwise_vjp_vs_torch.py")
validation = module("vjp_validation", "validate_pointwise_vjp_vs_torch.py")


def fixture(config):
    shape, seed = config["shape"], config["seed"]
    logical = [shape[1] - 1, shape[0], shape[2]]
    return dict(
        config,
        schema="spiraltorch.pointwise_vjp.bench_fixture.v1",
        logical_shape=logical,
        identity={},
        scale=0.75,
        input=[
            ((i * 13 + seed) % 61) / 64.0 - 0.46875 for i in range(math.prod(shape))
        ],
        gain=[0.5 + (i % 7) / 16.0 for i in range(shape[2])],
        cotangent=[
            ((i * 7 + seed) % 17) / 16.0 - 0.5 for i in range(math.prod(logical))
        ],
    )


class Admission(unittest.TestCase):
    def test_fixed_recipes_and_input_identity(self):
        self.assertEqual(len(bench.recipes()), 6)
        config = bench.recipes()[0]
        value = fixture(config)
        bench.validate_fixture(value, config, {})
        for key, bad in (
            ("identity", {"other": True}),
            ("input", []),
            ("gain", []),
            ("cotangent", []),
            ("scale", 1.0),
            ("iterations", 1),
            ("logical_shape", [1, 1, 1]),
            ("schema", "other"),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_fixture(dict(value, **{key: bad}), config, {})

    def test_missing_gradient_or_invalid_interval_rejected(self):
        shape = [2, 3, 4]
        value = dict(
            shape=shape,
            elapsed_ms=1.0,
            finite_checked=True,
            values=[[0.0] * 24, [0.0] * 24, [0.0] * 4, [0.0]],
        )
        bench.validate_sample(value, shape, True, True)
        for key, bad in (
            ("values", None),
            ("values", value["values"][:-1]),
            ("elapsed_ms", True),
            ("elapsed_ms", 0.0),
            ("elapsed_ms", float("nan")),
            ("finite_checked", False),
            ("shape", [24]),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.validate_sample(dict(value, **{key: bad}), shape, True, True)
        for bad in (float("inf"), float("nan"), True):
            corrupt = copy.deepcopy(value)
            corrupt["values"][2][0] = bad
            with self.assertRaises(ValueError):
                bench.validate_sample(corrupt, shape, True, True)
        with self.assertRaises(ValueError):
            bench.validate_sample(value, shape, False, True)

    def test_reaggregation_uses_all_intervals_and_gradients(self):
        report = dict(
            schema="spiraltorch.pointwise_vjp.torch_bench.v1",
            status="passed",
            warmups=2,
            samples=8,
            source_binding={"valid": True},
            fallback_enabled=False,
            identity={},
            cases=[],
        )
        for index, config in enumerate(bench.recipes()):
            value = fixture(config)
            shape = value["logical_shape"]
            captures = [
                [0.0] * math.prod(shape),
                [0.0] * math.prod(shape),
                [0.0] * shape[-1],
                [0.0],
            ]
            intervals = []
            for i in range(10):
                sample = dict(
                    shape=shape,
                    finite_checked=True,
                    values=captures if i == 2 else None,
                )
                intervals.append(
                    dict(
                        iteration=i,
                        warmup=i < 2,
                        order=(
                            ["rust", "torch"]
                            if (i + index) % 2 == 0
                            else ["torch", "rust"]
                        ),
                        rust=dict(sample, elapsed_ms=1.0),
                        torch=dict(sample, elapsed_ms=2.0),
                    )
                )
            report["cases"].append(
                dict(
                    fixture=value,
                    intervals=intervals,
                    max_abs_error=0.0,
                    median_ms={"rust": 1.0, "torch": 2.0},
                    torch_over_rust=2.0,
                )
            )
        self.assertEqual(len(bench.reaggregate(report)), 6)
        for mutate in (
            lambda r: r["cases"].pop(),
            lambda r: r["cases"][0].update(torch_over_rust=3.0),
            lambda r: r["cases"][0]["intervals"][0]["order"].reverse(),
            lambda r: r["cases"][0]["intervals"][5]["rust"].update(elapsed_ms=-1.0),
            lambda r: r["cases"][0]["intervals"][2]["rust"].update(
                values=[[0.0], [0.0], [0.0], [0.0]]
            ),
            lambda r: r.update(fallback_enabled=True),
        ):
            corrupt = copy.deepcopy(report)
            mutate(corrupt)
            with self.assertRaises(ValueError):
                bench.reaggregate(corrupt)

    def test_validation_requires_complete_vjp_guards(self):
        source = {
            "pointwise_vjp": {
                "schema": "spiraltorch.pointwise_vjp.fixture.v1",
                "status": "passed",
                "guards": dict(status="passed", **{k: True for k in validation.GUARDS}),
                "cases": [],
            }
        }
        with self.assertRaisesRegex(ValueError, "recipes"):
            validation.validate_fixture(source)
        for key in validation.GUARDS:
            corrupt = copy.deepcopy(source)
            corrupt["pointwise_vjp"]["guards"][key] = False
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "guards"):
                validation.validate_fixture(corrupt)


if __name__ == "__main__":
    unittest.main()

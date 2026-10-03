import importlib.util
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import spiraltorch as st


@pytest.fixture
def benchmark_module():
    source = Path(__file__).with_name("benchmark_fractional_learning.py")
    spec = importlib.util.spec_from_file_location("fractional_benchmark_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("alpha", [.5, 1., 1.5])
@pytest.mark.parametrize("time", [1, 5])
def test_same_math_routes_match_without_timing(benchmark_module, history, alpha, time):
    torch.set_num_threads(2)
    value = torch.linspace(-.8, .9, 6*time).reshape(2, 3, time).transpose(1, 2)
    upstream = value.cos()
    options = {"alpha": alpha, "kernel_len": 4, "step": .7, "history": history,
               "kernel": st.FractionalGlKernel(kernel_len=4, step=.7)}
    reference = benchmark_module.run_route("rust_joint", value, upstream, **options)
    for name in benchmark_module.ROUTES:
        actual = benchmark_module.run_route(name, value, upstream, **options)
        benchmark_module.check_result(actual, reference, exact=name != "torch_conv1d")


def test_reference_rejects_mismatches_and_nonfinite_results(benchmark_module):
    good = (torch.tensor([1.]), torch.tensor(2.))
    with pytest.raises(ValueError, match="differ"):
        benchmark_module.check_result((torch.tensor([1.1]), good[1]), good, exact=True)
    with pytest.raises(AssertionError):
        benchmark_module.check_result((torch.tensor([1.1]), good[1]), good, exact=False)
    with pytest.raises(ValueError, match="nonfinite"):
        benchmark_module.check_result((good[0], torch.tensor(float("nan"))), good, exact=False)


def test_published_timing_statistics_rebuild_without_timing(benchmark_module):
    directory = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-03-fractional-selective-vjp-timing"
    plan = json.loads((directory / "plan.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["successful_processes"] == 12 and summary["failed_processes"] == 0
    assert summary["status"] == "measured_under_ambient_load"
    assert set(summary["groups"]) == {f"{mode}-{'x'.join(map(str, shape))}"
                                      for mode in plan["maps"] for shape in plan["shapes"]}
    files = []
    expected_order = [list(order) for order in itertools.permutations(benchmark_module.ROUTES)] * 2
    for group in summary["groups"].values():
        assert len(group["files"]) == plan["repetitions"] == 3
        processes = []
        for name in group["files"]:
            assert Path(name).name == name
            files.append(name)
            row = json.loads((directory / name).read_text())
            assert row["status"] == "measured" and row["map"] == group["map"]
            assert row["shape"] == group["shape"] and row["round_order"] == expected_order
            for field in ("kernel_len", "alpha", "step", "seed", "threads", "warmup_per_route",
                          "native_sha256", "bridge_sha256", "benchmark_sha256"):
                assert row[field] == plan[field]
            assert row["native_profile_declared"] == plan["native_profile"] == "release"
            assert row["input_requires_grad"] is False and row["alpha_requires_grad"] is True
            assert row["device"] == "cpu" and row["dtype"] == "float32"
            for route in benchmark_module.ROUTES:
                values = row["measurements_ms"][route]
                assert len(values) == plan["rounds"] == 12
                assert all(math.isfinite(v) and v > 0 for v in values)
                assert row["median_ms"][route] == statistics.median(values)
                if route != "torch_conv1d":
                    assert row["correctness"][route]["max_abs_output_error"] == 0
                    assert row["correctness"][route]["abs_alpha_gradient_error"] == 0
            processes.append(row)
        assert len({r["input_sha256"] for r in processes}) == 1
        assert len({r["upstream_sha256"] for r in processes}) == 1
        for route in benchmark_module.ROUTES:
            values = [r["median_ms"][route] for r in processes]
            assert group["routes"][route] == {
                "median_of_process_medians_ms": statistics.median(values),
                "min_process_median_ms": min(values), "max_process_median_ms": max(values),
                "process_medians_ms": values,
            }
        joint, selective, reference = (group["routes"][route]["median_of_process_medians_ms"]
                                       for route in benchmark_module.ROUTES)
        assert group["observed_joint_over_selective"] == joint / selective
        assert group["observed_selective_over_torch"] == selective / reference
    assert len(files) == len(set(files)) == 12
    assert set(files) == {p.name for p in directory.glob("*-repeat*.json")}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert Path(name).name == name
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest

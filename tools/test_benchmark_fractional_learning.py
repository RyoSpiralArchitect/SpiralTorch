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
    report = benchmark_module.check_result(good, good, exact=True)
    assert report["output_sha256"] == hashlib.sha256(good[0].numpy().tobytes()).hexdigest()
    assert report["alpha_gradient_sha256"] == hashlib.sha256(good[1].numpy().tobytes()).hexdigest()
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


def paired_forward_records():
    directory = Path(__file__).resolve().parents[1] / "benchmarks/results/2026-10-03-fractional-paired-forward"
    plan = json.loads((directory / "timing-plan.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    return directory, plan, summary


def process_statistics(values):
    return {"median_of_process_medians_ms": statistics.median(values),
            "min_process_median_ms": min(values), "max_process_median_ms": max(values),
            "process_medians_ms": values}


def test_paired_forward_matrix_rebuilds_and_preserves_binary_parity(benchmark_module):
    directory, plan, summary = paired_forward_records()
    assert summary["status"] == "measured_under_ambient_load"
    assert summary["successful_processes"] == 24 and summary["failed_processes"] == 0
    assert summary["bit_exact_across_native_binaries"] is True
    assert set(summary["groups"]) == {f"{mode}-{'x'.join(map(str, shape))}"
                                      for mode in plan["maps"] for shape in plan["shapes"]}
    expected_order = [list(order) for order in itertools.permutations(benchmark_module.ROUTES)] * 2
    seen = []
    for group in summary["groups"].values():
        paired_rows = []
        for version in ("baseline", "paired"):
            detail = group["versions"][version]
            expected_files = [f"{version}-{group['map']}-{'x'.join(map(str, group['shape']))}-repeat{rep}.json"
                              for rep in range(1, plan["repetitions"] + 1)]
            assert detail["files"] == expected_files
            rows = [json.loads((directory / name).read_text()) for name in detail["files"]]
            seen.extend(detail["files"])
            paired_rows.extend(rows)
            for row in rows:
                assert row["status"] == "measured" and row["shape"] == group["shape"]
                assert row["map"] == group["map"] and row["round_order"] == expected_order
                assert row["native_sha256"] == plan[f"{version}_native_sha256"]
                for field in ("kernel_len", "alpha", "step", "seed", "threads", "warmup_per_route",
                              "bridge_sha256", "benchmark_sha256"):
                    assert row[field] == plan[field]
                assert row["native_profile_declared"] == plan["native_profile"] == "release"
                assert row["input_requires_grad"] is False and row["alpha_requires_grad"] is True
                assert row["device"] == "cpu" and row["dtype"] == "float32"
                for route in benchmark_module.ROUTES:
                    values = row["measurements_ms"][route]
                    assert len(values) == plan["rounds"] == 12
                    assert all(math.isfinite(v) and v > 0 for v in values)
                    assert row["median_ms"][route] == statistics.median(values)
                assert row["correctness"]["rust_joint"] == row["correctness"]["rust_selective"]
                assert row["correctness"]["rust_joint"] == rows[0]["correctness"]["rust_joint"]
                assert row["correctness"]["rust_joint"]["max_abs_output_error"] == 0
                assert row["correctness"]["rust_joint"]["abs_alpha_gradient_error"] == 0
            for route in benchmark_module.ROUTES:
                assert detail["routes"][route] == process_statistics([r["median_ms"][route] for r in rows])
        for field in ("input_sha256", "upstream_sha256", "torch", "machine"):
            assert len({r[field] for r in paired_rows}) == 1
        for field in ("output_sha256", "alpha_gradient_sha256", "alpha_gradient"):
            assert len({r["correctness"]["rust_joint"][field] for r in paired_rows}) == 1
        baseline = group["versions"]["baseline"]["routes"]["rust_selective"]["median_of_process_medians_ms"]
        paired = group["versions"]["paired"]["routes"]
        selective = paired["rust_selective"]["median_of_process_medians_ms"]
        reference = paired["torch_conv1d"]["median_of_process_medians_ms"]
        assert group["observed_baseline_over_paired_selective"] == baseline / selective
        assert group["observed_paired_selective_over_torch"] == selective / reference
    assert len(seen) == len(set(seen)) == 24
    assert set(seen) == {p.name for p in directory.glob("*-repeat*.json")}
    ledger = json.loads((directory / "processes.json").read_text())["python"]
    assert {r["name"] + ".json" for r in ledger} == set(seen)
    assert len(ledger) == 24 and all(r["exit_code"] == 0 for r in ledger)


def test_paired_forward_rust_diagnostic_statistics_rebuild():
    directory, _, summary = paired_forward_records()
    plan = json.loads((directory / "rust-plan.json").read_text())
    assert summary["rust"]["successful_processes"] == 6 and summary["rust"]["failed_processes"] == 0
    files = set()
    for index, group in enumerate(summary["rust"]["groups"]):
        for version in ("baseline", "paired"):
            detail = group["versions"][version]
            values = []
            assert len(detail["files"]) == plan["repetitions"] == 3
            for name in detail["files"]:
                assert Path(name).name == name and name.startswith(f"rust-{version}-")
                files.add(name)
                records = [json.loads(line) for line in (directory / name).read_text().splitlines()]
                assert len(records) == summary["rust"]["cases_per_process"] == 6
                row = records[index]
                for field in ("shape", "axis", "history", "output_fnv1a64", "alpha_gradient_f32_bits"):
                    assert row[field] == group[field]
                times = row["measurements_ms"]
                assert len(times) == plan["iterations"] == 12
                assert all(math.isfinite(v) and v > 0 for v in times)
                assert row["median_ms"] == statistics.median(times)
                values.append(row["median_ms"])
            assert {k: detail[k] for k in process_statistics(values)} == process_statistics(values)
        assert group["observed_baseline_over_paired"] == (
            group["versions"]["baseline"]["median_of_process_medians_ms"]
            / group["versions"]["paired"]["median_of_process_medians_ms"])
    assert files == {p.name for p in directory.glob("rust-*-repeat*.jsonl")}
    assert len(files) == 6


def test_paired_forward_published_receipts_and_hashes_remain_bound():
    directory, plan, summary = paired_forward_records()
    receipt = json.loads((directory / "pretrained-parity.json").read_text())
    assert receipt["status"] == "passed" and receipt["adapters"] == 6
    assert receipt["updates_per_binary"] == 12 and receipt["total_auxiliary_updates"] == 24
    for field in ("baseline_native_sha256", "paired_native_sha256"):
        assert receipt[field] == plan[field]
    for field in ("all_loss_gradient_parameter_adam_bits_equal", "frozen_base_unchanged",
                  "original_studies_unchanged"):
        assert receipt[field] is True
    assert receipt["heldout_losses_computed"] is False
    assert receipt["probe_sha256"] == hashlib.sha256((directory / "pretrained_probe.py").read_bytes()).hexdigest()
    assert receipt["comparator_sha256"] == hashlib.sha256((directory / "compare_pretrained.py").read_bytes()).hexdigest()
    wasm = json.loads((directory / "wasm-tiled.json").read_text())
    assert wasm["status"] == "passed" and wasm["cases"] == 128
    host = json.loads((directory / "host-samples.json").read_text())["samples"]
    assert len(host) == 30 and all(len(row["samples"]) == 2 for row in host)
    idle = [sample["idle_percent"] for row in host for sample in row["samples"]]
    assert summary["host_cpu_idle_percent_range"] == [min(idle), max(idle)]
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert Path(name).name == name
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest

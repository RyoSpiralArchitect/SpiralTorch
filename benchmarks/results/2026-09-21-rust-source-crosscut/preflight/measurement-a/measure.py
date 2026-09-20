"""Run matched frozen workers and an explicitly single-threaded PyTorch CPU comparator."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

if not __debug__:
    raise RuntimeError("Run this evidence checker without Python optimization flags")


def fixture(rows, cols, seed):
    values = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            seed = (seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
            row.append(((seed >> 48) - 32768) / 32768.0)
        values.append(row)
    return values


def torch_run():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = []
    with torch.inference_mode():
        for batch, dims in [(8, 31), (32, 64), (96, 128)]:
            a = torch.tensor(fixture(batch, dims, 17), dtype=torch.float32)
            p = torch.tensor(fixture(batch, dims, 29), dtype=torch.float32)
            for normalize in [False, True]:
                def run():
                    logits = a @ p.T
                    if normalize:
                        an = torch.linalg.vector_norm(a.double(), dim=1).float().clamp_min(torch.finfo(torch.float32).eps)
                        pn = torch.linalg.vector_norm(p.double(), dim=1).float().clamp_min(torch.finfo(torch.float32).eps)
                        logits = logits / (an[:, None] * pn[None, :])
                    logits = logits / 0.3
                    loss = (torch.logsumexp(logits, dim=1) - logits.diag()).mean()
                    return float(loss), logits, torch.arange(batch)
                for _ in range(3):
                    run()
                intervals = []
                for _ in range(9):
                    start = time.perf_counter_ns()
                    for _ in range(32):
                        run()
                    intervals.append((time.perf_counter_ns() - start) / 32)
                loss, logits, labels = run()
                cases.append(dict(batch=batch, dims=dims, normalize=normalize, loss=loss,
                                  logits=logits.flatten().tolist(), labels=labels.tolist(),
                                  elapsed_ns=intervals, repetitions=32))
    print(json.dumps(dict(torch_version=torch.__version__, device="cpu", threads=1,
                         interop_threads=1, inference_mode=True, cases=cases)))


def key(case):
    names = ("kind", "api", "layout", "batch", "dims", "normalize", "len", "octaves", "iterations")
    return json.dumps({k: case[k] for k in names if k in case}, sort_keys=True)


def expected_keys():
    cases = []
    for batch, dims in [(8, 31), (32, 64), (96, 128)]:
        for normalize in [False, True]:
            for api, layout in [("vector", "rows"), ("tensor", "row_major"), ("tensor", "col_major"),
                                ("tensor_as_result", "row_major"), ("tensor_as_result", "col_major")]:
                cases.append(dict(kind="info_nce", batch=batch, dims=dims, normalize=normalize, api=api, layout=layout))
    for length in [16, 4096, 65536]:
        for octaves, iterations in [(1, 1), (4, 16)]:
            cases.append(dict(kind="fractal_weave", len=length, octaves=octaves, iterations=iterations))
    return {key(case) for case in cases}


def reference(batch, dims, normalize):
    a, p = fixture(batch, dims, 17), fixture(batch, dims, 29)
    norm = lambda row: max(math.sqrt(sum(v * v for v in row)), 2 ** -23)
    temperature = 0.30000001192092896
    logits = [sum(x * y for x, y in zip(ar, pr)) / (norm(ar) * norm(pr) if normalize else 1) / temperature
              for ar in a for pr in p]
    loss = 0.0
    for index in range(batch):
        row = logits[index * batch:(index + 1) * batch]
        maximum = max(row)
        loss += maximum - row[index] + math.log(sum(math.exp(v - maximum) for v in row))
    return logits, loss / batch


def report(directory):
    documents = [json.loads((directory / (name + ".json")).read_text())
                 for name in ["baseline-a", "candidate-a", "baseline-b", "candidate-b"]]
    maps = []
    for doc in documents:
        assert doc["schema"] == "spiraltorch.source_crosscut.v1"
        assert len(doc["cases"]) == 36
        mapping = {key(case): case for case in doc["cases"]}
        assert len(mapping) == 36 and set(mapping) == expected_keys()
        assert doc["tolerance"] == {"atol": 1e-4, "rtol": 1e-4}
        assert doc["fixture"]["anchor_seed"] == 17 and doc["fixture"]["positive_seed"] == 29
        assert abs(doc["fixture"]["temperature"] - 0.3) < 1e-7
        for case in mapping.values():
            intervals = case["measurement"]["elapsed_ns"]
            assert len(intervals) == 9 and all(math.isfinite(v) and v > 0 for v in intervals)
            assert case["measurement"]["repetitions"] == (32 if case["kind"] == "info_nce" else 4)
        maps.append(mapping)
    assert all(set(mapping) == set(maps[0]) for mapping in maps)
    comparisons = []
    for identity in maps[0]:
        b1, c1, b2, c2 = [mapping[identity] for mapping in maps]
        assert c1["validation"]["passed"] and c2["validation"]["passed"]
        valid_baseline = b1["validation"]["passed"] and b2["validation"]["passed"]
        ratios = [statistics.median(b["measurement"]["elapsed_ns"]) / statistics.median(c["measurement"]["elapsed_ns"])
                  for b, c in [(b1, c1), (b2, c2)]]
        comparisons.append(dict(condition=json.loads(identity), baseline_valid=valid_baseline,
                                baseline_over_candidate=ratios if valid_baseline else None,
                                exclusion=None if valid_baseline else "baseline numerical failure; raw timings retained, not a speedup",
                                baseline_allocations=[b["measurement"] for b in (b1, b2)],
                                candidate_allocations=[c["measurement"] for c in (c1, c2)]))
    torch_doc = json.loads((directory / "torch.json").read_text())
    assert len(torch_doc["cases"]) == 6
    assert {(c["batch"], c["dims"], c["normalize"]) for c in torch_doc["cases"]} == {
        (b, d, n) for b, d in [(8, 31), (32, 64), (96, 128)] for n in [False, True]}
    assert torch_doc["device"] == "cpu" and torch_doc["threads"] == torch_doc["interop_threads"] == 1
    torch_comparisons = []
    for case in torch_doc["cases"]:
        assert len(case["elapsed_ns"]) == 9 and all(math.isfinite(v) and v > 0 for v in case["elapsed_ns"])
        expected, loss = reference(case["batch"], case["dims"], case["normalize"])
        assert len(expected) == len(case["logits"])
        assert all(math.isfinite(a) and abs(a - b) <= 1e-4 + 1e-4 * abs(b) for a, b in zip(case["logits"], expected))
        assert math.isfinite(case["loss"]) and abs(case["loss"] - loss) <= 1e-4 + 1e-4 * abs(loss)
        assert case["labels"] == list(range(case["batch"]))
        for candidate in maps[1].values():
            if candidate["kind"] != "info_nce" or candidate["layout"] == "col_major":
                continue
            if any(candidate[k] != case[k] for k in ["batch", "dims", "normalize"]):
                continue
            identity = key(candidate)
            medians = [statistics.median(m[identity]["measurement"]["elapsed_ns"]) for m in (maps[1], maps[3])]
            torch_comparisons.append(dict(condition=json.loads(identity),
                torch_over_candidate=[statistics.median(case["elapsed_ns"]) / median for median in medians],
                torch_max_logit_abs=max(abs(a - b) for a, b in zip(case["logits"], expected))))
    return dict(comparisons=comparisons, torch_comparisons=torch_comparisons,
                candidate_numerical_passes=72, baseline_numerical_failures=sum(not c["validation"]["passed"] for m in (maps[0], maps[2]) for c in m.values()),
                numerical_reference="independent float64 dot, normalization, logsumexp; atol=rtol=1e-4; fractal bitwise base+branch",
                limitations=["CPU forward only, no gradient or resident GPU comparison",
                             "PyTorch has one CPU/inter-op thread; default-thread results may differ",
                             "Host exclusivity and thermal state unknown; two fixed AB/BA rounds, no outlier trimming",
                             "Allocator counts are Rust allocation requests, not peak memory or Python/GPU allocations"])


def measure(directory, torch_python):
    receipt = dict(platform=platform.platform(), machine=platform.machine(), steps=[], hashes={})
    if platform.system() == "Darwin":
        receipt["hardware"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize", "hw.logicalcpu"], text=True).splitlines()
    for name in ["baseline-worker", "candidate-worker"]:
        receipt["hashes"][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    for name in ["baseline-a", "candidate-a", "candidate-b", "baseline-b"]:
        command = [str(directory / (name.split("-")[0] + "-worker")), name]
        with (directory / (name + ".json")).open("x") as output:
            subprocess.run(command, check=True, stdout=output)
        receipt["steps"].append(command)
        print(name, "complete", flush=True)
    command = [torch_python, str(Path(__file__).resolve()), "torch"]
    with (directory / "torch.json").open("x") as output:
        subprocess.run(command, check=True, stdout=output)
    receipt["steps"].append(command)
    for name in ["baseline-a", "candidate-a", "candidate-b", "baseline-b", "torch"]:
        path = directory / (name + ".json")
        receipt["hashes"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "measurement-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (directory / "comparison.json").write_text(json.dumps(report(directory), indent=2) + "\n")
    print("All candidate and PyTorch numerical gates passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["torch", "measure", "report"])
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--torch-python", default=sys.executable)
    args = parser.parse_args()
    if args.mode == "torch":
        torch_run()
    elif args.mode == "report":
        print(json.dumps(report(args.directory), indent=2))
    else:
        measure(args.directory, args.torch_python)

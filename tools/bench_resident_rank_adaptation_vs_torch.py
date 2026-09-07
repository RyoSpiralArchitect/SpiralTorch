#!/usr/bin/env python3
"""Source-bound resident candidate controls, real Rust policy feedback, and CUDA reference."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

import bench_rank_vs_torch as audit
import bench_resident_rank_vs_torch as resident
import torch_rank_reference as cuda_reference


def requests(bench):
    for seed in [17, 29, 43]:
        for kind in ["topk", "midk", "bottomk"]:
            for cols, k in [(257, 7), (8193, 65)]:
                values, _ = bench.fixture(2 * cols, seed)
                if seed == 43:
                    values = [float(int(value * 8)) for value in values]
                tiles = [32, 128, 256, 512]
                rotation = seed % len(tiles)
                tiles = tiles[rotation:] + tiles[:rotation]
                scripts = [f"u2: true; rank_tile: {tile}; ctile: {tile};" for tile in tiles]
                for policy in ["ucb", "thompson_sampling"]:
                    yield dict(kind=kind, rows=2, cols=cols, k=k, scripts=scripts,
                               input=values, seed=seed, policy=policy, rounds=64), tiles


def validate_native(result, request, tiles):
    if result.get("status") != "passed" or result.get("repetitions") != 16:
        raise ValueError("native resident adaptation did not pass")
    if any(result.get(key) != request[key] for key in ["kind", "rows", "cols", "k", "seed", "policy"]):
        raise ValueError("native request identity mismatch")
    candidates = result["initial"]["candidates"]
    controls = result["control_samples_per_op_ms"]
    if len(candidates) != len(tiles) or len(controls) != len(tiles):
        raise ValueError("candidate cardinality mismatch")
    for index, (candidate, tile, samples) in enumerate(zip(candidates, tiles, controls)):
        signature = (f"spiraltorch.rank_execution.v1/backend=wgpu/kind={request['kind']}"
                     f"/rows={request['rows']}/cols={request['cols']}/k={request['k']}"
                     f"/fallback=forbid/scope=declared_native/path=exact_2ce/tile={min(tile, request['cols'])}")
        if type(candidate["index"]) is not int or candidate["index"] != index or candidate["execution_signature"] != signature:
            raise ValueError("candidate execution footprint mismatch")
        if candidate["spiralk_source_sha256"] != hashlib.sha256(request["scripts"][index].encode()).hexdigest():
            raise ValueError("candidate source mismatch")
        if len(samples) != 12 or any(not math.isfinite(v) or v < 0 for v in samples):
            raise ValueError("invalid control samples")
    initial_counts = result["initial"]["observation_counts"]["rank_plan_variant"]
    if any(initial_counts.values()):
        raise ValueError("controls must not seed the policy")
    observations = result["observations"]
    if len(observations) != request["rounds"]:
        raise ValueError("adaptive observation cardinality mismatch")
    counts = Counter()
    for step, item in enumerate(observations, 1):
        selection, observation = item["selection"], item["observation"]
        index = selection["candidate_index"]
        if type(index) is not int or not 0 <= index < len(tiles):
            raise ValueError("invalid selected candidate")
        if selection["execution_signature"] != candidates[index]["execution_signature"]:
            raise ValueError("selected execution footprint changed")
        if (selection["spiralk_source_sha256"] != candidates[index]["spiralk_source_sha256"] or
            selection["plan"] != candidates[index]["plan"]):
            raise ValueError("selected candidate identity changed")
        elapsed = observation["elapsed_ms"]
        if not math.isfinite(elapsed) or elapsed < 0 or item["per_op_ms"] != elapsed / 16:
            raise ValueError("invalid timing boundary")
        if (selection["selection_id"] != step or observation["selection_id"] != step or
            observation["candidate_index"] != index or not observation["credited"] or
            not observation["correctness_passed"] or observation["candidate_quarantined"] or
            observation["reward"] != 1 / (1 + elapsed)):
            raise ValueError("invalid Rust feedback receipt")
        counts[str(index)] += 1
        if Counter(observation["observation_counts"]["rank_plan_variant"]) != counts:
            raise ValueError("intermediate policy counts do not match observations")
    final = result["rank_adaptation"]
    if (final["candidates"] != candidates or final["pending_selection_id"] is not None or
        Counter(final["observation_counts"]["rank_plan_variant"]) != counts):
        raise ValueError("final policy counts do not match real observations")


def run(executable):
    report = {"schema": "spiraltorch.resident_rank_adaptation_comparison.v2", "status": "error", "cases": []}
    try:
        resident.require_uncontended_gpu()
        bench = audit.load_bench_module()
        before = audit.source_identity()
        original = audit.file_identity(executable)
        all_requests = list(requests(bench))
        payload = "".join(json.dumps(r, allow_nan=False) + "\n" for r, _ in all_requests)
        report.update(source=before, request_sha256=hashlib.sha256(payload.encode()).hexdigest())
        with tempfile.TemporaryDirectory(prefix="resident-adaptation-", dir=executable.parent) as directory:
            image = Path(directory) / "resident_rank_adaptation_bench"
            os.link(executable, image)
            image_before = audit.file_identity(image)
            identity = audit.read_native_build_identity(image)
            binding = audit.validate_source_binding(identity, before)
            report.update(native_build_identity=identity, build_source_binding=binding)
            if not binding["valid"]:
                raise ValueError("source/build identity mismatch")
            native = subprocess.run([str(image)], input=payload, text=True, capture_output=True, timeout=900)
            report.update(native_returncode=native.returncode, native_stderr=native.stderr)
            if native.returncode or native.stderr:
                report["native_stdout"] = native.stdout
                raise RuntimeError("native benchmark failed; raw output retained")
            results = [json.loads(line) for line in native.stdout.splitlines()]
            report["native_results"] = results
            if len(results) != len(all_requests):
                report["native_stdout"] = native.stdout
                raise ValueError("native result cardinality mismatch")
            # Start CUDA work only after the native GPU process is terminal.
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA required; no CPU fallback")
            report.update(torch=torch.__version__, torch_device=torch.cuda.get_device_name(),
                          comparison="separate native/PyTorch process blocks, not interleaved or GPU-event timing",
                          boundary="16 resident operations plus completion fence per batch; validation readbacks between batches, outside timing; policy uses whole-batch elapsed_ms")
            with torch.inference_mode():
                for (r, tiles), result in zip(all_requests, results):
                    validate_native(result, r, tiles)
                    if result["adapter"]["name"] != torch.cuda.get_device_name():
                        raise RuntimeError("WGPU and CUDA must use the same named GPU")
                    host = torch.tensor(r["input"], dtype=torch.float32).reshape(r["rows"], r["cols"])
                    expected = cuda_reference.contract(r)
                    actual = torch.tensor(result["values"], dtype=torch.float32)
                    expected_bits = torch.tensor(expected["values"], dtype=torch.float32).view(torch.int32)
                    if not torch.equal(actual.view(torch.int32), expected_bits) or result["indices"] != expected["indices"]:
                        raise ValueError("native output differs from canonical PyTorch reference")
                    device = host.cuda()
                    controls = cuda_reference.measure(r, device, torch, bench.summarize)
                    operation = controls["best_fixed"]
                    report["cases"].append({"request": {key: value for key, value in r.items() if key != "input"},
                        "tiles": tiles, "default_candidate_index": tiles.index(256), "native": result,
                        "controls": [bench.summarize(samples) for samples in result["control_samples_per_op_ms"]],
                        "adaptive": bench.summarize([item["per_op_ms"] for item in result["observations"]]),
                        "torch_timing": controls["timings"][operation], "torch_operation": operation,
                        "torch_controls": controls, "legacy_torch_operation": resident.cuda_rank_operation(r)})
            after = audit.source_identity()
            resident.require_uncontended_gpu()
            report["provenance"] = {"valid": before == after and original == audit.file_identity(executable) and image_before == audit.file_identity(image),
                                    "source_after": after, "executable": original, "execution_image": image_before}
            if not report["provenance"]["valid"]:
                raise RuntimeError("source or executable changed during measurement")
            report["status"] = "passed"
            del report["native_results"]
    except subprocess.TimeoutExpired as error:
        report.update(error=str(error), native_stdout=str(error.stdout), native_stderr=str(error.stderr))
    except Exception as error:
        report["error"] = str(error)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.executable.resolve(strict=True))
    audit.write_report_exclusive(args.output, report)
    print(json.dumps({"status": report["status"], "cases": len(report["cases"]), "error": report.get("error")}))
    raise SystemExit(report["status"] != "passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strict rank parity plus diagnostic Rust/Python host-boundary timings."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile


SCHEMA = "spiraltorch.rank_backend_bench.v2"
BUILD_IDENTITY_SCHEMA = "spiraltorch.native_build_identity.v1"
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mode": stat.st_mode,
    }


def git_bytes(*args):
    completed = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def source_identity():
    status = git_bytes("status", "--porcelain=v1", "--untracked-files=no")
    diff = git_bytes("diff", "--binary", "HEAD", "--")
    return {
        "repository": str(REPO_ROOT),
        "commit": git_bytes("rev-parse", "HEAD").decode().strip(),
        "tree": git_bytes("rev-parse", "HEAD^{tree}").decode().strip(),
        "tracked_dirty": bool(status),
        "tracked_status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def read_native_build_identity(executable):
    completed = subprocess.run(
        [str(executable), "--build-info"],
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "native build identity failed "
            f"with exit {completed.returncode}: {completed.stderr[-4000:]}"
        )
    if completed.stderr:
        raise RuntimeError(
            f"native build identity wrote to stderr: {completed.stderr[-4000:]}"
        )
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise RuntimeError(
            f"native build identity returned {len(lines)} lines instead of one"
        )
    try:
        identity = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"native build identity is not JSON: {error}") from error
    if not isinstance(identity, dict):
        raise RuntimeError("native build identity is not an object")
    return identity


def validate_source_binding(identity, source):
    manifest = identity.get("manifest")
    manifest = manifest if isinstance(manifest, dict) else {}
    package = manifest.get("pkg")
    package = package if isinstance(package, dict) else {}
    git = manifest.get("git")
    git = git if isinstance(git, dict) else {}
    checks = {
        "schema_matches": identity.get("schema") == BUILD_IDENTITY_SCHEMA,
        "package_matches": package.get("name") == "st-core",
        "commit_matches": git.get("commit") == source.get("commit"),
        "tree_matches": git.get("tree") == source.get("tree"),
        "build_was_clean": git.get("dirty") is False,
        "source_is_clean": source.get("tracked_dirty") is False,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "embedded_commit": git.get("commit"),
        "embedded_tree": git.get("tree"),
        "source_commit": source.get("commit"),
        "source_tree": source.get("tree"),
    }


def load_bench_module():
    path = Path(__file__).with_name("bench_backend_vs_torch.py")
    spec = importlib.util.spec_from_file_location("backend_bench", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def requests(bench):
    for backend in ("wgpu", "cuda"):
        for seed_index, seed in enumerate((17, 29, 43)):
            for kind_index, kind in enumerate(("topk", "bottomk", "midk")):
                shapes = ((256, 8, "random"), (2048, 16, "concentrated"))
                for shape_index, (cols, k, pattern) in enumerate(shapes):
                    values, _ = bench.fixture(2 * cols, seed)
                    if pattern == "concentrated":
                        # All winners map to one heap thread; uniform inputs hide this.
                        for row in range(2):
                            for i in range(16):
                                values[row * cols + i * 128] = (100.0 + i) * (
                                    -1 if kind == "bottomk" else 1
                                )
                    if backend == "cuda":
                        # CUDA rank kernels consume the validated workgroup directly.
                        workgroups = (32, 128, 256)
                        rotation = seed_index % len(workgroups)
                        workgroups = workgroups[rotation:] + workgroups[:rotation]
                        scripts = [f"wg: {workgroup};" for workgroup in workgroups]
                    else:
                        scripts = ["u2: true;"]
                        if cols == 256:
                            scripts.append("u2: false;")
                            rotation = (
                                seed_index + kind_index + shape_index
                            ) % len(scripts)
                            scripts = scripts[rotation:] + scripts[:rotation]
                    yield {
                        "backend": backend,
                        "kind": kind,
                        "rows": 2,
                        "cols": cols,
                        "k": k,
                        "input": values,
                        "iterations": 12,
                        "warmup": 2,
                        "seed": seed,
                        "scripts": scripts,
                    }, pattern


def failure_report(stage, error, state):
    report = {
        "schema": SCHEMA,
        "status": "error",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "comparison": "not completed",
    }
    for key in (
        "source_before",
        "source_after",
        "source_executable_before",
        "source_executable_after",
        "execution_image_before",
        "execution_image_after",
        "native_build_identity",
        "source_binding",
        "request_sha256",
    ):
        if state.get(key) is not None:
            report[key] = state[key]
    native = state.get("native")
    if native is not None:
        report["native_returncode"] = native.returncode
        report["native_stderr"] = native.stderr[-4000:]
        report["native_stdout_sha256"] = hashlib.sha256(
            native.stdout.encode()
        ).hexdigest()
    return report


def run_benchmark(args):
    stage = "source_preflight"
    state = {}
    try:
        state["source_before"] = source_identity()
        stage = "executable_preflight"
        state["source_executable_before"] = file_identity(args.executable)
        source_executable = args.executable.resolve(strict=True)

        with tempfile.TemporaryDirectory(
            prefix=".spiraltorch-rank-bench-", dir=source_executable.parent
        ) as directory:
            execution_path = Path(directory) / source_executable.name
            os.link(source_executable, execution_path)
            state["execution_image_before"] = file_identity(execution_path)
            if (
                state["execution_image_before"]["sha256"]
                != state["source_executable_before"]["sha256"]
            ):
                raise RuntimeError("private execution image differs from source executable")

            stage = "build_identity_preflight"
            state["native_build_identity"] = read_native_build_identity(execution_path)
            state["source_binding"] = validate_source_binding(
                state["native_build_identity"], state["source_before"]
            )
            if not state["source_binding"]["valid"]:
                raise RuntimeError(
                    "native executable build identity does not match the clean source tree"
                )

            stage = "benchmark_helper_import"
            bench = load_bench_module()
            stage = "torch_import"
            import torch

            torch.set_num_threads(1)
            stage = "cuda_preflight"
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required; no fallback")

            stage = "request_generation"
            cases = list(requests(bench))
            payload = "".join(json.dumps(request) + "\n" for request, _ in cases)
            state["request_sha256"] = hashlib.sha256(payload.encode()).hexdigest()

            stage = "native_execution"
            native = subprocess.run(
                [str(execution_path)],
                input=payload,
                text=True,
                capture_output=True,
            )
            state["native"] = native
            state["execution_image_after"] = file_identity(execution_path)
            state["source_executable_after"] = file_identity(args.executable)

            stage = "native_output_parse"
            results = []
            for line_number, line in enumerate(native.stdout.splitlines(), start=1):
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(
                        f"native output line {line_number} is not JSON: {error}"
                    ) from error
            if len(results) != len(cases):
                raise RuntimeError(
                    f"native process returned {len(results)}/{len(cases)} cases: "
                    f"{native.stderr[-4000:]}"
                )

            report = {
                "schema": SCHEMA,
                "status": "passed",
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0),
                "request_sha256": state["request_sha256"],
                "executable_sha256": state["execution_image_before"]["sha256"],
                "native_returncode": native.returncode,
                "native_stderr": native.stderr,
                "comparison": (
                    "diagnostic unpaired timings; Rust and Python wrappers differ; "
                    "no speed ratio"
                ),
                "native_control": (
                    "equal-count rotating round-robin, separate from adaptive rewards"
                ),
                "candidate_order": "seed-rotated before native execution",
                "execution_image": {
                    "snapshot_method": "private hardlink",
                    "source_before": state["source_executable_before"],
                    "source_after": state["source_executable_after"],
                    "executed_before": state["execution_image_before"],
                    "executed_after": state["execution_image_after"],
                },
                "native_build_identity": state["native_build_identity"],
                "source": state["source_before"],
                "cases": [],
            }

            stage = "torch_comparison"
            for (request, pattern), result in zip(cases, results):
                host = torch.tensor(request["input"], dtype=torch.float32).reshape(
                    2, request["cols"]
                )
                k, kind = request["k"], request["kind"]

                def torch_op():
                    tensor = host.cuda()
                    if kind == "midk":
                        values, indices = tensor.sort(dim=1, stable=True)
                        start = (request["cols"] - k) // 2
                        return (
                            values[:, start : start + k].cpu(),
                            indices[:, start : start + k].cpu(),
                        )
                    values, indices = tensor.topk(
                        k, dim=1, largest=kind == "topk", sorted=True
                    )
                    return values.cpu(), indices.cpu()

                expected, expected_ids = host.sort(
                    dim=1, descending=kind == "topk", stable=True
                )
                start = (request["cols"] - k) // 2 if kind == "midk" else 0
                expected = expected[:, start : start + k]
                expected_ids = expected_ids[:, start : start + k]
                actual, _ = torch_op()
                bench.correctness(actual, expected.double(), torch, 0, 0)
                if result["status"] == "passed" and (
                    result["values"] != expected.flatten().tolist()
                    or result["indices"] != expected_ids.flatten().tolist()
                ):
                    result = {
                        "status": "error",
                        "stage": "native_torch_parity",
                        "error": "native/PyTorch canonical rank mismatch",
                        "native": result,
                    }
                timing = bench.paired_timings(
                    {"torch_cuda_host_to_host": torch_op},
                    2,
                    12,
                    torch.cuda.synchronize,
                    request["seed"],
                )
                report["cases"].append(
                    {
                        "request": {
                            key: value
                            for key, value in request.items()
                            if key != "input"
                        },
                        "pattern": pattern,
                        "native": result,
                        "torch_timings": timing,
                    }
                )
                print(
                    json.dumps(
                        {
                            "backend": request["backend"],
                            "kind": kind,
                            "cols": request["cols"],
                            "seed": request["seed"],
                            "status": result["status"],
                        }
                    ),
                    flush=True,
                )

            stage = "provenance_postflight"
            state["source_after"] = source_identity()
            source_stable = state["source_before"] == state["source_after"]
            source_executable_stable = (
                state["source_executable_before"]
                == state["source_executable_after"]
            )
            execution_image_stable = (
                state["execution_image_before"] == state["execution_image_after"]
            )
            report["source_after"] = state["source_after"]
            report["provenance"] = {
                "source_stable": source_stable,
                "source_executable_stable": source_executable_stable,
                "execution_image_stable": execution_image_stable,
                "build_source_binding": state["source_binding"],
                "valid": state["source_binding"]["valid"]
                and source_stable
                and source_executable_stable
                and execution_image_stable,
            }
            case_failure = any(
                case["native"]["status"] != "passed" for case in report["cases"]
            )
            if not report["provenance"]["valid"]:
                report["status"] = "error"
                report["stage"] = "provenance_postflight"
                report["error"] = "source or executable identity changed during benchmark"
            elif native.returncode != 0:
                report["status"] = "error"
                report["stage"] = "native_execution"
                report["error"] = f"native process exited with {native.returncode}"
            elif case_failure:
                report["status"] = "error"
                report["stage"] = "case_validation"
                report["error"] = "one or more benchmark cases failed"
            return report, int(report["status"] != "passed")
    except Exception as error:
        return failure_report(stage, error, state), 1


def write_report_exclusive(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "x",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["RAYON_NUM_THREADS"] = "1"
    report, exit_code = run_benchmark(args)
    write_report_exclusive(args.output, report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

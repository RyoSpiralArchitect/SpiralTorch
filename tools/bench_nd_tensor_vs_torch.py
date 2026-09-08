#!/usr/bin/env python3
"""Bounded, alternating-order resident N-D chains vs eager PyTorch; no fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_rank_vs_torch as audit
from bench_resident_nn_vs_torch import admit_device, match_adapter


class Worker:
    """One owned process, bounded responses/exit, and no automatic restart."""

    def __init__(
        self,
        command,
        stderr,
        response_timeout=120.0,
        exit_timeout=10.0,
        limit=64 * 1024 * 1024,
    ):
        self.response_timeout = response_timeout
        self.exit_timeout = exit_timeout
        self.limit = limit
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            bufsize=0,
        )
        self.messages = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        # Raw pipes avoid a buffered-I/O lock if an inherited pipe outlives its owner.
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _offer(self, value):
        while not self.stop.is_set():
            try:
                self.messages.put(value, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def _read(self):
        pending = bytearray()
        try:
            while not self.stop.is_set():
                chunk = self.process.stdout.read(65536)
                if not chunk:
                    raise RuntimeError(
                        "Rust worker closed stdout before the next complete response"
                    )
                pending.extend(chunk)
                while (end := pending.find(b"\n")) >= 0:
                    if end + 1 > self.limit:
                        raise ValueError(
                            "worker response exceeds bounded fixture budget"
                        )
                    line = bytes(pending[: end + 1])
                    del pending[: end + 1]
                    if not self._offer(line):
                        return
                if len(pending) > self.limit:
                    raise ValueError("worker response exceeds bounded fixture budget")
        except Exception as error:
            self._offer(error)

    def request(self, payload):
        self.process.stdin.write((json.dumps(payload) + "\n").encode())
        try:
            response = self.messages.get(timeout=self.response_timeout)
        except queue.Empty as error:
            raise TimeoutError(
                "resident worker response deadline exceeded; no restart"
            ) from error
        if isinstance(response, Exception):
            raise response
        return json.loads(response)

    def finish(self):
        self.process.stdin.close()
        self.process.wait(timeout=self.exit_timeout)
        if self.process.returncode != 0:
            raise RuntimeError(f"worker exit {self.process.returncode}")

    def abort(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=self.exit_timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=self.exit_timeout)

    def close_streams(self):
        self.stop.set()
        self.process.stdin.close()
        self.reader.join(timeout=0.25)
        if not self.reader.is_alive():
            self.process.stdout.close()


def recipes():
    return [
        dict(shape=list(shape), seed=seed, iterations=20)
        for shape in ((8, 16, 64), (16, 32, 128))
        for seed in (17, 29, 43)
    ]


def validate_fixture(value, config, identity):
    shape, seed = config["shape"], config["seed"]
    if (
        value.get("schema") != "spiraltorch.nd_bench.fixture.v1"
        or any(value.get(k) != v for k, v in config.items())
        or value.get("identity") != identity
        or value.get("gain") != 0.75
        or value.get("input")
        != [((i * 13 + seed) % 61) / 64.0 - 0.46875 for i in range(math.prod(shape))]
        or value.get("bias") != [(i % 5) / 32.0 - 0.0625 for i in range(shape[2])]
    ):
        raise ValueError("worker fixture/identity mismatch")


def validate_sample(value, shape, capture, rust):
    expected = [shape[1] - 1, shape[0], shape[2]]
    elapsed = value.get("elapsed_ms")
    values = value.get("values")
    if (
        value.get("shape") != expected
        or type(elapsed) not in (int, float)
        or not math.isfinite(elapsed)
        or elapsed <= 0
        or rust
        and value.get("finite_checked") is not True
    ):
        raise ValueError("invalid interval")
    if capture:
        if (
            not isinstance(values, list)
            or len(values) != math.prod(expected)
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            raise ValueError("invalid captured values")
    elif values is not None:
        raise ValueError("unexpected capture in timed response")


def run(binary: Path, output: Path, torch_device: str) -> None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    import numpy as np
    import torch

    if torch_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing CPU fallback")
    before, product = audit.source_identity(), audit.file_identity(binary)
    identity = audit.read_native_build_identity(binary)
    binding = audit.validate_source_binding(identity, before)
    if not binding["valid"] or audit.git_bytes(
        "ls-files", "--others", "--exclude-standard"
    ):
        raise ValueError(
            "freeze source and build a source-bound binary before measuring"
        )
    admission = (
        admit_device(torch_device) if torch_device == "mps" else {"device": "cpu"}
    )
    torch.set_num_threads(1)
    report = {
        "schema": "spiraltorch.nd_tensor.torch_bench.v1",
        "status": "error",
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "identity": identity,
        "torch_version": torch.__version__,
        "torch_device": torch_device,
        "source": before,
        "product": product,
        "source_binding": binding,
        "device_admission": admission,
        "fallback_enabled": False,
        "warmups": 2,
        "samples": 8,
        "cases": [],
        "worker_deadlines_seconds": {"response": 120.0, "exit": 10.0},
        "boundary": "resident inputs; 20x(add broadcast,mul scalar,tanh GELU); views and terminal contiguous host read included; upload/pipeline construction/JSON excluded; Rust preserves intermediate finite guards, Torch does not; uncontrolled OS load; diagnostic not universal speedup",
    }
    with output.open("x", encoding="utf-8") as handle, output.with_suffix(
        ".stderr.log"
    ).open("x", encoding="utf-8") as stderr:
        worker = None
        try:
            worker = Worker([str(binary)], stderr)
            request = worker.request
            for config in recipes():
                shape, seed = config["shape"], config["seed"]
                fixture = request({"op": "init", "config": config})
                validate_fixture(fixture, config, identity)
                if torch_device == "mps":
                    match_adapter(fixture["adapter"], torch_device, admission["name"])
                x = torch.tensor(
                    fixture["input"], dtype=torch.float32, device=torch_device
                ).reshape(shape)
                b = torch.tensor(
                    fixture["bias"], dtype=torch.float32, device=torch_device
                )
                gain = torch.tensor(
                    fixture["gain"], dtype=torch.float32, device=torch_device
                )
                if torch_device == "mps":
                    torch.mps.synchronize()
                case = {"fixture": fixture, "intervals": []}
                report["cases"].append(case)

                def torch_sample(capture):
                    with torch.no_grad():
                        start = time.perf_counter()
                        y = x.permute(1, 0, 2).narrow(0, 1, shape[1] - 1)
                        for _ in range(20):
                            y = torch.nn.functional.gelu(
                                (y + b) * gain, approximate="tanh"
                            )
                        host = y.contiguous().cpu().numpy()
                        elapsed = (time.perf_counter() - start) * 1000.0
                    if not np.isfinite(host).all():
                        raise RuntimeError("non-finite Torch output")
                    return {
                        "elapsed_ms": elapsed,
                        "shape": list(host.shape),
                        "values": host.reshape(-1).tolist() if capture else None,
                    }

                for iteration in range(10):
                    capture = iteration == 2
                    order = (
                        ["rust", "torch"]
                        if (iteration + len(report["cases"])) % 2 == 0
                        else ["torch", "rust"]
                    )
                    interval = {
                        "iteration": iteration,
                        "warmup": iteration < 2,
                        "order": order,
                    }
                    for lane in order:
                        sample = (
                            request({"op": "sample", "capture": capture})
                            if lane == "rust"
                            else torch_sample(capture)
                        )
                        validate_sample(sample, shape, capture, lane == "rust")
                        interval[lane] = sample
                    case["intervals"].append(interval)
                    if capture:
                        a = np.asarray(interval["rust"]["values"], dtype=np.float32)
                        bvalues = np.asarray(
                            interval["torch"]["values"], dtype=np.float32
                        )
                        if not np.isfinite(a).all() or not np.allclose(
                            a, bvalues, atol=1e-6, rtol=1e-4
                        ):
                            raise RuntimeError("Rust/Torch mismatch")
                        case["max_abs_error"] = float(np.max(np.abs(a - bvalues)))
                medians = {
                    lane: statistics.median(
                        i[lane]["elapsed_ms"]
                        for i in case["intervals"]
                        if not i["warmup"]
                    )
                    for lane in ("rust", "torch")
                }
                case["median_ms"] = medians
                case["torch_over_rust"] = medians["torch"] / medians["rust"]
                print(
                    json.dumps(
                        {
                            "shape": shape,
                            "seed": seed,
                            "median_ms": medians,
                            "max_abs_error": case["max_abs_error"],
                        }
                    ),
                    flush=True,
                )
            worker.finish()
            if (
                audit.source_identity() != before
                or audit.file_identity(binary) != product
            ):
                raise RuntimeError("source or executable changed during measurement")
            if torch_device == "mps" and admit_device(torch_device) != admission:
                raise RuntimeError("GPU admission changed during measurement")
            report["status"] = "passed"
        except BaseException as error:
            report["error"] = repr(error)
            try:
                if worker is not None:
                    worker.abort()
            except Exception as cleanup_error:
                report["cleanup_error"] = repr(cleanup_error)
            raise
        finally:
            try:
                if worker is not None:
                    worker.close_streams()
            except Exception as cleanup_error:
                report["status"] = "error"
                report["cleanup_error"] = repr(cleanup_error)
                raise
            finally:
                json.dump(report, handle, indent=2, allow_nan=False)
                handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--torch-device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args()
    run(args.binary.resolve(), args.output, args.torch_device)

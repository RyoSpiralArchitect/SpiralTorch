#!/usr/bin/env python3
"""Same-device resident matmul: input/output allocation and transfer excluded."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys

_helper_path = Path(__file__).resolve().with_name("bench_backend_vs_torch.py")
_spec = importlib.util.spec_from_file_location("resident_bench_helpers", _helper_path)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
correctness, fixture = _helpers.correctness, _helpers.fixture
paired_timings, parse_sizes = _helpers.paired_timings, _helpers.parse_sizes


def validate_cuda_identity(device_count, device_name, expected_adapter):
    # WGPU 0.20 does not expose a portable UUID to match CUDA's device ordinal.
    if device_count != 1:
        raise RuntimeError("resident CUDA comparison requires exactly one visible CUDA device")
    if not expected_adapter.strip() or expected_adapter not in device_name:
        raise RuntimeError("CUDA device does not match expected WGPU adapter")


def parse_tiles(spec):
    tiles = [tuple(int(x) for x in part.split(",")) for part in spec.split(";")]
    if any(len(tile) != 3 for tile in tiles) or len(set(tiles)) != len(tiles):
        raise ValueError("tiles must be unique M,N,K triples")
    return tiles


def run(args):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[name] = "1"
    import spiraltorch as st
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.torch_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no fallback")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no fallback")
    if device.type == "cuda":
        validate_cuda_identity(torch.cuda.device_count(), torch.cuda.get_device_name(), args.expected_adapter)
    if not args.expected_adapter.strip():
        raise ValueError("expected adapter must not be empty")
    torch_sync = torch.cuda.synchronize if device.type == "cuda" else torch.mps.synchronize
    native = Path(st._rs.__file__).resolve()
    if args.native_prefix and not native.is_relative_to(args.native_prefix.resolve()):
        raise RuntimeError("native extension outside expected prefix")
    report = {
        "schema": "spiraltorch.resident_matmul_bench.v3",
        "tiles_mnk": parse_tiles(args.tiles_mnk),
        "kernels_requested": args.kernels,
        "platform": platform.platform(), "host": platform.node(), "python": sys.version,
        "torch": torch.__version__, "torch_device": str(device),
        "torch_gpu": torch.cuda.get_device_name() if device.type == "cuda" else "MPS",
        "visible_cuda_devices": torch.cuda.device_count() if device.type == "cuda" else None,
        "device_identity_boundary": "single visible CUDA GPU plus matching adapter names; no cross-API UUID attestation",
        "spiraltorch_native": str(native),
        "spiraltorch_native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "helper_sha256": hashlib.sha256(Path(__file__).with_name("bench_backend_vs_torch.py").read_bytes()).hexdigest(),
        "iterations": args.iters, "warmup": args.warmup, "dispatches_per_sample": args.dispatches,
        "dtype": "float32", "tf32": False, "int8": False,
        "boundary": "equal Python-loop resident matmul calls then backend synchronize; allocation/upload/readback excluded",
        "notes": "includes API/queue/host synchronization costs; not GPU timestamp or model-quality evidence",
        "cases": [],
    }
    for m, k, n in parse_sizes(args.sizes):
        for seed in args.seeds:
            case = {"shape_m_k_n": [m, k, n], "seed": seed}
            try:
                a, ha = fixture(m * k, seed)
                b, hb = fixture(k * n, seed + 1)
                case["input_sha256"] = [ha, hb]
                ta = torch.tensor(a, dtype=torch.float32).reshape(m, k).to(device)
                tb = torch.tensor(b, dtype=torch.float32).reshape(k, n).to(device)
                out = torch.empty((m, n), dtype=torch.float32, device=device)
                reference = ta.cpu().double() @ tb.cpu().double()
                torch.mm(ta, tb, out=out)
                case["correctness"] = {
                    "torch": correctness(out, reference, torch, 1e-4, 1e-5),
                }
                def torch_batch():
                    for _ in range(args.dispatches):
                        torch.mm(ta, tb, out=out)
                    torch_sync()

                functions = {"torch_resident": torch_batch}
                case["wgpu_variants"] = {}
                for tile, kernel in ((tile, kernel) for tile in parse_tiles(args.tiles_mnk)
                                     for kernel in args.kernels):
                    workspace = st.WgpuMatmul(m, k, n, tile_mnk=tile,
                                             kernel=None if kernel == "auto" else kernel)
                    name = "st_" + kernel + "_" + "x".join(map(str, tile))
                    info = workspace.adapter_info()
                    if info["device_type"] == "Cpu":
                        raise RuntimeError("software WGPU adapter is not admitted as GPU timing")
                    if args.expected_adapter not in info["name"]:
                        raise RuntimeError("WGPU adapter does not match expected adapter")
                    if tuple(workspace.tile_mnk) != tile:
                        raise RuntimeError("actual tile does not match requested tile")
                    if kernel != "auto" and workspace.kernel != kernel:
                        raise RuntimeError("actual kernel does not match requested kernel")
                    case["wgpu_variants"][name] = {
                        "tile_mnk": tile, "adapter": info, "kernel_request": kernel,
                        "kernel": workspace.kernel, "workgroup_size": workspace.workgroup_size,
                        "outputs_per_thread": workspace.outputs_per_thread,
                    }
                    workspace.upload(st.Tensor(m, k, a), st.Tensor(k, n, b))
                    workspace.dispatch()
                    case["correctness"][name] = correctness(
                        torch.tensor(workspace.readback().tolist(), dtype=torch.float64),
                        reference, torch, 1e-4, 1e-5)

                    def st_batch(workspace=workspace):
                        for _ in range(args.dispatches):
                            workspace.dispatch()
                        workspace.synchronize()

                    functions[name] = st_batch
                    workspace.synchronize()
                torch_sync()
                timings = paired_timings(functions,
                                         args.warmup, args.iters, lambda: None, seed)
                case["timings"] = timings
                case["torch_over_st_resident_batch"] = {
                    name: timings["torch_resident"]["median_ms"] / timings[name]["median_ms"]
                    for name in case["wgpu_variants"]
                }
                case["status"] = "passed"
            except Exception as error:
                case.update(status="error", error=f"{type(error).__name__}: {error}")
            report["cases"].append(case)
            print(json.dumps(case, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="64,64;256,256;512,512;1024,64x64,32")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 29, 43])
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dispatches", type=int, default=16)
    parser.add_argument("--tiles-mnk", default="8,8,16")
    parser.add_argument("--kernels", nargs="+", choices=["auto", "scalar", "register_2x2"], default=["auto"])
    parser.add_argument("--torch-device", choices=["cuda", "mps"], default="cuda")
    parser.add_argument("--expected-adapter", required=True)
    parser.add_argument("--native-prefix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.iters, args.dispatches) < 1 or args.dispatches > 1024 or args.warmup < 0:
        parser.error("positive iterations, 1..1024 dispatches, nonnegative warmup required")
    parse_sizes(args.sizes)
    parse_tiles(args.tiles_mnk)
    if len(set(args.kernels)) != len(args.kernels):
        parser.error("kernel choices must be unique")
    with args.output.open("x") as stream:
        try:
            report = run(args)
        except Exception as error:
            report = {"status": "setup_error", "error": f"{type(error).__name__}: {error}"}
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return int("cases" not in report or any(x["status"] != "passed" for x in report["cases"]))


if __name__ == "__main__":
    raise SystemExit(main())

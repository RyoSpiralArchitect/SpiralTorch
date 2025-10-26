#!/usr/bin/env python3
import argparse
import random
import statistics as stats
import time

import spiraltorch as st
import torch


class SafeMatmulRunner:
    """Invoke ``Tensor.matmul`` while gracefully falling back on backend errors."""

    def __init__(self, requested_backend: str, fallback_backend: str = "faer"):
        self.requested_backend = requested_backend
        self._active_backend = requested_backend
        self._fallback_backend = fallback_backend
        self._chain = [requested_backend]

    def __call__(self, a: st.Tensor, b: st.Tensor):
        backend_to_try = self._active_backend
        try:
            return a.matmul(b, backend=backend_to_try)
        except RuntimeError as err:
            fallback = self._choose_fallback(err, backend_to_try)
            if fallback is None or fallback == backend_to_try:
                raise

            err_msg = str(err).strip().splitlines()[0]
            print(
                f"[SpiralTorch] backend '{backend_to_try}' unavailable ({err_msg}). "
                f"Falling back to '{fallback}'."
            )
            self._active_backend = fallback
            if self._chain[-1] != fallback:
                self._chain.append(fallback)
            return a.matmul(b, backend=self._active_backend)

    def _choose_fallback(self, err: RuntimeError, backend: str):
        message = str(err).lower()
        if backend in {"wgpu", "auto"} and (
            "wgpu backend failure" in message or "wgpu adapter" in message
        ):
            return self._fallback_backend
        return None

    @property
    def active_backend(self) -> str:
        return self._active_backend

    def describe_backend(self) -> str:
        if len(self._chain) == 1:
            return self._chain[0]
        return "->".join(self._chain)


def sync_torch(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        try:
            torch.mps.synchronize()
        except Exception:
            pass


def flat_rand(rows: int, cols: int, seed: int = 0):
    rnd = random.Random(seed)
    return [rnd.uniform(-0.5, 0.5) for _ in range(rows * cols)]


def st_tensor(rows: int, cols: int, seed: int = 0) -> st.Tensor:
    return st.Tensor(rows, cols, flat_rand(rows, cols, seed))


def torch_tensor(rows: int, cols: int, device: torch.device, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tensor = torch.empty((rows, cols), dtype=torch.float32)
    tensor.uniform_(-0.5, 0.5, generator=generator)
    return tensor.to(device)


def bench_once(fn, iters: int = 30, warmup: int = 5, sync=None):
    for _ in range(warmup):
        fn()
        if sync:
            sync()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        if sync:
            sync()
        end = time.perf_counter()
        times.append((end - start) * 1000.0)

    return {
        "best_ms": min(times),
        "median_ms": stats.median(times),
        "per_call_us": 1000.0 * stats.median(times),
        "n": iters,
    }


def fmt(res: dict) -> str:
    return (
        f"best={res['best_ms']:.3f} ms, median={res['median_ms']:.3f} ms, "
        f"per-call≈{res['per_call_us']:.2f} µs"
    )


def parse_sizes(spec: str):
    pairs = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "x" in chunk:
            left, right = chunk.split("x", 1)
            m, k = map(int, left.split(","))
            k2, n = map(int, right.split(","))
            if k != k2:
                raise ValueError("inner dims mismatch in spec")
        else:
            k, k2 = map(int, chunk.split(","))
            if k != k2:
                raise ValueError("square spec requires equal dims")
            m = n = k
            k = k2
        pairs.append((m, k, n))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="512,512;1024,1024;1024,2048x2048,1024",
        help="shapes like 'M,N;K,K;A,BxB,A' => pairs for (M,K)@(K,N)",
    )
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--torch-device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
    )
    parser.add_argument(
        "--st-backend",
        default="auto",
        help="auto|faer|cpu-simd|python-simd|naive|wgpu",
    )
    args = parser.parse_args()

    if args.torch_device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.torch_device)
    print(f"[Torch] device = {device}")

    pairs = parse_sizes(args.sizes)

    has_mm = hasattr(st.Tensor, "matmul")
    has_hadamard = hasattr(st.Tensor, "hadamard")
    if not has_mm:
        print("[SpiralTorch] Tensor.matmul is not exposed to Python. Apply the tiny pyo3 patch first.")
        return

    for (m, k, n) in pairs:
        print(f"\n== GEMM {m}x{k} @ {k}x{n} ==")

        a_st = st_tensor(m, k, seed=0)
        b_st = st_tensor(k, n, seed=1)
        a_th = torch_tensor(m, k, device, seed=0)
        b_th = torch_tensor(k, n, device, seed=1)

        matmul_runner = SafeMatmulRunner(args.st_backend)
        res_st = bench_once(
            lambda: matmul_runner(a_st, b_st),
            iters=args.iters,
            warmup=args.warmup,
            sync=None,
        )
        res_th = bench_once(
            lambda: torch.mm(a_th, b_th),
            iters=args.iters,
            warmup=args.warmup,
            sync=(lambda: sync_torch(device)),
        )
        backend_label = matmul_runner.describe_backend()
        print(f"SpiralTorch.matmul[{backend_label:>5}]  -> {fmt(res_st)}")
        print(f"Torch.mm[{device.type:>4}]               -> {fmt(res_th)}")

        if has_hadamard:
            c_st = st_tensor(m, n, seed=2)
            d_st = st_tensor(m, n, seed=3)
            c_th = torch_tensor(m, n, device, seed=2)
            d_th = torch_tensor(m, n, device, seed=3)
            res_st_h = bench_once(
                lambda: c_st.hadamard(d_st),
                iters=args.iters,
                warmup=args.warmup,
                sync=None,
            )
            res_th_h = bench_once(
                lambda: c_th * d_th,
                iters=args.iters,
                warmup=args.warmup,
                sync=(lambda: sync_torch(device)),
            )
            print(f"SpiralTorch.hadamard         -> {fmt(res_st_h)}")
            print(f"Torch.mul                    -> {fmt(res_th_h)}")


if __name__ == "__main__":
    main()

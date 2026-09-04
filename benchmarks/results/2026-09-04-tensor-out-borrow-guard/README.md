# Python Output Borrow Guard, 2026-09-04

The binding previously dropped `PyRefMut` before releasing the GIL, then wrote
through a `usize`-transported raw pointer. A competing or reentrant Python call
could borrow the same destination while that write was active. The fix retains
the PyO3 guard and passes only a safe `&mut Tensor` into the detached closure.
It also removes the raw pointer from the locally allocated packed-matmul path.

All six `out=` entrypoints use one Rust-owned helper. GELU still replaces the
output through its existing allocating implementation; no new math kernel or
numerical policy is claimed here. There is no global mutex or added output
snapshot. External mutable DLPack aliases remain a separate interop boundary.

## Validation

- Two embedded-Python Rust tests passed. A competing thread acquired the GIL
  during the detached closure, but both shared/mutable destination borrows
  were rejected. Reentrant borrowing was also rejected. Successful writes kept
  the original storage pointer, and errors released the guard.
- Six real-wheel Python tests passed, including strict WGPU output reuse on
  Furnace's non-CPU adapter. Five CPU tests also passed against the old wheel,
  confirming the preserved serial-use behavior and exception classes.
- Existing Python unittest smoke: 37 passed. CPU-only cargo check and the full
  CUDA/WGPU/Golden-enabled private wheel build passed; pinned rustfmt passed.
- The existing Python CI smoke job now runs `test_tensor_out.py`; its optional
  real-GPU case requires explicit runtime opt-in, not a silent CPU substitute.

The Rust tests were run with a command-scoped link argument pointing to the
installed `/usr/lib/x86_64-linux-gnu/libpython3.14.so.1.0`, because extension-module
builds intentionally omit that embedding link. No system configuration changed.
Local logs: `~/.local/state/spiraltorch/validation/tensor-out-borrow-guard-v1/`.

## Serial Throughput Check

Same Furnace x86_64 Linux host, private 0.4.27 development wheels, float32 CPU
math, one thread, sizes 128/256, seeds 17/29/43, 5 warmups and 20 samples.
Inputs and packed RHS are prepared before timing. Returned output handles are
discarded inside each timed call so the timing helper cannot retain a stale
copy-on-write alias into the next call. Ordinary core scratch allocations are
not excluded: this is a Python host-Tensor call boundary, not kernel timing.

Median of the three seed medians, milliseconds per call:

| Shape | Operation | Before | Guarded | PyTorch CPU (guarded run) |
| --- | --- | ---: | ---: | ---: |
| 128 cubed | Faer `out=` | 0.04894 | 0.04884 | 0.02829 |
| 128 cubed | packed `out=` | 0.33970 | 0.32517 | 0.02829 |
| 256 cubed | Faer `out=` | 0.37074 | 0.36401 | 0.22930 |
| 256 cubed | packed `out=` | 4.83554 | 4.83782 | 0.22930 |

All 18 correctness checks in each run passed the unchanged float64-reference
gate (`rtol=1e-4`, `atol=1e-5`). Variants are paired within a run, but the old and
new wheels were measured in separate processes. This is a bounded diagnostic
with no confidence interval, not a statistically established speedup. It did
not reveal a material throughput regression from retaining the borrow guard.
The packed CPU path is still much slower than PyTorch; the safety fix does not
resolve that pre-existing performance gap.

`baseline.json.gz` and `candidate.json.gz` retain every raw timing, input hashes,
extension hashes and harness hashes. The baseline is the private PR2063 wheel;
the candidate uses PR2063 head `fd648c84f204c756d4d250f77f82571a07b6b4a8` plus
the binding change. Its `bindings/st-py/src/tensor.rs` SHA-256 was verified on
both worktrees as `7e1cb22a24c4e12f7ddea543a68798f764670cde2c0deb6e7bfabcb05df47952`.
These are not PyPI artifacts. Unsafe concurrent accesses were not exercised
against the old wheel; race prevention evidence comes from the new Rust tests
and the removal of the raw-pointer bypass, not from these serial timings.

Reproduce with the intended installed wheel, in an otherwise idle test process:

```bash
python -P benchmarks/results/2026-09-04-tensor-out-borrow-guard/bench_out.py \
  --helper tools/bench_backend_vs_torch.py \
  --native-prefix /absolute/private/venv --output /absolute/new-output.json
```

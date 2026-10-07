# Captured Topos Learning

Topos retains the finite Picard map, not an implicit fixed-point derivative:
`state[0] = 0; state[n+1] = saturate(input * gate + coupling * state[n])`.
`ToposResonatorOperator::capture` stores the audited output and the exact
finite-unroll drive sensitivity in Rust. Repeated VJPs do not rerun the
recurrence or consult mutable caller inputs. Forward audit calculations,
shape/finite/budget checks and derivative multiplication order are unchanged.

```python
from array import array
from spiraltorch import ToposResonatorKernel

kernel = ToposResonatorKernel(coupling=0.25, iterations=4, porosity=0.2)
batch = kernel.capture_buffer(array('f', [0.2, -0.3]), array('f', [0.5, 0.8]), 1, 2)
output = memoryview(batch.output_buffer()).cast('f')
dx, dg = batch.vjp_buffer(array('f', [1.0, 1.0]))
```

Sequence clients can use `capture`, `output`, and `vjp`; legacy stateless
`forward`/`backward` remain available, with corresponding bulk-buffer methods.
Bulk inputs must expose a C-contiguous native-endian float32 buffer. Outputs
are independent bytearrays. Inputs are copied into an owned immutable tape;
this is not zero-copy. A tape retains four float32 vectors of element count N
(input, per-element gate, sensitivity, output), about **16 N bytes** plus
metadata, excluding temporary transport buffers and the client's saved tensors.

`topos_resonator_autograd` and `ToposResonatorAdapter` use the same capture API.
Torch still handles gate broadcasting and reduction; Rust returns both full
per-element gradients. Only the upstream gradient is uploaded during backward.
Optional NumPy enables bulk transport; the sequence fallback has identical
semantics. Saved Torch tensors still enforce in-place version checks. These
are first derivatives only, and the execution backend remains **Rust f32 CPU**
with explicit host transfers for accelerator tensors, not GPU residency.
Inference (`no_grad`, `inference_mode`, or neither input requiring gradients)
uses the stateless forward and does not allocate the learning tape.

WASM exposes `kernel.capture(Float32Array, Float32Array, rows, features)`,
`batch.output`, `batch.audit_json()`, and `batch.vjp(upstream)`. Pullbacks have
typed `grad_input` and `grad_gate` arrays. Copies outlive their Rust handles;
call `.free()` on batches and pullbacks when finished. A batch can outlive its
kernel. Direct Rust consumers use the same `ToposResonatorLearningBatch` core.

Rust callers with owned `Vec<f32>` inputs can use
`ToposResonatorOperator::capture_owned(input, gate, rows, features)` to transfer
both allocations into the tape without cloning them. The vectors are consumed
even on error, and any spare capacity is retained; borrowed `capture` remains
available with its existing behavior.
Python sequence/buffer capture and WASM capture use this owned path after
establishing Rust ownership. Foreign inputs are still copied and remain safe
to modify or discard after capture. This removes two internal N-element copies,
not the foreign-memory safety copy, and does not change the four-vector tape
or imply whole-process memory or throughput gains. See the
[owned-capture measurements](../benchmarks/results/2026-10-07-topos-owned-capture/README.md).

For matched performance comparisons, `tools/benchmark_topos_learning.py`
requests both input and broadcast-gate gradients on every route: legacy list,
bulk with recurrence recomputation, public captured bulk, and an independent
Torch finite unroll. Native routes must match bitwise; Torch comparisons allow
the documented f32 tolerance. `tools/probe_topos_capture_wasm.mjs` checks scalar
WASM parity, learning and saved-gate continuation in Node, not WebGPU speed.

The [matched measurements and single-update migration replay](../benchmarks/results/2026-10-07-topos-captured-vjp/README.md)
publish all conditions, hashes and numerical receipts, not model weights or text.

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

## Rust NN Learning

The CPU route of `st_nn::ToposResonator` now captures that same core tape during
forward. Backward calls `batch.vjp_audited(upstream)`, which checks finite
gradients and the sensitivity bound and computes the same backward audit from
the saved transition. It does not run the finite recurrence twice again.
This is an audit of Rust-owned results, not independent verification of an
external executor; WGPU results still use the full formula-comparison audit.

The NN cache shares its immutable tape across repeated band pullbacks, checks
input and gate consistency bit-for-bit (including signed zero), and invalidates
it on mutable parameter access or topos/config changes. CPU/WGPU routing can
change between forward and backward;
only an actual CPU captured pullback reports `capture_reused: true`.

Capture trades extra forward work and retained storage for cheaper backward.
The core tape owns four N-element vectors, while the returned NN output is a
separate Tensor; the older NN cache shared the caller's Tensor storage. This
is not an inference-speed or peak-memory improvement. Stateless
`ToposResonatorOperator::forward` and Python's no-grad path remain available
without capture. Tests compare saturated and unsaturated gradients/audits and
100 synthetic SGD updates against recomputation, not language-model quality.

### Shared Gates For Variable Batches

`ToposResonator::with_shared_gate` owns a `(1, features)` trainable gate instead
of one parameter per input element. Each forward accepts a different row count;
the supplied topos must admit the full `rows * features` volume. Existing
constructors keep their elementwise gate and exact-shape behavior.

```rust
use st_nn::{Module, OpenCartesianTopos, Tensor, ToposResonator, ToposResonatorConfig};

let features = 8;
let max_rows = 32;
let topos = OpenCartesianTopos::new(-1.0, 1e-6, 1.0, 16, max_rows * features)?;
let mut layer = ToposResonator::with_shared_gate(
    "topos.gate", features, ToposResonatorConfig::new(0.2, 5)?, topos,
)?;
let input = Tensor::from_fn(3, features, |r, c| (r + c) as f32 * 0.1)?;
let output = layer.forward(&input)?;
let upstream = Tensor::from_fn(3, features, |_, _| 1.0)?;
let grad_input = layer.backward(&input, &upstream)?;
assert_eq!(grad_input.shape(), input.shape());
assert_eq!(layer.parameter().gradient().unwrap().shape(), (1, features));
```

Rust expands the shared gate for the core recurrence and sums its elementwise
VJPs over rows using the existing fallible `sum_axis0` primitive. There is **no
additional row average**: a mean-loss factor belongs in the upstream gradient.
CPU uses the primitive's f64 accumulator with finite f32 output checks; the
WGPU VJP route requests its WGPU reduction. The reduction emits its own typed
execution receipt. Nonfinite reduction results are rejected before adding a
gate gradient or committing the backward audit. The core backward audit still
covers the expanded elementwise VJPs, not the reduced shared-parameter gradient.

Input, gate and upstream tensors are normalized by logical row/column layout,
including the existing elementwise mode. Cache checks compare canonical input
bits and an isolated canonical parameter snapshot, not the differently shaped
expanded shared gate. Synchronized writes through a writable DLPack alias are
detected before backward; WGPU cached operands are isolated as well. On
successful accumulation, parameter values and Tensor-backed gradients are
normalized together to row-major optimizer storage. Tape gradients already use
that logical coordinate order. Metadata distinguishes `gate_rows` / `trainable_parameters` from
`expanded_gate_rows` / `expanded_gate_values` and labels the raw-sum reduction.

This is a host-Tensor Rust NN layer, usable in ordinary `Sequential`; it does
not make that graph GPU-resident. `Sequential::backward` currently recomputes
its activations, so direct-layer capture savings must not be generalized to
the whole graph. Isolated parameter and WGPU operand snapshots can add copies;
previous capture timings are not measurements of this revised path.
Existing Python Torch adapters still perform broadcast and
reduction in Torch; the WASM core kernel still takes a full per-element gate.
This change does not add the shared-gate NN constructor to either frontend.

The correctness-only example and independent Torch checker run two 100-update
synthetic SGD trajectories using `Parameter::apply_step`, variable row counts
and both tensor layouts:

```sh
cargo run --locked --release -p st-nn --example topos_shared_gate_probe -- /tmp/topos-shared-new.json
python tools/check_topos_shared_gate_learning.py /tmp/topos-shared-new.json /tmp/topos-shared-check-new.json
```

Both commands require fresh output paths. The checker uses a true `(1, F)`
Torch leaf gate, native broadcasting, autograd and mean loss, keeping its own
weights across all updates rather than resetting to Rust weights. This is not
a speed comparison or evidence about pretrained-model quality.
Reference values, observed values and comparison errors must remain finite;
the saved Rust f64 loss is not downcast to f32 for comparison. A historically
passing fixture does not cover every guard: the separately appended
[review corrections](../benchmarks/results/2026-10-07-topos-shared-gate-review.json)
retain the new adversarial failures and corrected validation. The original
bundle's `parent_revision` means comparison base, not immediate source parent;
its measured source's immediate parent is `7c9d4f33161685d589af18ae6f3710b817c4dd43`.

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

The browser page `bindings/st-wasm/tests/topos_resonator_learning.html` uses
captured VJPs for its gate-learning loop. Its shared `.mjs` contract also runs
in Node CI through `tools/probe_topos_browser_learning.mjs`, keeping browser
and CLI checks aligned. The page reports import/fixture failures explicitly
and bypasses fixture HTTP caching. Native reference values must be finite
numbers; the real-WASM CI guard suite rejects 27 malformed output/input-VJP/
gate-VJP references, including nonfinite numbers and coercible strings. See the
[browser execution and reproduction record](../benchmarks/results/2026-10-07-topos-browser-captured-learning/README.md).

The [matched measurements and single-update migration replay](../benchmarks/results/2026-10-07-topos-captured-vjp/README.md)
publish all conditions, hashes and numerical receipts, not model weights or text.

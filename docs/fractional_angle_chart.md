# Angular Shape Coordinates For GL History

The [completed learned-gain comparison](../benchmarks/results/2026-10-05-fractional-gain-study/README.md)
found an ordinary learned short filter better than either GL arm under its
fixed budget. Shape coordinates differed: an angle for the ordinary filter,
log-order for GL. This module permits a chart-aligned follow-up without
changing GL convolution, normalization, gain semantics or old checkpoints.
It does not establish that changing coordinates improves model quality.

## Rust Contract

`st_frac::learning::FractionalGlAngleChart` captures an immutable scalar map:

```text
alpha = 1 + 2*tan(angle)
d(alpha)/d(angle) = 2*(1 + tan(angle)^2)
-atan(1/2) < angle < pi/2
```

For positive alpha, the normalized strictly-past K=3 GL coefficients are
`gain * [-cos(angle), sin(angle)]` mathematically. K=32 still uses the full
GL coefficient family; changing its chart does not make it a two-tap filter.
Independent ordinary implementations remain comparison controls, not the
production backend.

Rust owns domain validation, map evaluation and scalar VJP/JVP. Input and
output are float32, with f64 intermediate evaluation and differential
storage. Float32 rounding means the composed short GL and ordinary paths
need not be bit-identical; derivatives refer to the real map, as in the
other learning operators. Near chart boundaries conditioning and float32
resolution deteriorate; this is not a replacement for every representable
positive log-order. Invalid angles, nonfinite directions and overflowing
gradient/tangent results fail. There is no wrapping, clipping or projection.

## Python And WASM

Python and WASM expose the same immutable `FractionalGlAngleChart`, including
`angle`, `alpha`, `alpha_derivative`, `vjp` and `jvp`. The native class remains
usable without Torch. Python's optional first-order AD client only transports
the scalar and delegates the math:

```python
import torch
import spiraltorch as st

angle = torch.tensor(0.4636476090008061, requires_grad=True)
alpha = st.fractional_gl_angle_autograd(angle)
kernel = st.FractionalGlKernel(kernel_len=3)
x = torch.randn(2, 16, 8, requires_grad=True)
log_gain = torch.tensor(0.3, requires_grad=True)
history = st.fractional_gl_history_log_gain_autograd(
    x, alpha, log_gain, axis=1, kernel=kernel,
)
history.square().mean().backward()

adapter = st.FractionalAngleGainHistoryAdapter(
    8, initial_angle=0.4636476090008061, initial_gain=5**0.5,
    kernel_len=3, strength=0.1,
)
```

The adapter is identity-initialized and has `2*F+2` parameters, registered
as `gate`, `local_gate`, `history_angle`, `log_gain`. Its distinct state
schema prevents loading it silently as the log-order adapter. The two
feature gates and gain remain redundant amplitude controls. Ordinary
angle controls can leave the positive-alpha chart; a matched study must
declare how that domain difference is handled before running.

The shared HF trainer reads the native `alpha` property for angular
adapters before/after each update and at endpoints. A post-update domain
exit is rejected before checkpoint publication, without silently editing
the parameter. Legacy log-order semantics and fields stay unchanged.

Only first-order reverse/forward AD is supported, not higher-order AD,
`torch.func` or vmap. Parameter and direction tensors must be scalar f32.
Each scalar pullback rounds independently before composition, consistent
with the f32 AD boundary; no fused cross-operation differential is claimed.
Sequence adapters still require complete unpadded prefixes, no packed
documents and no KV cache. Kernels run on CPU host data; device transport
is not GPU residency.

## Evidence Scope

Native tests compare map differentials to finite differences and compose
with both short and long ND GL. Python tests compare nonzero-gate outputs,
input and all parameter gradients with an independent ordinary short
filter, exercise buffer/sequence transports, joint JVP and interrupted
tiny-HF resume, and reject invalid updated charts before a new checkpoint.
The compiled-WASM fixture learns shape and gain on a synthetic target;
that is executable learning evidence, not pretrained language quality.

A subsequent real-model comparison must freeze its own recipe and client,
report all outcomes and preserve the preceding completed studies. No
held-out outcome or speed improvement is established by this API addition.

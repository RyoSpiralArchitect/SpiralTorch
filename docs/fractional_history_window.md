# Full-Normalization History Windows

The [angular study](fractional_angle_study.md) compares K=3 and K=32
filters, but changing K changes both the available history and normalization.
This Rust primitive separates those effects without redefining old results.

For the declared K and strictly-past coefficients c(alpha), a half-open
window [start,end) uses

```text
w_k = exp(log_gain) * c_k(alpha) / ||c_1,...,c_(K-1)||_2
y_t = sum(w_k * x_(t-k), start <= k < end)
```

Normalization and its alpha derivative still cover ALL declared past taps.
Rust masks coefficients and their derivatives only after normalization.
Changing the window does not renormalize retained taps. Input, alpha and
log-gain VJPs, selective scalar VJPs and joint JVP use the same immutable
snapshot. An integer-order tail can have zero output but nonzero order
derivative; it must not be detached as an apparent zero feature.

## Shared API

Rust callers use `FractionalGlKernel::forward_history_log_gain_window` with
a `Range<usize>`. Python and WASM expose the same method with integer
`lag_start, lag_end` bounds and the existing gain-learning snapshot type.
Python also supplies a native-f32 bulk-buffer variant.

The range must satisfy `1 <= start <= end <= K`. Empty or unobservable
windows return checked zero maps without unused coefficient recurrences.
Finite inputs, positive finite order, representable positive f32 gain,
shape and full-K allocation/product budgets still apply. No clipping,
wrapping, learned mask or new optimizer rule is added.

```python
import torch
import spiraltorch as st

x = torch.randn(2, 128, 768)
alpha = torch.tensor(0.65, requires_grad=True)
log_gain = torch.tensor(0.0, requires_grad=True)
kernel = st.FractionalGlKernel(kernel_len=32)
short = st.fractional_gl_history_log_gain_autograd(
    x, alpha, log_gain, axis=1, kernel=kernel, lag_window=(1, 3))
tail = st.fractional_gl_history_log_gain_autograd(
    x, alpha, log_gain, axis=1, kernel=kernel, lag_window=(3, 32))
(short + tail).square().mean().backward()
```

The two windows sum to the full operator up to final f32 rounding; bitwise
additivity is not promised. A full window uses the unchanged original
execution path. `(1,3)` on K=32 is NOT a separately normalized K=3 filter.

## Model Integration

Both `FractionalGainHistoryAdapter` and `FractionalAngleGainHistoryAdapter`
accept `lag_window`. They retain identity initialization and 2*F+2 trainable
parameters, with Rust owning the filter and differentials.

```python
adapter = st.FractionalAngleGainHistoryAdapter(
    768, initial_angle=-0.2, kernel_len=32, lag_window=(3, 32))
```

Windowed state records the exact lag bounds and rejects silent loading into
a full-history or differently windowed adapter. Default `lag_window=None`
keeps the old state shape and computation. Restore using the same window;
do not edit a saved study's recipe to turn it into an ablation.

CPU f32 host transport, complete unpadded prefixes, no packed document
boundaries, no KV cache, and first-order AD limitations remain unchanged.
This provides an actual trainable/inference operator, not an audit-only
mask. It does not yet prove which component caused the published language
loss improvement. Any post-training intervention is a new diagnostic with
its own immutable outputs, not a replacement for a matched training study.
No speed or browser/GPU residency claim follows from native/WASM tests.

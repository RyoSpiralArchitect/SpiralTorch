# Ordinary Control For Geometric Language Learning

[Language MLOps](../language_mlops.md) | [HF Trainer](hf-geometry-training.md)

**Source candidate after 0.4.31, not a released wheel.**
`ChannelAffineAdapter` is deliberately **not geometry**. It is the ordinary
channel-wise residual `x + strength * (x * gate + bias)`, with zero-initialized
gate and bias and exactly 2F trainable parameters. There is no token mixing,
saturation, projection, learned radius or implicit optimizer.

This control uses the same public artifact, attachment, document-safe dataset,
HF Trainer, clipping and checkpoint/resume route as the geometric adapters.
It replaces the need for an example-only subclass that cannot be exported.
It is not a full-FT, LoRA, attention or parameter-matched Transformer baseline.

## Use The Same Training Route

```python
import spiraltorch as st

# Choose an actual MLP output site and width for your pinned HF model.
ordinary = st.GeometryAdapterStack({
    "transformer.h.11.mlp": st.ChannelAffineAdapter(768, strength=0.1),
})
geometric = st.GeometryAdapterStack({
    "transformer.h.11.mlp": st.WaveGateAdapter(
        768, strength=0.1, curvature=-1.0,
    ),
})
# Export each with the same verified base-runtime identity.
# Run each through hf_geometry_trainer_class(), not a separate learning loop.
```

WaveGate v1 above has no radius parameter. At curvature -1, zero gate/bias
give both arms identity output and equal first-update derivatives. The same
number of parameters is not sufficient to match training: also pin insertion
sites, initialization, data/tokenizer identity, document boundaries, seeds,
batch/accumulation, schedule, optimizer, clipping, dtype and evaluation policy.
Use a fresh frozen base and optimizer for each arm. Do not continue one arm
from the other's artifact or optimizer.

The [offline FT example](../../bindings/st-py/examples/hf_geometry_finetune.py)
accepts either artifact without changing its learning loop. Keep the bare-base
and initial-adapter references. Use held-out documents and multiple seeds;
report every declared arm, including failures and non-improvements. A small
workflow probe is not evidence that geometry improves language quality.

## One Numerical Core

Rust `st_nn::ChannelAffineKernel` reuses existing CPU `row_affine`, `mul_row`,
`hadamard` and row reductions. Its owned snapshot returns input, gate and bias
VJPs; parameter gradients are sums, not row averages. Shape and value budgets
are checked before copying directions; nonfinite values and overflow fail.

Python `ChannelAffineKernel.forward` / `forward_buffer` expose this directly.
The buffer route accepts native contiguous float32 buffers. The optional
`channel_affine_autograd` transports Torch tensors of shape `[..., features]`;
only first-order reverse AD is supported. Inputs and parameters saved for
backward retain Torch's mutation checks. Empty leading dimensions are allowed.
The adapter additionally rejects overflow in its completed residual addition.
An explicit zero-strength adapter bypasses the intervention.

The recipe `spiraltorch.channel_affine_adapter.v1` travels inside the existing
`GeometryAdapterArtifact` with gate/bias, strength, width and value budget.
Old geometric artifact schemas and the shared 13-placement fixture remain
unchanged. Reconstruction never runs arbitrary subclasses.

WASM `nn` exports `ChannelAffineKernel(maxValues)`,
`forward(Float32Array, gate, bias, rows, features)`, snapshot `output`,
`vjp(Float32Array)`, and pullback `grad_input`, `grad_gate`, `grad_bias`.
These are the affine **branch**, not the residual wrapper or an HF executor.
Native, Python and real WASM/Node use the same fixture.

Both this control and WaveGate currently execute their kernels on Rust f32
CPU. Torch MPS/CUDA tensors copy through CPU explicitly. This is not a resident
GPU benchmark, and wall-clock speed is not a geometry-quality metric.

The [six-run GPT-2 workflow qualification](../../benchmarks/results/2026-10-18-hf-affine-control/README.md)
retains all paired results and exact continuation checks. In that small,
previously used evaluation fixture, WaveGate does not beat this ordinary
control; no quality advantage or new default is claimed.

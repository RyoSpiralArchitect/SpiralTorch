# Geometric Operators In The Learning Path

Two evaluation tracks stay separate:

- Equivalent standard operators: compare numerical agreement, transfer boundaries,
  memory and throughput against PyTorch under matched computation.
- SpiralTorch-specific geometry: demonstrate forward effects, correct gradients,
  accepted parameter updates and checkpoint continuation, then compare learning
  quality/stability with disabled and simple-capacity controls. Record extra cost;
  do not call different mathematics a PyTorch speedup.

## First Connection: Topos Resonance

The existing Rust ToposResonator now has an immutable core operator and thin
Python/WASM clients. It computes a finite Picard recurrence, element by element:

```text
drive = input * gate
state[0] = 0
state[t+1] = OpenCartesianTopos.saturate(drive + coupling * state[t])
```

The core owns the rewrite and exact first-order VJP of the finite unroll, not an
implicit fixed-point approximation. Python does not reconstruct this derivative.
Within the unsaturated region the recurrence is a linear gain. Outside it,
zero porosity clamps; nonzero porosity uses the existing porous rewrite, whose
tail slope can be negative. This is not a generic smooth monotone activation,
global topological reasoning, or evidence that all geometric mechanisms help.

### Python And Hugging Face

```python
import torch
from spiraltorch import ToposResonatorAdapter

adapter = ToposResonatorAdapter(
    features=768, strength=0.1, coupling=0.25, iterations=4,
    saturation=1.0, porosity=0.2, max_values=1_048_576,
)
hidden = torch.randn(2, 32, 768, requires_grad=True)
adapted = adapter(hidden)
adapted.square().mean().backward()
assert adapter.gate.grad is not None
```

The adapter returns `x + strength * resonance(x, gate)`. Its per-feature gate
starts at zero: output is initially unchanged, but the gate can receive a
gradient immediately. `strength=0` is an exact bypass without a kernel call.
The operation accepts any nonempty final feature axis, broadcasts the gate,
and never mixes tokens. Shape/finite-value errors fail explicitly.

For an already loaded **float32** GPT-2-shaped HF model, placement is explicit:

```python
model.requires_grad_(False)
adapter = ToposResonatorAdapter(model.config.n_embd, strength=0.1)
adapter.to(next(model.parameters()).device)
block = model.transformer.h[0]
block.mlp = torch.nn.Sequential(block.mlp, adapter)
optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)
optimizer.zero_grad()
loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
loss.backward()
optimizer.step()
```

Mask ignored/padded labels with `-100` as appropriate for the model's ordinary
loss. This is an insertion example, not a universal model patcher: other
architectures need an explicitly chosen tensor-valued block and matching width.
Tuple/dict outputs, sharded models and quantized/mixed-precision FT are not
automatically adapted. The underlying functional API is
`topos_resonator_autograd(input, gate, kernel=ToposResonatorKernel(...))`.

**Execution boundary:** this first bridge runs Rust f32 on CPU. GPU inputs and
upstream gradients cross the host boundary; outputs/gradients return to the
original device. The `execution_backend` label remains `rust_f32_cpu`, even
on MPS/CUDA. It is not a resident WGPU implementation. Higher-order autograd,
AMP, `torch.compile`, DDP and throughput improvements are not claimed.

Checkpoint the adapter using ordinary `torch.save(adapter.state_dict(), ...)`
and the optimizer state. The adapter state includes a versioned primitive-data
recipe and the learned gate; `torch.load(..., weights_only=True)` is supported.
Recreate the same placement before loading. Save the adapter separately from
HF `save_pretrained`/safetensors: its non-tensor extra state is not a native HF
adapter format. Model identity, placement, data cursor and RNG remain the
training orchestrator's responsibility. A live backward retains the immutable
kernel recipe used by its forward; in-place input/gate changes are rejected by
Torch's saved-tensor version checks.

### Rust And Browser

Rust callers use `st_core::dynamics::topos_resonator::ToposResonatorOperator`
with a validated `ToposResonatorConfig` and `OpenCartesianTopos`. The existing
`st_nn::layers::ToposResonator` remains the parameter-owning Rust NN layer.

```javascript
import init, {ToposResonatorKernel} from "./spiraltorch_wasm.js";
await init();
const kernel = new ToposResonatorKernel(0.25, 4, 1.0, 0.2, 1024);
try {
  const input = new Float32Array([0.2, -0.4]);
  const gate = new Float32Array([0.3, -0.2]);
  const output = kernel.forward(input, gate, 1, 2);
  const gradients = kernel.backward(input, gate, new Float32Array([1, 1]), 1, 2);
  // gradients.grad_input and gradients.grad_gate are per-element VJPs.
} finally {
  kernel.free();
}
```

The browser client executes f32 Rust in WASM linear memory, without WebGPU.
All constructor arguments are explicit. Rows/features/counts reject fractional,
negative, nonfinite and coercible non-number JS values before integer conversion.
Callers own the optimizer and broadcast reductions, not a second geometric rule.

## Evidence And Next Operators

See the [complete wiring experiment](../benchmarks/results/2026-10-02-topos-learning-bridge/README.md).
It includes native finite differences inside/outside saturation, actual HF loss
updates, adapter/optimizer continuation, and browser/native forward/VJP agreement.
The random tiny HF experiment is deliberately not pretrained FT evidence or a
claim of improved language quality. It preserves adverse development-loss results.

The [elliptic/Lie bridge](elliptic_learning_bridge.md) now has numerical repairs,
batched Rust VJPs and a bounded pretrained-model connection. Next expose the
existing Rust WaveGate derivatives under the same learning/continuation tests.
Only after those individually work should combinations be evaluated. Each needs
an identity or disabled control, a simple parameter-count control, fixed data and
seeds, saturation/gradient diagnostics and an explicitly priced execution boundary.
Resident acceleration follows demonstrated demand; it must preserve the same VJP.

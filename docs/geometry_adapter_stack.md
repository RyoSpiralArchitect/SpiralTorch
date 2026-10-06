# Shared Geometry Adapter Placement

`from spiraltorch import GeometryAdapterStack` connects existing Rust-backed
Torch adapters to explicitly chosen model outputs. It replaces repeated manual
module surgery, not any geometric operator, derivative or optimizer rule.
Different families can share this placement API without gaining a second
Python implementation of their mathematics.

```python
import torch
from spiraltorch import (
    GeometryAdapterStack, WaveGateAdapter, FractionalAngleGainHistoryAdapter,
)

# An already loaded float32 HF model; adapt paths and widths to its architecture.
model.requires_grad_(False).eval()
model.config.use_cache = False
geometry = GeometryAdapterStack({
    "transformer.h.0.mlp": WaveGateAdapter(768, strength=0.1),
    "transformer.h.1.mlp": FractionalAngleGainHistoryAdapter(
        768, kernel_len=32, initial_angle=0.2, strength=0.1,
    ),
})
geometry.to(next(model.parameters()).device)
optimizer = torch.optim.Adam(geometry.parameters(), lr=1e-3)

with geometry.attach(model):
    optimizer.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
    loss.backward()  # Complete autograd before leaving the attachment scope.
    optimizer.step()

# All placement hooks are now gone. Base names, modules and flags were not changed.
torch.save({"geometry": geometry.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
saved = torch.load(checkpoint, weights_only=True)
geometry.load_state_dict(saved["geometry"])
optimizer.load_state_dict(saved["optimizer"])
```

## Ownership And Lifetimes

- The stack owns its adapters. Model parameters, buffers, module paths, training
  mode and `requires_grad` flags stay untouched. Optimize `geometry.parameters()`;
  `model.parameters()` intentionally does not include these separate parameters.
- Move adapters explicitly to the input device and set their train/eval mode
  separately when an adapter has mode-sensitive behavior. The built-in geometry
  remains **Rust f32 CPU with explicit host transfers**, including on GPU tensors.
  This API is not resident execution or a speed optimization.
- `attach` is a context manager, not a persistent global patch. It removes only
  its own hooks after success, exceptions or partial installation failure.
  Existing user hooks and keyword-based module interfaces are preserved.
- Backward must complete inside the same scope. Reusing an old graph after
  detach or in a later attachment raises rather than silently replaying a graph
  with missing placement. `torch.no_grad()` inference inside a scope is supported.
- Paths and concrete adapter types are saved in mapping order. Loading rejects
  a reordered/missing placement header before descending into parameters, even
  through a parent module or with `strict=False`. Loading while attached fails.
  Other malformed child state still follows normal Torch load behavior; loading
  arbitrary state is not an all-or-nothing transaction.

Save this state separately from HF `save_pretrained`/safetensors. The adapter
recipes include non-tensor extra state. Model identity, data schedule, optimizer
cursor and RNG are still the training orchestrator's responsibility, not facts
that a path-only header can prove.

## Explicit Boundaries

Targets must return a float32 tensor; adapters must preserve its shape, dtype
and device. Choose tensor-valued MLP/projection blocks rather than asking the
library to guess an element of a tuple/dictionary. Nested/aliased targets, shared
adapter ownership and duplicate active hooks on a target are rejected.

The stack is for full-prefix, unpadded, single-document sequences when a history
adapter is present. It does not infer masks, reset packed-document boundaries or
build a KV cache. It rejects enabled HF caching, explicit past caches and known
HF activation checkpointing; these guards are not a general validator of custom
model internals. Custom checkpointing, sharding, AMP, compiled/distributed
execution and concurrent use remain unsupported. DataParallel/DDP wrappers are
rejected because separate adapters would otherwise evade their synchronization.

## Tests And Offline Example

Tiny GPT-2 and Llama models exercise WaveGate, Topos, anchored elliptic geometry
and fractional history together. Their losses, gradients, adapter/Adam state and
saved-state continuation match independent manual insertion bitwise. Negative
tests cover wrong paths/types, stale autograd, aliases, cache calls and failed
hook installation. These are connection tests, not language-quality evidence.

The [recorded pretrained GPT-2 check](../benchmarks/results/2026-10-07-geometry-adapter-stack/README.md)
executes seven auxiliary updates with 12,294 trainable parameters at four MLPs.
All losses, parameter gradients, final adapter/Adam state and RNG match the
direct-insertion reference, including an in-process serialized-checkpoint
continuation. All twelve parameter tensors have nonzero gradients at update
three; the base remains unchanged. This is connectivity, not evidence that the
combination improves language quality or that tiny scalar gradients are useful.

The bounded local-model example uses the same public import, keeps placement
configuration outside model code and never downloads a model or corpus:

```bash
export SPIRALTON_MAGIC=0 SPIRALTON_TORCH=0 SPIRALTON_MODEL_PATCHES=0 SPIRALTON_NUMPY=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
PYTHONPATH="$PACKAGE_ROOT:$REPO/tools" "$PYTHON" -P -B \
  "$REPO/bindings/st-py/examples/hf_geometry_adapter_stack.py" \
  --config "$REPO/bindings/st-py/examples/hf_geometry_adapter_stack.json" \
  --model-dir "$MODEL_DIR" --corpus "$CORPUS" \
  --package-root "$PACKAGE_ROOT" --runtime-manifest "$RUNTIME_MANIFEST" \
  --output "$NEW_OUTPUT_DIRECTORY"
```

Use a versioned frozen runtime manifest (`source_revision` plus `files` hashes).
The example compares three auxiliary updates via manual insertion with three
via the public API, then resumes the manual midpoint for one public-API update.
It stores actual states privately and publishes no generation, heldout score
or timing. Mixed geometry's quality and cost must be measured against equally
specified ordinary controls, not inferred from exact placement parity.

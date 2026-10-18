# Native Language Execution and Optional HF Interoperability

This describes the source tree after PyPI 0.4.32, not the APIs in those immutable
wheels. Use the [changelog](../../bindings/st-py/CHANGELOG.md) and
[release record](../ops/release.md) to identify an installed distribution.

The design rule is **Rust owns computation; Python transports and orchestrates**.
An optional dependency is not the same as a dependency-free implementation of
every exported function. HF-named interoperability helpers still execute foreign
models; they must not be mistaken for the native path below.

## Choose the Execution Owner

| Work | Native entry point | Boundary |
| --- | --- | --- |
| Read local checkpoint tensors | [`SafeTensorArchive`, `NdTensor`](native-weights.md) | Rust safetensors ingress; no pickle, remote code or implicit model conversion |
| Execute a supported GPT-2 | [`NativeCausalLm.from_gpt2`](native-causal-lm.md) | Frozen base, f32 CPU storage, exact integer token IDs; not an arbitrary HF architecture loader |
| Learn attached geometry from language loss | [`geometry_loss_gradients`, `NativeGeometryAdamW`, `NativeCausalLmTrainer`](native-corpus-training.md) | Rust decoder pullback, optimizer and document cursor; base weights stay frozen |
| Add repetition-aware language learning | [Native repetition objective](native-repetition.md) | Shared Rust candidates, loss/VJP, two-normalizer reduction and pinned recipe; no Torch auxiliary loss |
| Generate with ordinary or Z-Space control | [`NativeCausalLmGenerator`, `NativeCausalLmContext`](native-generation.md) | Rust selection, RNG and explicit K/V reuse; no Torch generation loop |
| Execute portable geometry independently | [`NativeGeometryRuntime`](native-geometry-runtime.md) | Same admitted artifact recipes without a foreign model; standalone MSE is not LM fine-tuning |
| Train a native dense or low-rank layer | `st.nn.Linear`, `st.nn.LoraLinear` | Rust forward/VJP and SGD, Realgrad or AmegaHypergrad; LoRA freezes its base, but is not a native decoder attachment or PEFT importer |
| Run other HF models or PEFT adapters today | `load_hf_causal_lm_artifact`, `prepare_hf_finetune_model`, HF Trainer helpers | Explicit optional Transformers/Torch/PEFT path, not a fallback inside `NativeCausalLm` |

The native importer rejects unsupported configurations and extra tensors,
including unimported LoRA weights. It does not silently route them through Torch.
Tokenization remains external; supplying exact IDs needs no Python tokenizer.
Supported native geometry and source-owned checkpoint continuation are described
in the linked guides, including memory/work budgets and numerical policies.
Core WASM compilation does not imply a browser HF checkpoint importer, and the
native GPT-2 CPU path is not a resident WGPU execution claim.

## Remaining Foreign Computation

This inventory separates intentional transport from missing native execution:

- [`hf_peft.py`](../../bindings/st-py/spiraltorch/hf_peft.py) loads foreign model
  objects and calls PEFT to attach/train LoRA. Native `LoraLinear` is a useful
  layer primitive, but is not yet a PEFT-checkpoint executor or trainable LoRA
  attachment in `NativeCausalLm`.
- [`hf_geometry_trainer.py`](../../bindings/st-py/spiraltorch/hf_geometry_trainer.py)
  and [`geometry_adapters.py`](../../bindings/st-py/spiraltorch/geometry_adapters.py)
  retain foreign model execution, Torch hook/autograd transport and HF training
  orchestration. Their saved supported geometry has a separate native execution
  path; changing the client does not make an arbitrary foreign decoder native.
- [`hf_generation.py`](../../bindings/st-py/spiraltorch/hf_generation.py) still
  calls a foreign model's generation and uses Torch score transport/top-k in its
  logits processor. The native generator is the dependency-free alternative for
  the supported decoder, not a transparent replacement for every HF model.
- [`hf_repetition_unlikelihood.py`](../../bindings/st-py/spiraltorch/hf_repetition_unlikelihood.py)
  obtains candidate plans and control from Rust, but still computes log-softmax,
  clamped unlikelihood, position reduction and differentiation in Torch for
  foreign decoders. The separate [native objective](native-repetition.md) now
  owns this computation and its decoder pullback in Rust for the admitted
  GPT-2 path. This is not a claim that the HF wrapper itself is Torch-free.

The existing wrappers remain explicit interoperability clients. Removing an
import or changing a report's owner string is not a native implementation.

## Recovery Order

1. Qualify native repetition learning through the installed-wheel gates and
   matched corpus studies. Mechanics and numerical parity do not establish
   fewer generation loops or better language quality.
2. Connect native low-rank projections to decoder placements and weight loading,
   then optimizer/checkpoint ownership. Verify a real parameter update and exact
   continuation; a compatible artifact report or frozen forward is insufficient.
3. Extend explicit architecture support with the required attention, positional
   and normalization kernels. Admit only tested configurations; do not replace
   unsupported variants with approximations or hidden foreign execution.

For each milestone, test the installed wheel with optional ML imports forbidden,
retain all numerical comparison conditions and failures, and separate mechanical
correctness from language quality and speed. Conventional matched arithmetic is
the speed baseline; Z-Space/geometric interventions need their own ablations.

# Full-Answer Causal QKV Learning

Status: native implementation, CPU binding check, one CPU extension build and
isolated Python execution verified. No additional pretrained-model training,
old evaluation scoring, GPU execution, cache deletion, push or publication.

## Shared Native Contract

`NativeCausalLmQkvAdamW::causal_loss_gradients` and `step_windows` reuse
`prepare_objective`, `training_readout`, `decoder_parameter_pullback` and the
existing atomic QKV gradient application. Python transports bounded arrays and
returns native reports; shifting, masking, reduction, gradients, clipping and
AdamW remain in Rust. It exposes `loss_gradients` and `step_windows` with type
stubs, alongside unchanged one-answer-per-prefix methods.

Flat rectangular `[B,T]` windows predict shifted labels from `[B,T-1]` context.
Optional labels have unshifted `[B,T]` layout. `-100` omits loss, not context;
the first label is never scored. The gradient mean is over all selected target
tokens, not equally over sequences. No implicit padding, EOS, truncation,
attention mask, microbatch accumulation or repetition/rival objective exists.
Selected-token logit limits are checked after bounded labels are extracted,
before decoding; context/checkpoint limits are checked before foreign token
storage is read. This permits a sparse batch whose logits fit even when dense
logits, or one output per sequence, would not.

Only one chosen layer's fused Q/K/V weight matrix updates. Other base weights,
LoRA and geometry stay frozen. Global clipping is applied once after the whole
selected-token mean. Existing checkpoint and inference identity rules apply.
Raw gradient queries do not clip or mutate; zero learning rate does not change
weights, moments or the accepted-step clock. Separate window calls are separate
updates, and callers retain data order, labels and objective configuration.

## Checks And First Failure

The first native invocation stopped during test compilation: the new numerical
finite-difference fixture inferred its perturbation sign as f64, incompatible
with the fixture's f32 weights. The original failed log and source inventory are
preserved. Explicitly typing that test closure's sign as f32 fixed the issue;
production code, fixture values and numerical tolerances were not changed.

The corrected invocation passed 222 native decoder tests in 19.18 seconds
including compilation (2.04 seconds in test bodies). Four new tests cover:

- Full-window loss/gradients versus individual selected-token prefixes, with
  ordinary/key-lag attention, either QKV layer, smoothing off/on and frozen
  geometry/LoRA; a 1-versus-3 target batch distinguishes token and sequence means.
- Sparse admission, all-ignored/invalid labels and work bounds; future token
  changes leave earlier gradients exact, while masked context still influences
  them. Rejected updates leave the optimizer checkpoint unchanged.
- Analytical derivatives versus physical fused-QKV weight perturbations across
  Q/K/V partitions, either layer and both attention modes (tolerance 1e-4).
- Exact existing-answer/full-window single-target optimizer equality, clipped
  dense-window continuation after restore, and zero-rate state preservation.

The CPU binding check passed in 16.16 seconds. One offline CPU release extension
build passed in 115.64 seconds with two build jobs, incremental output disabled,
a 600-second cap and a 4 GiB free-space floor. No guard fired. The previous
saved extension remained byte-identical; nothing was installed into an existing
Python environment.

The new extension and current Python source were copied to a private 83-file
overlay with a pinned manifest. One fail-fast invocation passed all 72 tests,
with no errors, failures or skips, in 2.54 seconds including supervision
(0.315 seconds in test bodies). Torch, NumPy, Transformers, Python safetensors
and Accelerate import attempts and loaded modules were both zero. The two new
Python tests cover `CausalCompletion.training_document()` to masked QKV updates,
selected-prefix derivative agreement, restore, cached logits, bounded transport
and rejection without optimizer mutation. Existing generator/file-handoff
regressions remain in the same invocation.

These are synthetic numerical/transport/restart checks, not new evidence of
language quality, convergence, GPU behavior, speedup or released-wheel behavior.
The training path is CPU and updates one QKV matrix, not full-model SFT.

## Receipts

Private evidence directory: `2026-11-10-qkv-causal-windows-v1` in the local
SpiralTorch validation store. All supervised invocations verified their source
inventory remained unchanged. Only documentation was added after runtime checks.
Changed Rust files passed rustfmt checking; stable-toolchain warnings for the
repository's nightly-only formatting options remain. `git diff --check` passed.

| Evidence | SHA-256 |
| --- | --- |
| Initial failed native log | `136fe8f215f0b4530f8cac3e6c501ed826cd1cd22f0af8bc908703a4a1c53179` |
| Corrected tested source inventory | `3e6ff7719618a4fe6b567f57ae86107a5eb22028b35d4f62fd655b43bf1eb934` |
| Native 222-test log | `0fd330bd5e2062cff27f9a135a85f359863cfad1a0217b2b1117f6efe14df1c9` |
| CPU binding check log | `95dea51dd8250326b1fe18ff37163a1c9d1d2fb5c3f1d0d6198170ba8a47e04f` |
| CPU extension build log | `153f7882993e8e3dedc13da42ac9bc0e40a5ade965245dc82689f1bc330bec76` |
| CPU extension | `ea6aa60de51579d8b67bc1cf6e47cbcd6aa49c092079347d836890322e388319` |
| Frozen package manifest | `ec83210ef8a68a532221091b047a3ff78e65f5191c1c4b8c71d434dd9232b4df` |
| Python 72-test log | `f300edfb51be4b92bbefafa992f371fa2016bfcd1167f7fed223997532ddd557` |

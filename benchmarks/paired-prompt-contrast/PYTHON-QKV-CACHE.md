# Python QKV Snapshot Cache

Status: local CPU binding and synthetic-model execution verified. This is not
a wheel release, a new pretrained-model study or a throughput result. It
extends the [answer-QKV binding](PYTHON-QKV.md) with the existing Rust-owned
`NativeCausalLmProjectionPrefix`, rather than recreating a decoder in Python.
See the [usage guide](../../docs/python/native-causal-lm.md#cached-inference-after-qkv-updates).

## Contract

- `NativeCausalLmQkvAdamW.prefill` admits length and native cache budgets before
  materializing Python token values, then captures the current serving snapshot.
- The returned immutable prefix exposes `serving_id`, copied `token_ids`,
  `cache_values`, `max_cache_values`, `logits_f32le()` and `branch(token_id)`.
- A branch shares the native snapshot owner and advances native cache state.
  Neither a successful branch, a refused branch nor a later trainer update can
  change the parent prefix's weights, tokens or logits.
- Capacity counts retained f32 K/V and geometry history per prefix, not total
  process memory or the sum of live branches. Cache serialization, automatic
  sampling and EOS policy are not added here.

Rust owns the admission, arithmetic, snapshot identity, history and branching.
The Python binding only transports IDs, scalars and little-endian f32 bytes.
No geometry/optimizer rule was weakened to make the binding pass.

## Verification And First Failure

213 native decoder tests pass. The added native test covers ordinary and
lag-1 attention with frozen LoRA/geometry, length-only admission, exact cache
capacity, failed branches, old-snapshot stability after an update, and exact
agreement with full-prefix logits. The CPU binding check also succeeds.

The release library was built **once**, taking 185.71 seconds under a 600-second
limit and a 4 GiB free-space floor. Default features were disabled and only
`python-default` enabled. No WGPU/GPU execution was requested. The library was
copied into an isolated source-package overlay, not installed as a wheel.

The first Python run is retained as a failure. All six variants of the new
snapshot test stopped at model construction: the test incorrectly attached
LoRA to layer 1 but used the helper's default geometry at layer 0 MLP. The
existing Rust composition guard requires geometry at the same layer's
attention output. These variants never reached QKV updates or cache inference.
The 50 other test methods passed; the failing method recorded six subtest errors.

Only the test fixture was corrected to `transformer.h.1.attn`, matching the
existing Rust fixture and contract. After separate approval, one fail-fast
Python invocation against the **same library bytes** passed all 51 tests:
7 QKV/facade methods and 44 existing native causal-LM methods.

The new test crosses ordinary/key-lag attention with `f32`,
`f32_wide_backward` and `f64` readout. Each synthetic model has a frozen LoRA
attachment and nonzero affine geometry. It checks updated prefill, two divergent
two-token branches, byte-exact full-prefix logits, checkpoint-derived ownership,
and isolation from a subsequent QKV update. Another test checks admission before
reading token values and rejection of Boolean/negative/out-of-vocabulary tokens,
cache exhaustion and position overflow without altering prefix/trainer state.

`python3.12 -I -B -S` verified the extension path and SHA-256, and blocked imports
of Torch, NumPy, Transformers, Python safetensors and Accelerate. Both invocations
recorded zero attempted imports and zero loaded foreign ML modules. The success
does not erase or relabel the original failed invocation.

## Evidence

Local directory: `2026-11-10-python-qkv-cache-v1` in the SpiralTorch validation
store. Source inventories, logs, receipts, both test versions and the fixed
library are retained. Only three explicitly approved, rechecked-unused compiler
incremental cache directories were deleted; observed free space increased by
1,334,500 KiB. Existing wheels, model weights and experimental originals were
not deleted or overwritten. `cleanup-receipt.json` records the exact scope.

| Artifact | SHA-256 |
| --- | --- |
| CPU extension | `3497e4a24cd8df3fef1423c078dcf44c0bd5de45798f1a77bff7a5a5c27a2725` |
| Native 213-test log | `098bbade49d05307d96bf4272a7cb05a7f005feb439f72767e2020486a96ac43` |
| CPU binding-check log | `73939a31db1fe767f1e85a99f1bc0c0a901a9ddf191144da0c3257a4b2eb525f` |
| One library-build log | `b867649a3cd2f1c4d71a8e5676d79e52398e34e5fbee4f0fd301aa2c6775ccd7` |
| Initial Python failure log | `b22c3d34588f812bec9443ae551bf5f8fbf01e22888d88d80fcd7ad423548ada` |
| Corrected 51-test Python log | `e13d5c604aac6a798fb1001ebaeaf88be389ffc027ce1c13a34984346fddf714` |

There was no second library build, GPU run, pretrained-model update, new quality
evaluation, full CI run, push, version bump, tag or publication in this stage.

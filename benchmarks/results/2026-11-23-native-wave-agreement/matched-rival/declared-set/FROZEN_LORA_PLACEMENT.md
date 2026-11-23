# Train a relocated WaveGate without dropping its fixed LoRA

A native decoder can now promote its active attention-output LoRA into an
immutable derived base. The parent ID and the typed LoRA ID determine the
derived base identity. It runs the **same** low-rank forward computation and
transports the **same** low-rank input derivative; its factors are not merged
into a rounded dense weight or exposed as trainable parameters. A single
admitted geometry is rebound on promotion and can then be moved between
attention, MLP and block outputs. Existing active LoRA/geometry composition
still requires colocation.

## What was checked

- Rust: identical full-window and KV-cache outputs before/after promotion;
  identical colocated geometry gradients; distinct parent, derived and
  changed-LoRA identities; stale gradient and checkpoint rejection.
- On a two-layer artificial WaveGate decoder, each of six sites had
  nonzero gate/bias derivatives. For each parameter block, its strongest
  coordinate's full-answer gradient agreed with independently differenced
  complete, stop-terminated answer probabilities. A separate dense oracle
  checked the upstream gradient through a strong fixed LoRA. Its maximum
  error was below 2e-6, while removing only the low-rank input VJP with
  **identical forward logits** exceeded 1e-4.
- Cached window training: a frozen prefix including the promoted LoRA was
  reused after checkpoint restore; receipts, observations and final
  checkpoints matched uncached training. A cache/restore with different
  fixed weights was refused. The original LoRA bytes belong with the
  parent base, not inside an owner-aware geometry checkpoint.
- On the existing generated width-three decoder with a nonzero LoRA,
  Python ran a three-step SGD reference for each of the six sites: one
  set, one menu and one subset objective. Real WASM in Node resumed the
  first cut for each site and performed step two. Python imported those
  six second cuts and performed step three. Each import reproduced its
  supplied checkpoint bytes and ID exactly. The largest observed WASM
  difference from native among compared values (loss, logits, parameters)
  was 2.56e-7; the returned native states and final step comparisons had
  zero maximum absolute difference. This is 18 reference updates and
  6+6 replay updates, not 30 independent training examples.

The complete `st-nn` library suite passed (1,351 tests). Installed CPU-wheel
decoder, LoRA and session tests passed (71 + 5 + 35); the WASM binding built
with `nn`, and its TypeScript API type-checked. This is an engineering
comparison on generated weights, **not** a ranking of insertion sites or
evidence of pretrained language quality. No GPT-2, held-out, GPU, or prior
stopped signed-probe run was repeated. No release artifact was published.

## Reproduce the client handoff

From the repository root, use an installed development CPU wheel, a generated
WASM web package and two unused output paths. No Torch, NumPy, model download
or network request is required:

```sh
python -IB bindings/st-wasm/tests/native_frozen_lora_placement.py issue "$NATIVE_CUTS"
node bindings/st-wasm/tests/native_frozen_lora_placement.mjs "$WASM_JS" "$NATIVE_CUTS" "$WASM_CUTS"
python -IB bindings/st-wasm/tests/native_frozen_lora_placement.py finish "$NATIVE_CUTS" "$WASM_CUTS"
```

Development evidence SHA-256 (not publication or release qualification):

| Artifact | SHA-256 |
| --- | --- |
| Native cut file | `1bfb0987b447cc7e101927a22de9b93cddd9f6a3e0e5b703a6fdcddb1064000c` |
| Returned WASM cuts | `d46df38fe7f76827d9b9ec55826068347fd730264270d3aa1772422ca936e9e8` |
| Local macOS CPU wheel | `d5efac61abf9f045e98a89815bc5b83e17620e6e26a556943713e2589b8e7b04` |
| WASM module | `fa9acba8ded081f7b9c680ff105a069c28bc49969057717d23da2d3332b28118` |

See the [API and reconstruction example](../../../../../bindings/st-wasm/tests/native_causal_lm.md#keep-lora-fixed-across-geometry-placements).

# Native Generation From Trained QKV Snapshots

Status: source implementation, 215 native CPU decoder tests, and CPU Python
binding compilation passed. Python runtime verification is pending: this stage
did not relink the extension or replace the saved wheel. It did not run a new
pretrained-model study, score historical evaluation inputs, or publish artifacts.

The previous [saved-model restore comparison](PYTHON-QKV-RESTORE.md) established
full-prefix/cache compatibility, not automatic sampling. This change connects
QKV learning to the existing native generator, including its Z-Space decoding
policy, rather than writing another sampler in Python.

## Ownership And API

- Rust `NativeCausalLmQkvAdamW::generator` captures the current serving snapshot.
- `generation_input_count` admits the full generation budget before foreign
  prompt extraction; `restore_generator` requires the same serving identity.
- Python exposes `generator(prompt_ids, config_json)` and
  `restore_generator(payload, expected_id)`, returning `NativeCausalLmGenerator`.
- The generator exposes optional `projection_id`. Its checkpoint binds this ID
  in addition to the existing base, adapters and readout policy. Ordinary-model
  checkpoints omit the new field, so their serialization is unchanged.
- Ordinary-model restore and QKV restore reject each other's checkpoints, even
  before any QKV update. Updated weights remain separate from generation JSON;
  restore the appropriate QKV checkpoint before restoring generation.

Sampling, RNG continuation, EOS and cache execution remain in the existing Rust
generator. A later optimizer update cannot modify a live generator's snapshot.
Serving identity deliberately excludes AdamW moments and data order, so it
proves inference ownership, not equality of complete training trajectories.
Expected hashes are integrity checks, not signatures or authenticity proofs.

## Executed Checks

One offline native test invocation completed in 29.92 seconds, including
compilation; the 215 tests themselves took 2.70 seconds. Two added test methods
exercise ordinary/key-lag attention, all three readout policies, greedy/sampled
selection, ordinary softmax/Z-Space distributions, and full-prefix/cache paths.
Models are the small synthetic width-4 fixtures with frozen LoRA and geometry,
not the saved GPT-2 or historical width-32 study endpoints.

Checks cover updated QKV logits, matching native generation reports, exact
sampled continuation after checkpoint restore, snapshot stability during later
optimizer updates, cross-owner and cross-route refusal, length/cache admission,
EOS stopping, and preservation of optimizer state during inference. Existing
ordinary generation tests also assert omission of the projection field.

One CPU binding `cargo check` completed in 22.84 seconds. It verifies the Rust
types and PyO3 binding compilation, **not Python execution**. Two new Python
tests cover snapshot continuation/owner rejection and admission before foreign
token access, but await a new extension. The prior 51-test Python result belongs
to the earlier cache API and is not reused as evidence for these new methods.

Both commands used the existing scoped Cargo target with incremental output
disabled, two build jobs, one test thread, a 180-second wall cap per command,
and a 4 GiB free-space floor. Both completed on the first invocation with no
guard trigger and unchanged source inventories. No files were deleted. Existing
dead-code warnings remain; this was not a full CI or WASM/GPU validation.

Local evidence: `2026-11-10-qkv-generation-v1` in the SpiralTorch validation store.

| Evidence | SHA-256 |
| --- | --- |
| Tested source inventory | `1ca3a35751916ed342f8cbee86fdfa0469c379d55aca49ae8584048e210c1e43` |
| Native test log | `d9351068c672c565e6cd8ba8b7a64fe367a71488e99707bdeba875ac69605109` |
| CPU binding check log | `3d683bf3d25c8ac23e29f1f9235ba1490e70c87c8bab4b6ea4eef998a6a101ad` |

The code tested by these receipts precedes only the documentation additions.
No decoder/source code was changed after these successful invocations.
See the [usage guide](../../docs/python/native-causal-lm.md#generation-after-qkv-updates).

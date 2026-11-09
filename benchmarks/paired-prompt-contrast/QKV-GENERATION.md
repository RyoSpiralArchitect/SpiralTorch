# Native Generation From Trained QKV Snapshots

Status: source implementation, 215 native CPU decoder tests, CPU binding
compilation, and 66 isolated Python runtime tests passed. The CPU extension was
subsequently built once into a private source-package overlay; the saved previous
extension and wheels were preserved. No new pretrained-model study, historical
evaluation scoring, installed wheel, or publication was performed.

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

One CPU binding `cargo check` completed in 22.84 seconds. That result verifies
the Rust types and PyO3 compilation, **not Python execution**. It was followed
by the separate runtime checks below. The prior 51-test Python result belongs
to the earlier cache API and is not reused as evidence for these new methods.

Both commands used the existing scoped Cargo target with incremental output
disabled, two build jobs, one test thread, a 180-second wall cap per command,
and a 4 GiB free-space floor. Both completed on the first invocation with no
guard trigger and unchanged source inventories. No files were deleted. Existing
dead-code warnings remain; this was not a full CI or WASM/GPU validation.

## Python Runtime Verification

One CPU release build completed in 118.23 seconds under a 600-second cap and
the same 4 GiB free-space floor. It used `--no-default-features --features
python-default`, reused the current Cargo target, and neither deleted caches
nor installed a wheel. The previous saved library's SHA-256 was checked before
and after the work and remained unchanged.

An isolated 83-file package overlay was pinned before import. Under
`python3.12 -I -B -S`, one fail-fast invocation passed all **66 tests**:
11 QKV/interface tests, 44 existing native decoder tests, and 11 existing
generation tests. No skips, import attempts for foreign ML packages, or loaded
Torch/NumPy/Transformers/Python-safetensors/Accelerate modules were recorded.
Package and source inventories were unchanged during the invocation. The outer
CPU runner took 2.04 seconds against a 120-second cap; test-body time was 0.227
seconds. These timings describe validation, not a performance benchmark.

The new runtime checks cover snapshot ownership and refusal after further
training, admission before foreign prompt access, and sampled Z-Space report
and checkpoint equality across ordinary/key-lag attention and all three readout
policies, with frozen LoRA and geometry. Existing generation regressions also
exercise EOS, numeric failures, token-only/chunked generation, ordinary sampling,
and the file-based public example's uninterrupted/resumed output agreement.
All models are small synthetic test fixtures; no pretrained model was loaded.

Inspection also found the new methods missing from the Python type declarations.
The declarations now include QKV `generator`/`restore_generator` and generator
`projection_id`. A runtime regression checks the declared method names,
argument names, return annotations and property declaration against the new
native interface. This is not a full static-type-checker audit.

Local evidence: `2026-11-10-qkv-generation-v1` in the SpiralTorch validation store.

| Evidence | SHA-256 |
| --- | --- |
| Initial Rust/check source inventory | `1ca3a35751916ed342f8cbee86fdfa0469c379d55aca49ae8584048e210c1e43` |
| Native test log | `d9351068c672c565e6cd8ba8b7a64fe367a71488e99707bdeba875ac69605109` |
| CPU binding check log | `3d683bf3d25c8ac23e29f1f9235ba1490e70c87c8bab4b6ea4eef998a6a101ad` |
| One CPU extension build log | `d2fa151b4d7ba2e03c9c29e178e590b505afd4535fa5eb7217dd54298f1ef92c` |
| New CPU extension | `98cb127768f5d18ab330fe60739f40ab57a068a6e11ef34852f6d481dfceeace` |
| Frozen package manifest | `c5b024e4d356bb9977a96a32a3f8377a473b6da00283aa955d9f66a7618ea92e` |
| Python runtime source inventory | `eb78abaf172975f04bf73d1c36dece9ba0e1092ea9f77d08ccd59f082f8f9218` |
| Python 66-test log | `f4a8efdd66e8c413f3930f751a40ad9b25a0c224a824414cc8d16fb6d6b63fcb` |

The native Rust implementation did not change after the initial Rust checks.
Python tests and type declarations were extended before the Python runtime
invocation; only documentation changed after that invocation.
See the [usage guide](../../docs/python/native-causal-lm.md#generation-after-qkv-updates).

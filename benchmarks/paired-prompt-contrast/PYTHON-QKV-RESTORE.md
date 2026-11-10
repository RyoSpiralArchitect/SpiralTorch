# Saved Rust QKV Models Through Python

Status: one bounded CPU comparison completed. All 6,144 previously scored
inputs reproduce the saved Rust logits **byte for byte** through both Python
full-prefix inference and Python cached branching. No model was updated and
no new accuracy, NLL, calibration or generalization score was computed.

This connects the [Python QKV cache API](PYTHON-QKV-CACHE.md) to actual saved
endpoints of the [native smoothing study](QKV-SMOOTHING-COMPARISON.md), rather
than only to the width-4 unit-test fixture. These remain small **synthetic**
language models, not pretrained GPT-2 or an HF-model quality evaluation.

## Scope And Result

All four models have two decoder blocks, width 32, four heads, vocabulary 64
and position capacity 16. Each endpoint contains 256 QKV AdamW updates from
the earlier experiment. The original base archive, frozen LoRA and geometry
artifacts, `f32_wide_backward` policy, and checkpoint were reconstructed using
only exported `spiraltorch` APIs. Rust validates artifact ownership and restores
the optimizer; Python does not deserialize optimizer tensors or implement math.

| Saved endpoint | Recorded inputs | Full-prefix / cache / original | Optimizer after inference |
| --- | ---: | --- | --- |
| Ordinary attention, CE | 1,536 | All bytes equal | Unchanged |
| Ordinary attention, smoothing .1 | 1,536 | All bytes equal | Unchanged |
| Key-lag 1, CE | 1,536 | All bytes equal | Unchanged |
| Key-lag 1, smoothing .1 | 1,536 | All bytes equal | Unchanged |

The frozen roster has 24 cells per model. Observed calls match the plan:
6,144 full-prefix forwards, 3,072 prefix prefills, and 12,288 single-token cache
branches. Cache parents and serving identities are checked for every row.
Each restored checkpoint initially roundtrips to the exact saved bytes and
remains identical after all of that model's inference calls.

The isolated runner verifies the exact native library and package file hashes,
blocks foreign ML imports, and records zero attempted imports and zero loaded
Torch/NumPy/Transformers/Python-safetensors/Accelerate modules. Inputs and
runtime/source files are rechecked after execution. Seven model-free tests
cover exact output streams, shared-prefix orchestration, first-mismatch stop,
input hash/size checks, wrong owners/counts, and import-attempt recording.

The comparison succeeded on its first model invocation, with an outer elapsed
time of 1.63 seconds against a 300-second cap. This is **not** a performance
benchmark or speedup claim. An earlier preparation attempt was refused before
launching any tests because free disk was below 4 GiB. That refusal is retained.
Only two subsequently approved, rechecked-unused incremental compiler caches
were deleted, recovering an observed 1,103,096 KiB. There was no rebuild.

## Independent File Check

The run saved eight raw f32 little-endian files (393,216 bytes each), plus
per-cell and complete receipts. A separate disk-only check, with no SpiralTorch
import or model calls, reconstructed expected bytes directly from the sealed
Rust output JSON. It matched all eight files, all 96 per-cell hashes and all
6,144 rows. This also checks the persisted output, not just the in-process
assertions. Model weights, checkpoint originals and prior evaluation records
were neither overwritten nor deleted.

Local evidence directory: `2026-11-10-python-qkv-restore-v1` in the SpiralTorch
validation store. It includes plans, source inventories, raw streams, receipts,
the original pre-launch refusal, cleanup scope and `audit_raw.py`.

| Evidence | SHA-256 |
| --- | --- |
| Fixed saved native library | `3497e4a24cd8df3fef1423c078dcf44c0bd5de45798f1a77bff7a5a5c27a2725` |
| Frozen comparison plan | `e87ec917c0201d0a90cd392fc6f0dc21ec8d8229e501f94657c5b267d6b49df9` |
| Original Rust scoring record | `e9067ee01c9ff6e15f8e502e2a802f7935741d2839717178d7864c5cf6e5cc38` |
| Complete Python comparison | `6eb61b9d9bf223b9c108c1f0a4bb0e53304cd22c2e6bceb7dde7684e23d668ef` |
| Independent raw-file audit | `1a5472405af1be350367308f31f07e5c688d9aae821aa9bca200f670548f7bde` |

## Reproduction Boundary

The committed `python_qkv_restore.py` deliberately pins this historical study,
not an arbitrary dataset. It needs the retained private artifacts, their original
manifest paths and the exact saved package/library; a public wheel or fresh
checkout alone is insufficient. Use a new output directory and preserve prior
results. The original run additionally used the recorded outer CPU/disk guard.

```sh
python3.12 -I -B -S benchmarks/paired-prompt-contrast/test_python_qkv_restore.py
python3.12 -I -B -S benchmarks/paired-prompt-contrast/python_qkv_restore.py freeze "$STUDY" "$PACKAGE" "$OUT/plan.json"
# Use the hash printed by freeze, under the same CPU/time/disk guard.
python3.12 -I -B -S benchmarks/paired-prompt-contrast/python_qkv_restore.py run "$OUT/plan.json" "$PLAN_SHA256"
```

This result establishes saved-artifact and inference-path compatibility for the
specified models and inputs. It does not change the earlier experiment's failed
competence gate, turn used inputs into fresh holdout, validate GPU/WASM execution,
or authorize publication. No additional learning, quality search, foreign-ML
fallback, model retry, build, GPU run, push, tag or release occurred here.

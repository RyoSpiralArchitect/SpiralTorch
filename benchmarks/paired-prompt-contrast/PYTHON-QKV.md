# Native QKV Through Python

Status: local CPU extension and source-package overlay validated, not a wheel
release or a new language-quality experiment. See the
[Python API example](../../docs/python/native-causal-lm.md#answer-qkv-learning).

`spiraltorch.NativeCausalLmQkvAdamW` exposes the existing Rust optimizer's
ragged answer gradients, atomic steps, optional label smoothing, pinned
checkpoint restore and updated inference logits. The native layer owns both
shape/work admission and learning semantics. Python does not rebuild the
objective, grouping, gradient mean or optimizer.

The core now shares length-only admission between binding transport and actual
execution, including per-group geometry budgets. The wrapper admits prefixes
before reading token values, rejects Boolean dimensions/IDs/hyperparameters,
and releases Python's interpreter lock during native computation. Only one
QKV weight matrix is trained; the base handle and attached adapters stay frozen.

## Verification

- 212 Rust decoder tests pass, including shared length admission, finite
  differences, legacy/smoothed compatibility and exact optimizer continuation.
- CPU binding check succeeds with default features disabled and
  `python-default,cpu`; the local release library builds once with default
  features disabled and `python-default`. Neither command enables WGPU.
- 49 Python tests pass against that exact library: 5 QKV/facade tests and
  44 existing native causal-LM tests. The fixture is synthetic, width 4,
  two layers and 11 vocabulary entries, with a frozen geometry attachment.
- The Python path performs real updates and bit-exact checkpoint continuation,
  preserves the input model, rejects invalid inputs without state changes,
  and produces changed logits from the updated trainer.
- The isolated `python3.12 -I -B -S` runner rejects any attempt to import
  Torch, NumPy, Transformers, Python safetensors or Accelerate. Final counts
  are zero attempted imports and zero loaded foreign ML modules.

The first Python invocation passed all 48 then-existing tests but failed the
stricter import-attempt gate: the eager elliptic interoperability helper tried
to import Torch and swallowed the rejection. Its original failure receipt is
retained. The facade now resolves those four existing elliptic helper names
only on explicit access, like the other optional adapters. An added regression
test checks name forwarding and caching. This is not removal or reimplementation
of the optional PyTorch autograd feature.

## Local Evidence

The validation directory is `2026-11-10-python-qkv-binding-v1` under the local
SpiralTorch validation store. Plans include source inventories and execution
budgets; receipts include source-unchanged checks. The installed package,
existing wheels, model weights and experiment originals were not overwritten.

| Evidence | SHA-256 |
| --- | --- |
| CPU extension | `dae4510771fa997b2e16b37f515195b571baa7d70aa3d6aa7d1c027109902af7` |
| Rust test log | `a179f92b7220c42854aae94bca47001d522e393014179e39ad5366bfabdfc9cc` |
| Release library build log | `b6d405c2f34ef54bca0beae169e5cc24178e89a516090c289f4f0a4e1817311c` |
| Initial import-gate failure log | `ee850709d2302e18aa0956e3050a04eb4cf58b0abbcba5fe1c859373a06f1146` |
| Final 49-test Python log | `2f3667954759ce89b24ba6493035b9c45554d7d1325f93da2b556ce0d2dd172f` |

The native library was not rebuilt after the Python-only lazy-import change.
No pretrained-model learning, GPU execution, WASM binding qualification, full
repository CI, package installation, push, tag or publication occurred here.
The earlier [smoothing comparison](QKV-SMOOTHING-COMPARISON.md) remains separate:
its relative improvement does not become general language competence merely
because the same Rust mechanism is now callable from Python.

For an extension-enabled environment, the direct regression entry point is:

```sh
python -B bindings/st-py/tests/test_native_causal_lm_qkv.py
```

That entry point records rejected optional import attempts rather than relying
only on the absence of imported modules. The local evidence used an additional
isolated runner that verified the loaded extension path and digest before
running both test modules.

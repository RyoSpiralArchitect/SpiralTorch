# Native GPT-2 Forward and Geometry Probe

This is a bounded offline **compatibility** probe on an existing local GPT-2
snapshot, not a training, model-quality or speed benchmark. No model was
downloaded and no provider API was called. Raw logits and both original reports
are retained locally; this directory publishes all six conditions, hashes and
reproduction code without redistributing model weights.

## Result

On Apple M4 / macOS 26.4.1, the complete Rust f32 CPU-prepacked decoder was
compared with Torch 2.12.1 / Transformers 4.57.6, eager attention, eval mode,
float32 and no KV cache. The reference used two Torch CPU threads. Native
execution forbade imports of Torch, Transformers, NumPy, Accelerate and Python
safetensors. Three prompts were each evaluated with and without a fixed Wave
adapter at `transformer.h.0.mlp`, followed by four greedy continuation tokens.

| Check over all six conditions | Observed |
| --- | --- |
| Prompt vocabulary logits compared | 1,608,224 |
| Maximum absolute logit difference | 0.000152587890625 |
| Maximum prompt NLL absolute difference | 0.0000286102294921875 |
| Greedy continuation ID agreement | 6 / 6, all four new tokens |
| Native Wave-vs-base maximum logit change | 0.0115204 to 0.0247345 |

These are measurements, not a tolerance selected after observing the result.
The independently specified small-model parity tests remain separate. The
Wave intervention changes logits; all three short greedy continuations happen
to remain unchanged. Tiny prompt-NLL changes are **not** evidence of better
language modeling or fine-tuning. No elapsed-time comparison was recorded.

Reference text, with native token IDs identical in both arms:

```text
The meaning of life is not the same as
Machine learning models can be used to predict
SpiralTorch is a free, open source
```

[results.json](results.json) includes each prompt, source IDs, full shapes,
native/reference NLL, both generated ID lists, per-arm max/RMS errors, and
paired intervention differences. It binds the exact source/config/tokenizer,
geometry artifact, raw logits and original reports by SHA-256. Hashes establish
byte identity, not model provenance or a quality certificate. Comparison
requires equal inputs across both runtimes **and** both intervention arms.

## Reproduce

Use Python 3.12 and a locally built source wheel exposing `NativeCausalLm`.
This extension is **after the fixed 0.4.32 candidate**, not an official PyPI
0.4.32 wheel. The measured development wheel SHA-256 is
`c5e547a5e8245b2939ae585a27a4653c9734fed05688bd70bff5e27ec5e579a0`.

The existing local `gpt2` snapshot revision was
`607a30d783dfa663caf39e06633721c8d4cfcd7e`; exact file hashes are in the result.
Use separate native and optional reference environments. `NATIVE_PY` must have
the source-built SpiralTorch package but no foreign ML runtime. `REFERENCE_PY`
needs the same SpiralTorch package plus the pinned comparison dependencies
above and `tokenizers`. `MODEL_DIR` is that existing local snapshot; no step
below downloads a model. Output directories must not already exist.

```sh
"$NATIVE_PY" -I -B benchmarks/results/2026-10-18-native-gpt2/probe.py \
  --mode native --model-dir "$MODEL_DIR" --output "$RUN_DIR/native" --steps 4
"$REFERENCE_PY" -I -B benchmarks/results/2026-10-18-native-gpt2/probe.py \
  --mode reference --model-dir "$MODEL_DIR" --output "$RUN_DIR/reference" \
  --native-report "$RUN_DIR/native/report.json" --steps 4
"$NATIVE_PY" -I -B benchmarks/results/2026-10-18-native-gpt2/compare.py \
  "$RUN_DIR/native/report.json" "$RUN_DIR/reference/report.json" \
  --output "$RUN_DIR/comparison.json"
```

`probe.py` pins the three prompt IDs and constructs the fixed Wave state; the
reference checks the IDs against the local tokenizer. It writes full logits
as declared-endian float32 plus reports. `compare.py` verifies shape, finite
values, hashes, all cases, source/geometry identity, and paired inputs before
computing discrepancies. Its five synthetic receipt tests need no model.

## Qualification and Boundary

- Rust: 8 model tests, 14 weight-loader tests, 19 native geometry tests passed.
- Installed isolated wheel: 5 model tests, 8 weight tests and 14 geometry tests
  passed, with optional framework imports forbidden.
- Optional reference suite: 121 tests passed, including 12 new complete-model
  comparisons covering F32/F16/BF16 source weights, architecture variants and
  the attention/MLP/block geometry output positions.
- Release/compatibility: 32 tests, five probe-comparison tests and three core
  smoke programs passed. Strict WGPU-enabled Rust documentation and WASM core
  compilation passed; this is not browser execution or non-macOS qualification.
- Independent source review found and fixed missing cross-arm input checks in
  the comparator. The reviewer did not execute or independently reproduce the
  numerical experiments.

The model is frozen, CPU-only and full-prefix. There is no padding, KV cache,
whole-model backward/FT or arbitrary HF architecture import. Existing selected
geometry-site VJP/SGD is not a language-loss gradient through this decoder.
See [API and supported policy](../../../docs/python/native-causal-lm.md).

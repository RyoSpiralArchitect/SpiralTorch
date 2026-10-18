# Native GPT-2 Language-Loss Geometry Learning

This offline probe checks **language-loss gradients and geometry updates**
through an existing pretrained GPT-2, with no Torch in the native path.
It is not a held-out quality, full-model fine-tuning or speed experiment.
No model download or provider call was made. All three steps and parameter
group comparisons are published in [results.json](results.json); full raw
gradients, adapter snapshots and original receipts remain local and hash-linked.
The deterministic public projection removes all 96 operand-norm fields, since
a scalar norm would disclose an individual learned value. Every case and
max/RMS discrepancy remains, alongside nonzero-gradient booleans and the hash
of the unchanged private comparison. This is a publication boundary, not a
formal privacy guarantee.

## Setup And Results

Apple M4 / macOS 26.4.1, Python 3.12.6. The Rust f32 CPU-prepacked decoder uses
the same frozen local GPT-2 checkpoint as the [forward probe](../2026-10-18-native-gpt2/README.md).
The optional reference uses Torch 2.12.1 / Transformers 4.57.6, two CPU threads,
eager attention, float32, eval mode, and no KV cache. Native execution forbids
Torch, Transformers, NumPy, Accelerate and Python safetensors imports.

Three non-nested placements are active together, each with strength 0.1:

- `transformer.h.0.mlp`: WaveGate with learned radius.
- `transformer.h.6.attn`: fractional history-angle/gain adapter.
- `transformer.h.11.mlp`: Topos resonator.

The 3,843 geometry parameters receive mean shifted next-token NLL gradients.
The frozen base receives no parameter updates. Three sequential SGD steps
use rate 0.02, one fixed short window per step. These windows are the previous
probe's three short continuations, not a new benchmark corpus or holdout set.

| Step / window | Native NLL before / after | Reference NLL before / after |
| --- | --- | --- |
| 0 / meaning | 2.811478615 / 2.811465263 | 2.811447144 / 2.811438084 |
| 1 / models | 3.289907217 / 3.289893150 | 3.289875746 / 3.289865732 |
| 2 / spiraltorch | 3.433504581 / 3.433483124 | 3.433475494 / 3.433457613 |

Across all steps, maximum absolute NLL difference is `3.147125244140625e-5`,
gradient difference `2.348935231566429e-5`, and post-update parameter
difference `1.1268100000000049e-6`. Every parameter group has a nonzero gradient
norm. Native artifact restoration reproduces the next loss, all gradients,
and the next updated artifact **exactly**. The original native base probe
logits remain byte-identical; reference base parameter versions remain
unchanged and have no gradients.

These are measured differences, not pass thresholds chosen from this result.
The separate synthetic parity tests have their own assertions. The tiny
within-window loss changes do not establish a generalization benefit, and
different windows' NLL values must not be read as a learning curve. No timing
comparison or preferred geometry was selected.

## Reproduce

This extension follows the fixed 0.4.32 release candidate. The locally built
development wheel is **not** an official PyPI 0.4.32 artifact. Its SHA-256 is
`628079e576fb79a7508a50a9574908e8f5d101f81fa4ec464e6ea2b9c886e626`.

Use the same source-built SpiralTorch wheel in separate native and optional
reference environments. `NATIVE_PY` has no foreign ML runtime; `REFERENCE_PY`
has the versions above plus `tokenizers`. `MODEL_DIR` is an existing local
GPT-2 snapshot, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` in this run.
Exact config, checkpoint and tokenizer hashes are recorded in the result.
All output directories and comparison files must be new.

```sh
"$NATIVE_PY" -I -B benchmarks/results/2026-10-18-native-gpt2-learning/probe.py \
  --mode native --model-dir "$MODEL_DIR" --output "$RUN_DIR/native"
"$REFERENCE_PY" -I -B benchmarks/results/2026-10-18-native-gpt2-learning/probe.py \
  --mode reference --model-dir "$MODEL_DIR" --output "$RUN_DIR/reference" \
  --native-report "$RUN_DIR/native/report.json"
"$NATIVE_PY" -I -B benchmarks/results/2026-10-18-native-gpt2-learning/compare.py \
  "$RUN_DIR/native/report.json" "$RUN_DIR/reference/report.json" \
  --output "$RUN_DIR/comparison.json" --public-output "$RUN_DIR/public.json"
```

The reference checks exact token IDs with the local tokenizer and starts
from the native artifact's exact bytes. Do not re-indent or re-serialize an
artifact before verifying its byte-bound identity. The comparator checks
inputs, all three steps, source identities, receipt hashes, parameter coverage,
finite values and fixed parameters/buffers before computing discrepancies.
Hashes bind bytes, not model provenance or quality.

## Qualification And Review

- Rust: 14 decoder/learning, 14 weight-loader and 20 geometry-runtime tests;
  three packed-transpose tests and one input-only LayerNorm VJP test passed.
- Installed isolated wheel: seven model/learning, eight weight and 14 geometry
  tests passed with foreign ML imports forbidden.
- Optional reference suite: 164 tests passed, including 43 language-loss
  gradient/update cases across all 14 native recipe configurations, checkpoint
  dtypes, supported decoder variants, and a simultaneous three-site case.
- Release/compatibility: 32 tests, six learning-probe tests, five earlier
  forward-probe tests and three core
  smoke programs passed. Strict WGPU-enabled Rust documentation and WASM core
  compilation passed. This does not qualify browser model execution or GPU
  execution of this decoder. Clippy completed with existing warnings.

Independent read-only source review found two P2 issues: differentiation
continued past the earliest geometry site into an unused frozen prefix, and
concurrent first calls could construct duplicate packed transposes. An overflow
regression failed before the first fix and passes afterward. Reverse execution
now stops at the earliest participating site, including frozen sites whose
gradients are exposed. Transpose construction is inside synchronized one-time
initialization. Follow-up source review closed both findings and found no new
actionable pilot/comparator issues; it did not execute the numerical checks.
The rebuilt-wheel tests and both actual model probes were rerun by the main
agent, with identical private comparison bytes to the earlier measured build.
Final publication review caught the scalar operand-norm disclosure described
above; the projection removes it without re-running or altering measurements.

This is geometry-only learning through a supported frozen GPT-2. It is not
arbitrary HF import, base-weight FT, KV-cache decoding, padding/masking support,
or a resident-GPU runtime. See the [native API](../../../docs/python/native-causal-lm.md).

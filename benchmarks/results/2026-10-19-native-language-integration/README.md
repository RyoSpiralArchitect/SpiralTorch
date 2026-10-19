# Native Language Stack Integration

Local qualification on 2026-10-19 JST, Apple M4 / Python 3.12.6. This joins
`113e8e99ff90c98e96eec16a3cd0b49e53c8ac44` (corpus evaluation/clipped bindings)
and `08c4096df69bbe689050bd488f310df0da2da2ab` (packed projections, readout
policy, repetition learning and native Linear/LoRA layouts). It is not a release
record or proof that the earlier macOS CI contextual-study mismatch is fixed.

Rust remains the execution owner. Python evaluation now exposes the actual
readout policy and remains plain unsmoothed NLL during repetition-aware training.
The combined packaged-example test covers periodic evaluation and exact resume
both with and without a positive repetition objective. The installed-wheel gates
cover this on normal CI, manual/official wheels and their Linux Python 3.8 floor;
those remote configurations are not a claim of local all-platform execution.

## Local Gates

| Gate | Outcome |
| --- | --- |
| st-nn no-default-feature library tests | 1,013 passed |
| st-tensor no-default-feature library tests | 472 passed |
| Linear/LoRA layout integration tests | 8 passed |
| Installed native wheel, 13 suites without foreign ML packages | 98 passed |
| Independent optional Torch-reference suite | 627 passed |
| Installed GPU byte decoder/study suites | 32 passed; 2 CPU-only tests skipped |
| Release workflow regressions / Python compatibility checks | 22 / 3 passed |
| st-nn wasm32 check / CPU-only nn binding check | Passed |

Torch, NumPy, Transformers, PEFT, Accelerate and Python safetensors were absent
from the native environment. The reference tests used a separate environment.
Independent read-only integration review found no actionable regressions; its
two coverage gaps were closed and re-reviewed. Review itself did not run tests.

The development wheel was built from index tree
`6edb77b229e4dfc8ff99ab12fcc0a20af447e145`; subsequent changes through
`f60891677235c1d462fb6a2cbc1bf063649d5ead` were tests/workflow coverage only.
The latter tree ran the installed-wheel, reference and real-checkpoint checks.
Final changes after those trees only publish this evidence and guide links.
The local wheel retains the development filename version 0.4.32 and **must not
be uploaded**. Its SHA-256 is
`8c9470b7d94a450ca0009432b99ac261693984faf84ea58902f81f1bbed00c81`;
the measured native library is
`623ca28b2112ae6222bb4da2f56e1b15d46ac615acc42d822dd150407a46e9ea`.

## Saved GPT-2 Checkpoint

All conditions are in [comparison.json](comparison.json), SHA-256
`3769449c6cc23a6280a2faa7c2e972022bf5cb67a53985023550b3cc755f8123`.
It preserves the private result's values and number spellings, adding a trailing
newline; the original result hash is
`cf378aac0f3ea8a4fefded8d08aa8c70e67c15cdaba112f61e58ed4a863731af`.
The original probe and pre-run contract hashes are, respectively,
`ba26fff35a468eec1b57febd5e902df80b6206e26c70a9029251766b070cb1d2` and
`f60ee01a622afde20d8aee566041f42b2f16e6e54854a390da7954a5b48530eb`.

This is the **previously trained-on** chapter: one document, 1,303 IDs, 1,302
scored targets, context 32 with stride 31 and retained tails. The trained model
loads the previously saved 20-update geometry checkpoint; this run performs no
new learning. It is not a held-out result, quality improvement or speed benchmark.
Local compilation/GPU tests overlapped this diagnostic, so timings are not used.

The unchanged absolute NLL gate is `5e-4` against the retained, pinned FP32 Torch
reference on identical inputs. That reference was not rerun for this experiment;
its hash and all input hashes are in the result. Reference NLL is
`4.685049704509214` for base and `4.679868947159493` for trained.

| Model | Readout | Batch | Native NLL | Absolute error | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| base | f32 | 1 | 4.68507562932514 | 2.5924815926e-5 | pass |
| base | f32 | 2 | 4.68507566906157 | 2.5964552356e-5 | pass |
| base | f64 | 1 | 4.685075595265343 | 2.5890756129e-5 | pass |
| base | f64 | 2 | 4.685075544175648 | 2.5839666434e-5 | pass |
| trained | f32 | 1 | 4.679894611949012 | 2.5664789519e-5 | pass |
| trained | f32 | 2 | 4.679894583565848 | 2.5636406355e-5 | pass |
| trained | f64 | 1 | 4.679894782247997 | 2.5835088504e-5 | pass |
| trained | f64 | 2 | 4.679894753864834 | 2.5806705341e-5 | pass |

All eight generation combinations (base/trained x f32/f64 x full-prefix/KV-cache)
also passed exact step-report and checkpoint equality after saving at generated
token 1 and continuing for 3 more tokens. The prompt is the first 8 document IDs.
The JSON preserves every arm's checkpoint/report hashes and the full Z-Space
sampling recipe: seed 714, four tokens, top-k 40, top-p 0.9, temperature 0.8,
curvature -1, repetition window 32, repression 0.6, last-token repression 0.3,
and three-gram repression 0.4. KV capacity is 1,000,000 values. Equality is
within each fixed policy, not between arithmetic policies or execution modes.

## Replay

Build/install this source in a fresh environment without foreign ML packages.
Use the existing source tests; no provider credentials or network calls are needed:

```sh
cargo test --release --locked --offline -p st-nn --no-default-features --lib
cargo test --release --locked --offline -p st-tensor --no-default-features --lib
cargo test --release --locked --offline -p st-nn --no-default-features --test learning_parameter_layout
python -I -B bindings/st-py/tests/test_native_causal_lm_evaluation.py -v
python -I -B bindings/st-py/tests/test_native_causal_lm_repetition.py -v
python -I -B bindings/st-py/tests/test_native_causal_lm_readout.py -v
python -I -B bindings/st-py/tests/test_nn_native_learning.py -v
```

For the real-checkpoint replay, supply local `config.json`, `model.safetensors`,
the encoded document JSON and saved training checkpoint matching **all** hashes
in `comparison.json`. Raw inputs/weights/logs remain private, so the real-input
result cannot be independently reproduced from this repository alone. Load with
`SafeTensorArchive` / `NativeCausalLm.from_gpt2`, then
`NativeCausalLm.from_training_checkpoint(base, payload, "sha256:" + checkpoint_sha256)`.
For both models, select `with_readout_accumulator("f32"/"f64")` and call
`evaluate_documents(documents, 32, microbatch_windows=1/2)`. Require 1,302 targets
and the same fixed error gate for every arm, not just the best score.

Construct `NativeCausalLmGenerator` with the recorded generation config plus each
explicit `execution` object. After one `step()`, retain `to_json()` and
`checkpoint_id`; reconstruct with `NativeCausalLmGenerator.from_json`. Compare
each remaining step's JSON and both final checkpoints exactly. Do not infer
lower repetition or language quality from successful replay.

The original CI failure remains recorded on
[PR #2284](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2284).
Neither this local success nor the earlier instrumented cadence probe resolves
its cause. Parent-first reviewed CI integration and official release qualification
are still prerequisites for a new tag or PyPI upload.

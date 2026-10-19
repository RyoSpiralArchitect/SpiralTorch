# Native GPT-2 Microbatch Accumulation

Offline mechanism check on 2026-10-18, using the existing pinned GPT-2 snapshot.
This source-only extension is **not in the immutable PyPI 0.4.32 wheels**.
The supported entry points are described in
[Native Geometry AdamW](../../../docs/python/native-geometry-adamw.md#token-weighted-microbatches).

Three short, unpadded windows have 8, 7 and 10 next-token targets. Each group
reduces all 25 targets before one clipped AdamW step; the same group is replayed
for three updates. Wave, fractional angle/gain history and Topos placements
jointly train 3,843 geometry parameters through the frozen GPT-2 decoder.
Torch, Transformers, NumPy, Accelerate and Python safetensors imports are
forbidden in the native process. No model download or provider call is made.

| Update | Microbatches | Targets | Pre-update mean NLL | Post-update mean NLL | Optimizer clock |
| --- | --- | --- | --- | --- | --- |
| 1 | 3 | 25 | 3.194246082 | 3.193634758 | 1 |
| 2 | 3 | 25 | 3.193634758 | 3.193000584 | 2 |
| 3 | 3 | 25 | 3.193000584 | 3.192367582 | 3 |

The loss reduction agrees with target-weighted individual losses. The next
group's loss, every gradient, updated artifact and optimizer state exactly
match after restoring the final artifact/optimizer pair on this CPU runtime.
The frozen-base logit probe remains byte-identical. Pending accumulations are
not checkpointed; restoration starts at an optimizer boundary.

`results.json` contains **all** per-window/group outcomes, exact token IDs,
policy, source/checkpoint hashes and continuation flags. Raw geometry values,
gradients and moments are retained privately, not committed. This probe does
not compare real-model Torch AdamW, establish held-out quality, test long FT,
or measure speed/RSS. Separate optional tests compare 14 geometry recipes
against Torch/HF with ragged windows, SGD and AdamW (28 cases, three updates
each); those small-model parity results are not a real-GPT-2 quality comparison.

## Reproduce

Use an isolated installation built from this source, not the released wheel.
Provide a local GPT-2 snapshot containing the exact `config.json`,
`model.safetensors` and `tokenizer.json` hashes in the results. Token IDs and
initial geometry come from the existing
[three-window fixture](../2026-10-18-native-gpt2-learning/probe.py); all windows
are unpadded and positions restart at zero.

```sh
python -I -B benchmarks/results/2026-10-18-native-gpt2-accumulation/probe.py \
  --model-dir /absolute/path/to/gpt2/snapshot \
  --output /absolute/path/to/new/private/output
```

The output directory must not already exist. A mismatch in loss accounting,
per-group clocks, exact next-group continuation or frozen-base output fails
the run; no retry, threshold selection or discarded case is hidden here.

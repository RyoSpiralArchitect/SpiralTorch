# Native Selected-Target GPT-2 Learning

Offline mechanism check on 2026-10-18, using the existing pinned GPT-2 snapshot.
This source-only extension is **not included in released PyPI 0.4.32**.
See [native causal objectives](../../../docs/python/native-causal-lm.md#selected-targets-and-label-smoothing).

Three unpadded windows have 25 possible shifted targets. Explicit `-100` labels
select suffixes of 4, 2 and 4 tokens, for **10 supervised targets per group**.
Earlier tokens remain full attention context. Uniform label smoothing is 0.1,
so the reported value is **smoothed cross entropy**, not plain NLL/perplexity.
Three groups update 3,843 Wave, fractional-history and Topos parameters through
the frozen GPT-2 decoder, with one clipped AdamW update per group.

| Update | Pre-update objective | Post-update objective | Optimizer clock |
| --- | --- | --- | --- |
| 1 | 3.363684130 | 3.363010502 | 1 |
| 2 | 3.363010502 | 3.362315559 | 2 |
| 3 | 3.362315559 | 3.361633015 | 3 |

Torch, Transformers, NumPy, Accelerate and Python safetensors imports are
forbidden in the native process. Selected-target accounting agrees with
individual window losses. After saving/restoring the model/optimizer pair
and explicitly replaying the same objective, labels and inputs, the next
group's loss, every gradient, updated geometry and optimizer state are exact
on this CPU runtime. The frozen-base logit probe is byte-identical.

`results.json` preserves all group/window outcomes, labels, IDs, policy and
source/gradient/checkpoint hashes. Raw learned values, gradients and moments
are local-only. The selected suffixes are a deliberately bounded mechanism
fixture, **not an instruction-tuning corpus or held-out quality benchmark**.
No new matched real-GPT-2 Torch arm, RSS/speed measurement, padding support,
full-model FT or browser/GPU execution is claimed.

Separate optional small-model comparisons cover all 14 geometry recipes,
smoothing 0/0.2/1 and SGD/clipped AdamW over three updates (84 new cases).
The current validation passed 53 relevant Rust tests, 44 isolated no-framework
Python tests, 333 optional comparison/regression cases and 32 release/compatibility
contracts. WASM core compilation and binding Clippy passed with existing
warnings. Independent read-only source review found no actionable correctness
issue; it was not a separate executable test run.

## Reproduce

Install a development wheel built from this source in an isolated environment.
Use a local GPT-2 snapshot matching the hashes recorded in `results.json`.
No download or provider call is performed. Exact initial geometry and token IDs
come from the retained [three-window fixture](../2026-10-18-native-gpt2-learning/probe.py).

```sh
python -I -B benchmarks/results/2026-10-18-native-gpt2-objective/probe.py \
  --model-dir /absolute/path/to/gpt2/snapshot \
  --output /absolute/path/to/new/private/output
```

The output directory must be new. Incorrect target weighting, optimizer clocks,
next-group continuation or frozen-base output fails the run. Pending
accumulations are not serialized; checkpoint at completed update boundaries
and retain the objective/label policy and data cursor separately.

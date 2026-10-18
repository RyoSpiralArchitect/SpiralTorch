# Native GPT-2 Geometry AdamW: Bounded Execution Probe

On 2026-10-18, the local GPT-2 checkpoint completed three native language-loss
updates without Torch, Transformers, NumPy, Accelerate or Python safetensors.
The frozen decoder carried the same 3,843-parameter Wave-radius, fractional
angle/gain-history and Topos placements used by the earlier
[SGD probe](../2026-10-18-native-gpt2-learning/README.md).

| Step / window | Pre-update NLL | Post-update NLL |
| --- | ---: | ---: |
| 0 / meaning | 2.811478615 | 2.810555458 |
| 1 / models | 3.289887905 | 3.289332390 |
| 2 / spiraltorch | 3.433616877 | 3.433090687 |

Policy was fixed before execution: AdamW defaults, learning rate 0.001,
weight decay 0.01, shared global-norm clip 0.1. Updates run consecutively,
not as independent resets. Exact input IDs, source hashes, all three results
and geometry/optimizer checkpoint hashes are in [results.json](results.json).
Raw learned geometry and moment files stay in local validation storage.

Saving and restoring both owners reproduced the next gradients, geometry
update and optimizer state byte-for-byte. A frozen-base last-logit probe was
unchanged. This establishes bounded execution and continuation, not held-out
quality, a preferred optimizer setting, convergence, full-model FT or speed.
This real-model run has no new matched Torch arm; the separate optimizer test
suite compares 56 recipe/configuration cases over eight updates each.

## Reproduce

Use the reviewed source extension, **not** the published 0.4.32 wheel, which
does not include this API. The local tested development wheel's SHA-256 was
`791f931684944c70e7256cad5d45ac49c50fa42bf3f8af40bc26f4d441aec51c`.

```sh
python -I -B benchmarks/results/2026-10-18-native-gpt2-adamw/probe.py \
  --model-dir /path/to/local/gpt2/snapshot \
  --output /path/to/new/private/run-directory
```

The script refuses an existing output directory, performs no download, and
installs a foreign-ML import guard before importing SpiralTorch. The report
pins this script, the reused immutable fixture, model config, weights and
tokenizer bytes. It reuses declared token IDs; it does not perform tokenization.

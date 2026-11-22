# Six equal steps do not mean equal changes in predictions

The preceding [owner comparison](LINE_OWNERS.md) held the SGD rate constant.
It did not match the distance travelled. This comparison keeps the **same**
artificial decoder, WaveGate, nonzero LoRA A/B, three training examples, tile
size (11), and order (two epochs). There are six arms: LoRA, geometry, or both
at ordinary SGD rate 0.125, and the same three at a prescribed coordinate
distance of 1/256 per update. No settings were chosen from the outcome.

## Native CPU results

Path is the sum of the six realized Euclidean distances of the f32 weights;
endpoint is the straight-line distance from the common origin. Logit RMS
compares final vs initial full-vocabulary last-position logits on each of the
three training prompts, subtracting each prompt's mean change before taking
the RMS over all 33 values. Each loss column scores one different training
example after all updates. These are not held-out results.

| Policy / owner | Path | Endpoint | Centered logit RMS | Set NLL | Menu loss | Subset NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial | 0 | 0 | 0 | 4.568401 | 2.570731 | 0.114927 |
| SGD / LoRA | 0.238360 | 0.179709 | 0.038248 | 4.573523 | 2.561706 | 0.111868 |
| SGD / geometry | 0.113862 | 0.113625 | 0.016177 | 4.568263 | 2.546018 | 0.114526 |
| SGD / joint | 0.266701 | 0.213527 | 0.026197 | 4.573349 | 2.556907 | 0.113486 |
| Fixed L2 / LoRA | 0.023437 | 0.013991 | 0.023164 | 4.568306 | 2.553635 | 0.113209 |
| Fixed L2 / geometry | 0.023438 | 0.016155 | 0.006979 | 4.568369 | 2.558061 | 0.114692 |
| Fixed L2 / joint | 0.023438 | 0.014286 | 0.021738 | 4.568271 | 2.551859 | 0.113387 |

At a shared per-step distance, joint has the lowest set and menu losses, while
LoRA has the lowest subset loss. The menu ordering differs from ordinary SGD.
This does **not** isolate a universal owner advantage: the absolute path budget
also differs across policies, and every direction is recomputed on that arm's
own evolving state. Within the fixed-distance arms, the LoRA-only logit RMS is
about 3.32 times the geometry-only RMS despite equal path lengths. An intrinsic
distance, KL budget or output-space trust region is not being matched. LoRA's
factor coordinates depend on the choice of scale for A and B.

The [matched next-token KL comparison](LINE_ANCHOR_KL.md) tests the prediction-space
alternative on these same training prompts and reports individual-anchor variation.

## What the optimizer does

For selected gradients `g`, Rust constructs the concatenated f64 unit direction
`-g/||g||` and stores each coordinate as `f32(origin + distance * unit)`.
The full vector gets **one** budget, not a separate budget per tensor or
owner. Frozen tensors, buffers and the base cannot move. In this joint run,
LoRA used 74.7% to 99.94% of the squared step length depending on the example.
Receipts include both groups' norms, IDs, realized displacement and
`gradient_dot_step`, a first-order prediction, not a measured improvement.

A positive request with zero selected gradient, a nonfinite candidate, or
relative vector/length error above 1e-4 is rejected atomically. A zero target
preserves both owners exactly; a successful zero step still consumes one
scheduled example. The largest native relative vector error across the 18
fixed-distance updates here was 3.83e-5. Small subgroups may round to zero
without invalidating an otherwise representable *whole* vector.

The [machine-readable results](line_owner_distance_results.json) include all
36 pre-update losses, three final scores per arm, movement and cross-client
error. The [stdlib-only reduction](line_owner_distance_summary.py) checks the
native, Node and rendered Chrome readouts without re-running a decoder.

The [fixture](../../../../../bindings/st-wasm/tests/native_line_ray_fixture.py)
checks all six trajectories against explicit steps, every fixed-distance
coordinate against scalar math with actual f32 storage, and all ordinary SGD
losses and endpoints against the earlier frozen reference. To regenerate the
measurements use a development CPU wheel and two **new** output paths:

```sh
python -IB bindings/st-wasm/tests/native_line_ray_fixture.py run result.json bridge.json
python -IB bindings/st-wasm/tests/native_line_ray_fixture.py resume bridge.json
```

The [browser/Node demo](../../../../../bindings/st-wasm/tests/native_causal_lm.md)
also computes path, endpoint and centered output drift from its own weights.
It continues the native update-2 checkpoints for all three distance arms;
`resume bridge.json returned.json` completes the returned update-4 states in
fresh CPython. The cross-engine numerical tolerance is 3e-6 absolute; it is
not a promise of bit-identical training on arbitrary hardware. No pretrained
weights, GPT-2 evaluation prompts, Torch, NumPy, GPU or model download.

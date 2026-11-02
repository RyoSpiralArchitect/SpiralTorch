# PR 2301: first macOS Metal failure

[CI run 41431026840, attempt 1](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/41431026840/job/125229340767)
failed the exact detached-call comparison, not a checkpoint file write or a
CPU GPT-2 fit. Nine other jobs passed. Do not rerun for green, relax the
comparison, or treat the unqualified CI wheel as a published artifact.

Round 0 compared one serial result with two concurrent calls on the same
recipe. Worker 1 matched byte for byte. Worker 0 differed in **seven final
parameter scalars** of the geometry case, across token/position embeddings,
geometry projection bias, QKV bias, and block output bias. No other JSON
fields differed, including training and validation metrics. The largest
absolute difference was `3.725290298461914e-9`; the largest f32 distance was
four ULPs. These magnitudes describe the failure; they are not a tolerance.

[The compact inspection](first_failure.json) binds the original wheel, its
installation receipt and the runtime's native-library digest. The provider's
PR head and the synthetic merge used for the build are distinct and recorded.
The request hash also matches the retained historical failure from run
40252323265, but neither the cause nor the frequency follows from these two
failures. No first-update boundary was captured by this particular job.

One coordinate gives a specific next check. The affected token-embedding
scalar is row 2, column 0. That token contributes **once in update 1 and zero
times in update 2** (input positions only, not shifted targets). The embedding
pullback has one writer per output scalar and a fixed slot order: an unordered
sum of multiple contributions *at this coordinate* cannot explain the difference.
The input adjoint is already the sum of the block and geometry branches.
Compare its bits, the token-2 parameter gradient, and the post-SGD value at
update 1 before assigning blame to either branch, the update, or readback.
These intermediate values were not present in the retained reports.

To verify privately retained evidence without importing the package or
executing GPU code, place the four unmodified ZIPs and the Actions artifact
listing (`artifacts.json`) together. `run-identity.json` contains the provider
run's `id, run_attempt, conclusion, head_sha, html_url`; `merge-identity.json`
contains the Git commit's `sha, tree` (SHA) and ordered `parents` (SHA list).

```sh
python3.12 -I -S -B inspect_retained.py "$RETAINED" | cmp - first_failure.json
```

## Observe the missing boundary in a future run

The original wheel does **not** have this probe. In a development build,
selection is CPU-only and does not import Torch or NumPy:

```python
from spiraltorch.nn import ByteCorpusStudy

study = ByteCorpusStudy.from_json(request_bytes)  # exact bytes, not re-encoded JSON
selection = study.embedding_update_selection_json("geometry", 1, 2, 0)
```

For the pinned request, the selected input position is batch row 0, time 1.
The Rust equivalent is `study.embedding_update_probe("geometry", 1, 2, 0)?`.
The probe only supports uninterrupted, unclipped stateless SGD and a byte
occurring exactly once in the selected input batch. It does not substitute a
different optimizer or attempt to apportion a repeated token's gradient.

On an explicitly instrumented run, `study.run_wgpu_with_embedding_probe(...)`
retains four immutable scalar views of the **actual** update: the table weight
used by the forward, its summed embedding-output gradient at the input slot,
the token-table parameter gradient, and the submitted post-update weight.
There is no diagnostic copy, map, or replay between updates. The views do
retain their larger backing allocations, so scheduling/resource reuse can
still differ from the unobserved run. Save `observation.run_outcome_json`
before calling the one-shot `observation.read_bits_json()`, which copies four
f32 words and their validation flags. Failed runs and failed reads are not
successful measurements.

Given independently saved `run-outcome.json` and `bits.json` in each of
three directories, compare the four raw words and the final report weight:

```sh
python3.12 -I -S -B compare_embedding_bits.py SERIAL WORKER0 WORKER1
```

This compares one coordinate, not the whole report. A first differing word
locates an observed boundary, not a faulty shader; equal words do not clear
other coordinates or the original byte-for-byte gate. Bind wheel, adapter
and run provenance separately. No measurements from a new GPU run are
included here.

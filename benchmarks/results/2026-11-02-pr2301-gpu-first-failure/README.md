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

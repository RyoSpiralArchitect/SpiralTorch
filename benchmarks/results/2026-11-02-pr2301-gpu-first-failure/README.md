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
three directories, compare both the **complete instrumented report bytes**
and the four raw words plus the final report weight:

```sh
python3.12 -I -S -B compare_embedding_bits.py SERIAL WORKER0 WORKER1
```

Use `--reports-only` when a bits read fails. Whole-report comparison uses the
original UTF-8 JSON value inside the saved Rust envelope, not Python's
reserialization. A first differing word locates an observed boundary, not a
faulty shader. Equal coordinates do not clear other parameters; even identical
instrumented reports do not clear the original, uninstrumented CI gate.

### Fixed local scope

`run_embedding_trio.py WHEEL FAILURE_ZIP` defaults to CPU preflight in the
pinned wheel's venv (`python -I -B`). Only `--execute-once NEW_PRIVATE_DIR`
starts GPU work. One hardware Metal serial baseline followed by **one** pair of
simultaneously released host calls on the same immutable study; 600 seconds for
the owned process, no retries/rebuilds or fallback. The reference is consumed
before the pair; both worker calls must return before any worker views are read.
Host-call overlap is checked, not simultaneous GPU dispatch. Save individual
outcomes before mapping and the whole-report comparison before worker maps.
Stop on error or non-Metal reporting. Exit 0 means no observed difference in
this trio, 2 means a difference, 124 means timeout; none is qualification.
Existing attempt directories are never reused.

### Local result (not a CI rerun)

[The saved local comparison](local_metal_trio.json) records one instrumented
trio on Apple M4 Metal, using the pinned development wheel. Both workers'
complete report bytes matched the serial result; their host calls overlapped
for 155.9 ms. The four observed f32 words also matched across all three:

| Weight before | Embedding-output gradient | Token-table gradient | Weight after |
| --- | --- | --- | --- |
| `3bcd8de5` | `b5ab005e` | `b5ab005e` | `3bcd8e6e` |

The local final report kept that post-update word. With these measured local
inputs, both a rounded f32 product and an exact product land inside the same
output rounding cell. The original hosted serial and worker 1 ended at
`3bcd8e6e`; worker 0 ended four ULP away at `3bcd8e6a`. Its first-update
gradient is still unknown. Across the different adapters, wheels and observation
modes, even the two serial reports differ at 226 final-parameter fields and
nine other fields, including adapter metadata. This comparison cannot assign
those differences to hardware, establish determinism, or qualify the red CI run.

To reproduce the compact result **without** running the model:

```sh
python3.12 -I -S -B summarize_local_trio.py "$PRIVATE_TRIO" "$ORIGINAL_FAILURE_ZIP" | cmp - local_metal_trio.json
```

### Hosted comparison boundary

The historical job used the macOS 26 arm64 image `20261026.0455.2`, system
26.6.2 build `25G83`, Python 3.12.10 and an Apple Paravirtual Metal adapter.
The public fixture reproduces all 47,959 request bytes, so only the encrypted
development wheel needs transporting. CPU preflight:
`python -I -B run_embedding_trio.py WHEEL --hosted-fixture`.

The dispatch template is inert outside the dedicated branch. It admits only
the reviewed SHA, one first-attempt run, the pinned host and interpreter; the
serial result must identify the expected adapter before the concurrent pair.
No wheel is rebuilt, no diagnostic runs during ordinary CI, and only encrypted
results leave the job. The original CI invoked three prior study tests in the
same interpreter; this fresh instrumented trio intentionally does not. Even
on the same runner image, it cannot certify the original uninstrumented test.

### Hosted result (original gate unchanged)

[Dispatch 41470149544, attempt 1](https://github.com/RyoSpiralArchitect/SpiralTorch/actions/runs/41470149544)
admitted the exact image, OS build and Python above. All three instrumented
reports matched one another **and the original uninstrumented serial report
byte for byte**. They also matched the original worker 1. Comparing against the
original worker 0 recovered the same seven parameter differences and no others.
Worker host-call overlap was 13.70 s; this is not proof of GPU-level overlap.
There were no retries or rebuilds. The temporary input secret was retired;
the authenticated result and its decryption identity remain private.

| Source | Input adjoint | Token-table gradient | Weight before | Weight after update 1 | Final report |
| --- | --- | --- | --- | --- | --- |
| Hosted trio (all three) | `b5ab005d` | `b5ab005d` | `3bcd8de5` | `3bcd8e6e` | `3bcd8e6e` |
| Local M4 trio (all three) | `b5ab005e` | `b5ab005e` | `3bcd8de5` | `3bcd8e6e` | `3bcd8e6e` |
| Original hosted worker 0 | not observed | not observed | not observed | not observed | `3bcd8e6a` |

**Conditional bound, not a historical measurement.** If the archived initial
weight was used, the absent token in step 2 made no change, and the update
and readback were correct, what step-1 gradient could produce the outlier
word? For this unclipped single-input SGD coordinate, inverting the exact
round-to-nearest-even output cell requires at least a **2.413%** change from
the measured hosted gradient (270,429 representable f32 steps). This holds
whether the product rounds before subtraction or only the final result rounds.
The one-ULP cross-adapter gradient difference does not suffice. This bound
does not tell us which premise held for the missing historical intermediate.

[Compact hosted evidence](hosted_metal_trio.json) binds the encrypted artifact,
environment, full-report hashes and both inverse intervals. To verify against
the private originals without any model import or GPU execution:

```sh
set -o pipefail
age -d -i "$CONTROLLER/result.identity" "$CONTROLLER/evidence.tgz.age" | cmp - "$CONTROLLER/evidence.tgz"
python3.12 -I -S -B summarize_hosted_trio.py "$CONTROLLER" "$ORIGINAL_FAILURE_ZIP" "$PRIVATE_TRIO" | cmp - hosted_metal_trio.json
```

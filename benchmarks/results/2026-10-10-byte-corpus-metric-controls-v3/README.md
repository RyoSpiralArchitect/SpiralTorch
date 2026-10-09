# Byte corpus: metric by trainability

This extends the [three-arm pilot](../2026-10-09-byte-corpus-frozen-geometry-v2/README.md)
to five arms per seed: ordinary attention, Poincare/train, Poincare/frozen,
flat/train and flat/frozen. Rust owns all learning and update policies;
independent CPU-f32 PyTorch supplies a numerical reference, not a production
SpiralTorch learner. All four geometric arms start from identical float32
parameters and use the same backbone, corpus windows and SGD budget.

## Measured Effect

Native held-out bits per byte (BPB, lower is better):

| Seed | Ordinary | Poincare/train | Poincare/frozen | Flat/train | Flat/frozen |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 5.13633765 | 5.13636467 | 5.13645244 | 5.13626683 | 5.13633148 |
| 23 | 5.12804811 | 5.12497554 | 5.12486331 | 5.12554833 | 5.12549293 |
| 37 | 5.12810194 | 5.12610018 | 5.12614373 | 5.12643449 | 5.12646617 |

Flat minus Poincare in the trained arms is -0.00009784, +0.00057279 and
+0.00033431 BPB respectively. Learning the geometric weights helps seeds 11
and 37 but hurts seed 23 for **both** distance rules. No unfavorable arm or
seed is removed. These are small, mixed effects, not general evidence of a
superior geometry or proof that learning geometry is necessary.

Every contrast is computed from the original held-out batch losses, not the
tolerance-accepted report summary. The comparison files retain all seven
seedwise contrasts, their independent Torch values, observed numerical
discrepancies and sign agreement. This is numerical reproducibility, not a
statistical significance test or an error bound for unseen runs.

All 15 cases pass the frozen numerical criteria in native Metal and actual
browser WebGPU. All seven contrast signs agree with Torch for every seed in
both runtimes. Their largest observed contrast discrepancy is `1.29e-7` BPB
native and `1.40e-7` browser. This is much smaller than the reported effects in
this recipe, but is not a bound on sampling variation or generalization.

| Maximum error against Torch | Native | Browser |
| --- | ---: | ---: |
| Training CE | 9.536743e-7 | 1.430511e-6 |
| Held-out batch CE | 1.430511e-6 | 9.536743e-7 |
| Final parameter | 5.960464e-7 | 3.576279e-7 |
| Trained geometry change-vector relative L2 | 1.928225e-5 | 9.870353e-6 |

The nine inherited full case reports, including weights and losses, remain
strictly identical to their own retained v2 native/browser baselines after
removing the new metric field. This is checked separately from input lineage.

## Fixed Design

- Seeds 11/23/37, batch 2, 32 byte positions, width 16, hidden width 32,
  two attention heads, one decoder block, geometry width 4, curvature -0.75.
- Float32 SGD 0.05, 128 updates per arm, 8,192 training target bytes;
  validation at 0/64/128 on 2,048 held-out target bytes. Topos stays off.
- The pinned document revision, five document paths, windows, rate and all
  nine previous case initializations are unchanged. See preparation.json.
- Ordinary: 11,216 scalars, all trainable. Geometry: 11,290 scalars;
  frozen arms train only the 11,216 backbone scalars. Equal trainable counts
  do not make ordinary and geometric models architecture/compute matched.
- All four geometric arms share causal complex recurrence and bounded chart.
  Flat replaces squared Poincare distance with `4 * ||x_q - x_k||^2`, retaining
  the learned positive per-head gain. The factor 4 matches only the origin's
  local metric, not every initial attention bias or the complete function.
- Frozen means zero geometry parameter update rates, not detached coordinates
  or embedding pullbacks. Coordinates still change with the learned backbone.
- The flat control is still a nonlinear coordinate model, not a linear
  Euclidean encoder or a curvature-to-zero limit. Bias scale, distance shape
  and optimization conditioning are not isolated from one another here.
- Fixed 256-byte symbols and tiny repository prose, not pretrained LLM FT,
  large-corpus quality, generation evaluation or a speed benchmark.

Request SHA-256:
`759ed1282bad72c2351fcc9c78fef0f661087bf57f578f96e741882763204849`.
The independent reference was frozen before either device run. Its nine
inherited case reports are strictly identical to the retained v2 reference.

## Protocol And Controls

Study request/result/partial/checkpoint schemas are v3. Each seed must cover
exactly five distinct metric/update modes; the existing 16-case bound permits
three complete seed groups. v1/v2 keep their old contracts and reject explicit
metric fields. Ordinary/Poincare model checkpoints remain v1; only flat
models use v2. This mixed model schema is checked inside one v3 study.

`verify_controls.py` checks exact v2 input lineage and a separately executed
one-step prefix. Same-metric learned/frozen arms must have identical initial
losses and bit-identical first backbone updates, both embeddings must move,
frozen geometry must retain its initial bits, and trained geometry must move.
This complements, but does not replace, the independent full-gradient Torch
tests or the separate 37-to-128 resume verifier.

Both native and browser pass all six metric/trainability first-step pairs.
In these runs every trained geometry family moves already at step 1, while
both embeddings move and frozen geometry retains its exact initial bits.
Each runtime also completed a separate 37-update prefix and resumed from its
saved checkpoint in a fresh process/page. All 15 resumed full reports match
their own uninterrupted baseline with strict, signed-zero-sensitive equality;
the resumed outputs independently pass the same Torch comparator. Checkpoint
parameter bits are bound to the paused report, and no pause-only evaluation
is inserted. No cross-runtime bitwise-equality claim is made.

Numerical criteria are unchanged: `3e-6 + 5e-5 * abs(reference)` for every
training loss, validation batch loss and final scalar. Each trained geometry
tensor requires reference change-vector L2 greater than `1e-8`, nonzero
measured change and relative L2 error at most `0.002`; frozen weights instead
require exact initial float32 bits.

## Reproduction And Scope

[preparation.json](preparation.json) records the fixed input recipe.
[native.json](native.json), [browser.json](browser.json) and [torch.json](torch.json)
retain every scalar trajectory and held-out batch loss. The
[native comparison](native-comparison.json) and [browser comparison](browser-comparison.json)
retain every error and paired contrast. See the two `*-resume-verification.json`
and `*-controls.json` records for continuation and first-step checks, and
[inherited-reports.json](inherited-reports.json) for old-nine-arm equality.
[validation.json](validation.json) binds source/artifact hashes, test results
and review repairs. No numerical criterion was changed after measurement.

Use [the corpus recipe](../../../docs/byte_corpus_learning.md) with
`--metric-geometry-control`, retaining all five document paths, pinned revision
`01e12ac8cc9bb10a1b151a8d7b1fda6b157fea64` and seeds 11/23/37. Prepare and
generate the independent reference before running devices. Run all 128 updates,
a separate stop-37 prefix and a fresh-process/page resume. Also run stop-1
separately; never substitute the uninterrupted report as a claimed resume.

For each runtime run `tools/byte_corpus_study.py compare` on the uninterrupted
and resumed reports, `tools/verify_byte_corpus_resume.py` on its own baseline,
prefix, resumed report and checkpoint, then:

```sh
python3 -I -S -B benchmarks/results/2026-10-10-byte-corpus-metric-controls-v3/verify_controls.py \
  "$RAW/input/request.json" "$V2/frozen/request.json" \
  "$RAW/native.json" "$RAW/native-first-step.json" "$RAW/native-controls.json"
```

Repeat the last command with browser files. Use the actual browser resume page
and download the opaque Rust JSON strings without JavaScript reserialization.
Run `test_controls.py` for the artifact verifier's synthetic mutation tests.

Public scalar trajectories omit final parameter arrays and are explicitly
marked as incomplete comparator inputs. Full requests, tensors, checkpoints,
logs and binaries stay local; their hashes provide identity, not execution
attestation. Earlier experiments are not rewritten. Native and browser
measurements remain separate, and no throughput claim is made.

Read-only independent review closed the summary-derived-contrast issue,
first-step test coverage gap, off-schedule-evaluation acceptance and ambiguous
case-limit test. Final checks pass: 20 Rust study tests, 26 Python tests with
Torch, 13 resume tests and two artifact-test methods covering 14 mutations.
The earlier insensitive 2D synthetic-test failure is retained in local logs;
its sensitive replacement was selected before freezing this study, not after
seeing the corpus result.

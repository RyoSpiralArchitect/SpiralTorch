# Byte Corpus: Strength-Calibrated Geometry Controls

Seven arms per seed extend the [five-arm study](../2026-10-10-byte-corpus-metric-controls-v3/README.md):
ordinary, Poincare trained/frozen, flat trained/frozen, and flat trained/frozen
with one-time RMS-calibrated initial gains. Rust owns preparation and learning;
native, actual browser WebGPU and independent CPU-f32 Torch consume **one frozen
request**, including identical effective float32 initial weights.

## Measured Effect

Native held-out bits per byte (BPB, lower is better), after 128 updates:

| Seed | Ordinary | Poincare/train | Poincare/frozen | Flat/train | Flat/frozen | Calibrated flat/train | Calibrated flat/frozen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 5.13633765 | 5.13636467 | 5.13645244 | 5.13626683 | 5.13633148 | 5.13622833 | 5.13631139 |
| 23 | 5.12804811 | 5.12497554 | 5.12486331 | 5.12554833 | 5.12549293 | 5.12530216 | 5.12523102 |
| 37 | 5.12810194 | 5.12610018 | 5.12614373 | 5.12643449 | 5.12646617 | 5.12611988 | 5.12615323 |

Calibration improves flat's measured BPB in all three seeds, both trained and
frozen. In the trained arms the changes are -0.00003850, -0.00024617 and
-0.00031461 BPB. Relative to Poincare, calibrated flat is -0.00013634,
+0.00032662 and +0.00001970 BPB: the gap narrows in seeds 23/37, while flat's
small advantage grows in seed 11. No unfavorable result is discarded.

Learning geometry still helps seeds 11/37 but hurts seed 23 across all three
geometric treatments. Thus distance choice, initial strength and trainability
are not interchangeable explanations. These small, seed-dependent effects
do not establish general geometry superiority, statistical significance, or
the necessity of geometric learning.

All 21 native and 21 browser cases pass the unchanged Torch numerical gates.
All 14 contrast signs per seed agree with Torch in both runtimes. Maximum
observed contrast discrepancies are `3.01e-7` BPB native and `1.51e-7` browser,
not bounds on sampling uncertainty or future runs.

| Maximum Error Against Torch | Native | Browser |
| --- | ---: | ---: |
| Training CE | 9.536743e-7 | 1.430511e-6 |
| Held-out batch CE | 1.430511e-6 | 1.430511e-6 |
| Final parameter | 5.960464e-7 | 3.576279e-7 |
| Trained geometry change-vector relative L2 | 1.928225e-5 | 9.870353e-6 |

## Fixed Initialization And Learning

The source v3 request, corpus windows and initial fifteen models are unchanged.
Batch 2, context 32 bytes, width 16, hidden width 32, two heads, one block,
geometry width 4, curvature -0.75, Topos off, SGD 0.05 and seeds 11/23/37 remain
fixed. Each arm sees 8,192 training target bytes and 2,048 held-out target bytes;
validation occurs only at 0/64/128. This is tiny pinned repository prose, not
pretrained LLM FT, a large-corpus language benchmark or generation evaluation.

The original geometric arms retain every initial parameter bit. The new pair
changes only per-head/block raw gains, with identical fitted initialization for
its learned and frozen arms. All seven share the non-geometric backbone.
Ordinary has fewer parameters and less compute; this is not a speed comparison.
Flat still uses the causal wave and the same bounded coordinate chart, not a
linear Euclidean encoder or a curvature-to-zero limit.

Calibration uses only preselected training batches `[0,1,2,3]`, totaling 4,224
valid causal pairs per head. It fits row-centered RMS once and remeasures the
actual device output with a fixed relative tolerance `1e-5`. It never consumes
an update or evaluates held-out windows. The native preparation's maximum
realized relative RMS error is `2.091e-7`. Independent Torch remeasurement of
that frozen request has maximum error `9.860e-8` and performs no fitting.

Browser preparation is qualified **separately**, with maximum realized error
`2.111e-7`. Its fitted gains differ slightly from native, up to `2.98e-8` in raw
gain values. That separately prepared request is not used for browser learning.
All three learners use the designated native-prepared request:

`276a2245277cc36995eef024a99614811fe12c5be1a0efd897eda2d27b11afff`

Source v3 SHA-256:
`759ed1282bad72c2351fcc9c78fef0f661087bf57f578f96e741882763204849`.

Every v4 execution rechecks the frozen initial plans, including on resume; it
does not refit gains or mutate resumed weights. Startup forwards/readbacks are
explicit overhead, not a resident-performance optimization. RMS matching does
not match attention patterns, rowwise relationships with QK scores, conditioning
or later dynamics.

## Controls And Reproduction

The inherited fifteen full case reports, including parameter arrays and loss
histories, remain strictly equal to their own v3 native/browser/Torch originals
after removing the new initialization field. Each runtime also executes a
separate first-update control: all nine metric/initialization pairs retain
identical initial losses and first backbone update bits, both embeddings move,
frozen geometry stays bit-identical, and trained geometry moves.

Browser resumes 37 to 128 in a fresh page; native resumes 37 to 64, then 64 to
128 in fresh processes. All 21 final reports in each runtime exactly equal
their own uninterrupted baseline, including signed-zero-sensitive parameter
and history values, and pass the same Torch comparator. Native also captures
the actual initial state at cursor zero and resumes it for the first-update
control. Both runtimes load completed checkpoints without further updates;
those full reports and checkpoints remain exactly equal. Verification adds no
pause-only evaluations or learner revisions. No cross-runtime bitwise equality
is claimed.

Follow the [v4 recipe](../../../docs/byte_corpus_calibration.md), using the
retained source recipe in [source-preparation.json](source-preparation.json).
The same five source paths, pinned revision, seeds, windows and update budget
must be retained. The numerical gates remain `3e-6 + 5e-5 * abs(reference)`;
trained geometry requires reference change-vector L2 greater than `1e-8`,
nonzero measured change and relative L2 error at most `0.002`. Frozen geometry
requires exact initial float32 bits. No gate was retuned after measurement.

The comparison files expose all seedwise contrasts, Torch values, discrepancies
and sign agreement. Native/browser/Torch JSON files publish scalar trajectories
only and are explicitly marked as incomplete comparator inputs. Full request,
model/checkpoint arrays, logs and binaries remain local. Hashes identify retained
artifacts; they are not signatures or independent runtime attestation.
See [validation.json](validation.json) for those bindings and the exact scope.

Local checks pass: 27 Rust study tests, 5 native CLI tests, 33 Python study tests
including optional CPU Torch, and 21 artifact/resume tests. The stdlib-only CI
route also passes, skipping only its six optional Torch tests. Pinned rustfmt,
native/WASM release builds, HTML module syntax and license metadata preflight
pass. This is local qualification, not a claim that this child branch has
already passed hosted CI.

Independent review repaired three verifier false passes: ignored failed Torch
calibration, mutually consistent but truncated/relabelled reports, and boolean
scalars or float dimensions accepted in checkpoint comparisons. Mutation tests
now reject them. The unchanged raw runs were rechecked with the repaired tools.
The early Torch preparation attempt stopped before training because of a loop
variable shadowing the batch size; its log is retained, the bug is fixed, and a
three-seed regression test covers it. No failed attempt is used as evidence.

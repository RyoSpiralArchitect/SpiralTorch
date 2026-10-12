# Headwise Byte-Corpus Controls v6

Implementation: `df31c3684a79fa617c5651cc05533798521f5e65`.
The [frozen protocol](pre-registered.md) preceded v6 measurements.
See the [API and reproduction guide](../../../docs/byte_corpus_headwise_controls.md).

## Scope And Status

The experiment puts fixed per-head geometry **biases into QKV attention**. It
does not replace Q/K/V with manifold-valued projections or learn a geometry
selector. P = Poincare squared, E = Euclidean chord squared, F = categorical
Fisher-Rao squared.

Native, browser and independent Torch: all 63 cases completed and passed the
fixed numerical gates. Native and browser inherited cases, first-update twins,
pause/resume and completed no-op checks passed. Both preparation paths produced
the same byte-identical input. See the [validation manifest](validation.json)
and the [native](native-comparison.json) / [browser](browser-comparison.json)
comparisons for the complete checks and source/raw-artifact hashes.

- Seeds 11, 23, 37; 21 arms per seed; 128 updates per arm.
- All nine v5 arms retained; six ordered heterogeneous routes added, each
  trained and geometry-frozen. Same backbone and geometry initials except
  the explicitly inherited per-head gains. No new fitting or held-out tuning.
- One block, two heads, B=2, T=32, width=16, hidden=32, geometry=4, c=-0.75.
  Plain model: 11,216 parameters; every geometry model: 11,290.
- 8,192 training-target exposures per arm, not 8,192 unique training bytes.
  The same 2,048 held-out targets are evaluated at updates 0, 64 and 128.
- One request across native WGPU/Metal (Apple M4), browser WGPU/WASM, and
  independent CPU float32 Torch. Parameters and update budgets match across
  geometric arms; compute and initial bias patterns do not.

Common request SHA-256:
`262460760d71ed94ae31fb1f5d9b6886c66c5eb2185172fb7c6b0094685f32ca`.

Unchanged v5 source SHA-256:
`95ef8c21c6c9c8221d7aced6adcda9d80ce4c0dccf6ec3819e40475c52c1db38`.

## Numerical Qualification

Maxima against independent Torch:

| Check | Native | Browser |
| --- | ---: | ---: |
| Training CE absolute error | 1.430512e-6 | 1.430512e-6 |
| Held-out batch CE absolute error | 1.430512e-6 | 1.430512e-6 |
| Final parameter absolute error | 5.960465e-7 | 4.768372e-7 |
| Geometry-family learning-delta relative L2 | 4.328799e-5 | 3.927223e-5 |
| Individual head gain-delta relative error | 6.375518e-5 | 5.522727e-5 |

Fixed gates: atol 3e-6 + rtol 5e-5; geometry-family and individual routed-head
gain learning-delta relative error <= 0.002; reference delta norm > 1e-8 and
nonzero measured delta. Frozen geometry keeps all initial float32 bits.
The added per-head gate prevents a large update from hiding a disconnected head.

Native initial RMS maximum relative error: 2.090929e-7; browser preparation:
2.110578e-7; independent Torch: 1.306542e-7. Fixed initial RMS gate: 1e-5.
Each engine measures the declared training windows without refitting.

All 27 inherited v5 full case reports are unchanged within each of the three
engines. All 30 trained/frozen pairs in each GPU runtime have identical initial
evaluation losses, first training loss and first backbone updates, including
both embedding families.
Pausing all 63 models at update 37 and continuing in a fresh process or browser
page produces a byte-identical complete report within each runtime. Completed
no-op resume also preserves those exact report and checkpoint bytes.

## Learning Observations

Native mean held-out BPB differences across the three fixed seeds are below;
negative means lower loss. These are descriptive means, not significance tests.

| Head 0 / Head 1 | Versus P/P Trained | Seeds Better Than P/P | Trained Minus Frozen |
| --- | ---: | ---: | ---: |
| P / E | -0.000026980 | 1/3 | -0.000008065 |
| P / F | +0.000287931 | 0/3 | -0.000012336 |
| E / P | +0.000094877 | 1/3 | -0.000013429 |
| E / F | +0.000395130 | 0/3 | -0.000019122 |
| F / P | +0.000656408 | 1/3 | -0.000028685 |
| F / E | +0.000637504 | 1/3 | -0.000029022 |

No heterogeneous route improves on trained P/P in all three seeds. Every
heterogeneous trained/frozen comparison improves for seeds 11 and 37 and worsens
for seed 23. A small favorable mean is not a stable winner.

Reversing the assignment changes the result under these **fixed, distinct
QKV heads**. This is assignment sensitivity, not proof that the geometries
themselves are noncommutative or that universal head specialization was learned.
There are three seeds, not eighteen independent quality samples.

Native signs for all 225 retained and heterogeneous-head contrasts match Torch.
Maximum discrepancy is 2.257273e-7 BPB among the 162 heterogeneous-head
contrasts, or 3.009697e-7 including the 63 retained contrasts. Browser signs also
match Torch for all 225 contrasts; its maximum discrepancy is 2.149784e-7 BPB.
All routes and all seeds remain in the scalar records, including unfavorable
results. No general LLM quality, pretrained fine-tuning, tokenization advantage
or speed claim follows from this small pilot.

## Review And Reproduction

Read-only independent review found and resolved three P2 boundary issues:
expanded checkpoint-budget preflight, request-derived checkpoint topology
verification, and strict added-arm scalar/dimension types. Redundant preparation
measurements were also removed. Re-review reported no remaining P1/P2, with
nine stdlib tests and 72 rejected mutation probes.

A separate read-only native-evidence audit reproduced the saved verifier
receipts, all table cells and all 225 contrast signs. Its P3 wording correction
clarified the two contrast-error scopes above; it found no P1/P2 contradiction.
It checked 8,064 training losses, 6,048 held-out batch losses, 711,048 final
parameter scalars, 150 trained geometry-family gates and 36 routed-head gain gates.
The browser full-run and continuation evidence was subsequently verified by the
parent using the same numerical, control and exact-resume checks; it was not
part of that independent native-evidence audit.

Parent validation: 36 Rust study tests, five native CLI tests, 51 Python study
tests including optional Torch checks, and 64 artifact-verifier tests passed.
The review changed preparation/preflight only, not learning math or the frozen
input. The uninterrupted native binary and reviewed continuation binary are
retained separately; complete native reports agree byte-for-byte.

Reproduce from the unchanged [v5 source bundle](../2026-10-12-byte-corpus-fisher-controls-v5/)
using the linked guide. Preserve the opaque common request bytes and run every
case. Independently verify preparation, full Torch comparisons, inherited
reports/first-update controls and pause/resume/no-op evidence. Do not promote
a failed or unfinished gate by dropping a route or widening tolerance.

Raw requests, weights, checkpoints, binaries and logs stay local. Only scalar
trajectories/contrasts, validation records, hashes and reproduction instructions
are intended for this directory. No timing measurements are claimed.

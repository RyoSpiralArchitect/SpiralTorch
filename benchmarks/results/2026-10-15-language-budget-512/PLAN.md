# Fresh-Run Learning-Budget Pilot

Frozen before any training on 2026-10-15. This supersedes the unexecuted v1
historical-bitwise-identity pilot, without erasing its failed preflight.

V1 stopped before training: fresh Rust preparation changed some calibrated
Fisher-Rao gains and associated mixture units by small float32 differences.
The old request is therefore NOT an exact initial-condition match. Do not
relax that failed gate or present this as an extension of the historical run.
Retain the v1 preparation packets and report all initial differences.

V2 takes those same already-verified fresh preparation bytes as its initial
conditions. Its primary contrast is 128 versus 256 versus 512 updates WITHIN
THIS NEW RUN. All three observations share exact initial weights, batch prefix,
preparation and execution runtime. Historical outputs are context only.

## Fixed Recipe

- Source documents at 01e12ac8cc9bb10a1b151a8d7b1fda6b157fea64.
- Training: docs/reference/native-api.md, docs/reference/learning.md,
  docs/geometric_learning_bridge.md. Validation: docs/reference/geometry.md,
  docs/spiraltorch_manifesto.md. No new corpus acquisition.
- Seed 11: first historical seed, not selected as a best-performing seed.
- All 25 v8 arms; ordinary, frozen geometry, Poincare/chord/Fisher controls,
  fixed/trainable mixtures and query projections. No mid-run arm selection.
- B=2,T=32,width=16,hidden=32,heads=2,one non-Topos block,
  geometry_cols=4,curvature=-0.75,stateless SGD float32 rate=0.05.
- 512 updates/arm (32768 target-byte presentations), stop cursors128/256/512.
  Existing validation schedule every64,32 batches. Fixed training-only
  calibration indices [0,1,2,3]. Rust prepares and trains; Python orchestrates.
- Verify all three v3->v5->v7->v8 preparation packets again, unchanged data and
  shapes, exact training prefix, case policy differences and parameter-bit
  differences against the historical request; record differences, not parity.
- Resume only with the exact new request and its own opaque checkpoint.

## Measurements And Limits

- One native GPU run (maximum12800 accepted model updates total) plus one
  independent CPU float32 PyTorch reference. Existing numeric gates unchanged.
- Report every arm's complete held-out CE/bits-per-byte trajectory and failures.
- Add-one unigram/bigram fits use only training windows seen through each
  checkpoint; never future training or validation labels.
- For ordinary/Poincare-chord query-train/query-frozen, measure32 greedy bytes
  on the previously frozen first autograd-prose cohort batch at128/256/512.
  No sampling, prompt search or repetition penalty.
- No geometry-superiority, speed, three-seed or broad language-quality claim.
  No production promotion or PyPI update. Keep corpus, weights and raw scores
  private; publish aggregate results, hashes, verification and reproduction.

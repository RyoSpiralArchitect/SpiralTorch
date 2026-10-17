# Learning-Budget Pilot

Frozen before training on 2026-10-15. This is a learning-budget diagnosis, not a
production promotion, broad language-quality benchmark or speed comparison.

## Fixed Recipe

- Source documents at 01e12ac8cc9bb10a1b151a8d7b1fda6b157fea64.
- Training: docs/reference/native-api.md, docs/reference/learning.md,
  docs/geometric_learning_bridge.md. Validation: docs/reference/geometry.md,
  docs/spiraltorch_manifesto.md. No new corpus acquisition.
- Seed 11, selected as the first historical seed, not as a best-performing seed.
- Keep all 25 v8 arms, including ordinary, geometry-frozen, metric controls,
  fixed/trainable mixtures and query projections. Do not select winners mid-run.
- B=2, T=32, width=16, hidden=32, heads=2, one non-Topos block,
  geometry_cols=4, curvature=-0.75, stateless SGD float32 rate=0.05.
- 512 updates per arm (32768 target-byte presentations); checkpoints at absolute
  cursors 128, 256, 512. Existing validation schedule every 64 updates, 32 batches.
- Training-only RMS preparation uses fixed indices [0,1,2,3]. Rust performs
  calibration, mixture and query preparation. Python only orchestrates.
- Verify the first 128 training batches, all validation batches, model shapes,
  rate and every initial float32 parameter bit match the historical seed-11
  request before training. If any differs, stop; do not call it a budget-only
  comparison. New request identity, fresh training from zero, not a relabeled
  historical checkpoint or transplanted optimizer clock.

## Measurements And Limits

- Independent CPU float32 PyTorch reference on the same request, existing
  numerical tolerances unchanged. Passing is numeric qualification, not quality.
- Native installed Python binding executes Rust/WGPU, no CPU fallback. No
  simultaneous local GPU/build jobs. Resume only with the exact new request.
- Report complete 25-arm held-out CE/bits-per-byte trajectories and failures.
- Compare each checkpoint against add-one unigram/bigram baselines fitted only
  to the training windows actually seen through that checkpoint, not future data.
- Ordinary, Poincare/chord query-train, query-frozen: same previously frozen
  generation prompts and 32 greedy bytes at all three checkpoints. No sampling,
  repetition penalties, prompt search or new quality threshold after results.
- At most 512 updates x 25 arms, one local GPU run plus one CPU reference. This
  one-seed pilot is not three-seed evidence. More seeds require a later recorded
  experiment; keep all outcomes, not just a winning arm or checkpoint.
- Keep raw documents, checkpoints, score arrays and weights private. Publish
  aggregate results, hashes, verification and reproduction code only.
- Do not promote a serving model or change the PyPI release in this pilot.

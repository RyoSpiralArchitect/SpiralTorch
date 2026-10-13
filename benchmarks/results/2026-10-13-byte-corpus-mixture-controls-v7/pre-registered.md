# Byte-Corpus Mixture Controls v7: Frozen Measurement Protocol

Declared before preparation or learning measurements. This is a bounded
document-held-out byte-model pilot, not large-model efficacy or speed evidence.

- Fork the exact prepared v5 request, SHA-256
  `95ef8c21c6c9c8221d7aced6adcda9d80ce4c0dccf6ec3819e40475c52c1db38`.
  Keep all nine original cases per seed, parameters, documents, windows, rate
  and evaluation schedule unchanged. v6 remains its separate fixed-head study.
- Add all four distinct subsets: P/E, P/F, E/F and P/E/F. P = Poincare squared,
  E = origin-scaled Euclidean chord squared, F = categorical Fisher-Rao squared.
  Each subset has a learned-router and fixed-router arm. Both train the shared
  coordinate encoder, causal wave and shared gains, preserving full pullbacks.
- Initialize routers with canonical +0 logits (uniform softmax). Derive fixed
  per-head component unit scales from the v5 matched positive-gain ratio to P.
  Canonical unit ratios admit relative rounding error <= 1e-6 across hosts.
  Fit only each mixture's initial shared gain, once, to the P row-centered RMS
  on the v5 predeclared training windows. Initial RMS gate remains 1e-5.
  No held-out fitting, resume refitting, router-rate multiplier or search.
- Three seeds 11, 23, 37; 17 arms per seed, 51 complete cases; 128 SGD updates
  per case. B=2, T=32, width=16, hidden=32, heads=2, one block, geometry=4,
  curvature=-0.75. Equal update/data budgets do not mean equal compute across
  different metric subsets; the learned/fixed-router pair runs the same math.
- Compare every training loss, held-out batch loss and final parameter against
  independent one-thread CPU-f32 Torch. Preserve 3e-6 + 5e-5*abs(reference).
  Every learned geometry-family and every individual router component's final
  learning delta must have reference magnitude/norm > 1e-8, nonzero actual delta
  and relative error <= .002. Frozen router values retain initial float32 bits.
  A failed sensitivity gate is retained, not silently omitted or widened.
- Verify all inherited v5 reports unchanged, plus first-update twin equality:
  initial evaluation, first training loss and every non-router parameter update
  must be bit-identical within each runtime. Fixed routers are not detached.
- Use one opaque prepared request for native WGPU, browser WASM and Torch.
  Native/browser preparation may differ by rounding; never substitute distinct
  fitted requests per engine. Every learner remeasures but never refits it.
- Pause all cases at update 37, resume in a fresh process/browser page, compare
  with uninterrupted reports exactly within each runtime, and test completed
  no-op resume. Keep full checkpoints and raw weights local.
- Primary descriptive contrast: learned-router minus fixed-router held-out BPB
  for each subset and seed. Also retain comparisons with trained P, calibrated
  E and calibrated F. Report all signs and engine discrepancies, including
  unfavorable or numerically unresolved contrasts. Three seeds, not twelve
  independent quality samples; no significance or general superiority claim.
- Publish scalar histories/contrasts, checks, hashes and reproduction only.
  No raw checkpoints, full weights, binary artifacts or private request archive.

# Headwise corpus controls v6: frozen protocol

This protocol was written before preparing or measuring v6. It extends v5,
not a replacement selected after looking at held-out losses.

- Source: v5 request SHA-256
  `95ef8c21c6c9c8221d7aced6adcda9d80ce4c0dccf6ec3819e40475c52c1db38`.
- Preserve every v5 case, parameter, document, window and update unchanged.
- Seeds 11, 23, 37; one block; two ordered heads; batch 2; length 32;
  width 16; hidden 32; geometry 4; curvature -0.75; no Topos; 128 SGD updates
  at 0.05. Evaluate at 0, 64, 128 on the same 2,048 held-out target bytes.
- Retain all nine v5 arms per seed. Add P/E, P/F, E/P, E/F, F/P, F/E, each
  trained and geometry-frozen: 21 arms per seed, 63 total.
  P = Poincare squared; E = Euclidean chord squared; F = categorical
  Fisher-Rao squared. Assignments are fixed, not a learned mixture or selector.
- Each head inherits its corresponding head's raw gain from the unchanged
  P-original, E-RMS-matched or F-RMS-matched v5 arm. No fitting or reinitializing.
  All other weights are bit-identical within seed; train/frozen twins identical.
- Native WGPU, browser WGPU/WASM, independent CPU Torch consume the exact same
  Rust-prepared request bytes. Initial causal, row-centered bias RMS per head
  must still match the original P target within 1e-5 on training windows
  0, 1, 2, 3. No held-out calibration; no update consumed during qualification.
- Retain numerical tolerances atol 3e-6, rtol 5e-5, geometry learning-delta
  relative L2 <= 0.002 and reference delta norm > 1e-8. Additionally each
  heterogeneous head's scalar gain delta must satisfy the same relative bound
  and norm floor, with nonzero measured delta. A failure is not permission
  to relax these gates or omit an arm.
- Compare every held-out batch loss and every training loss/final parameter.
  Check all 27 inherited v5 full case reports exactly within runtime.
  Separately run first-update controls for all 30 trained/frozen geometry pairs.
  Pause at update 37, resume on a fresh runtime to 128, then completed no-op
  resume: complete reports and saved parameter/history bits must agree.
- Publish all scalar trajectories and contrasts, including per-route/per-seed
  learned-minus-frozen, each homogeneous comparator, and reversed-head order.
  No winner-only report, no general quality claim, no speed claim.
- Keep full requests, weights, checkpoints, binaries and logs local; publish
  hashes, validation records and reproducible instructions. Record any failed
  attempt or uncompleted gate explicitly.

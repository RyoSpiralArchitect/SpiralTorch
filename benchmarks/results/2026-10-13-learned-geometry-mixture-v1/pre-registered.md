# Learned geometry mixture v1: correctness protocol

This protocol precedes the native/browser full-model measurements. This is a
synthetic numerical qualification, not a corpus-quality or speed comparison.

- Mix Poincare squared, origin-scaled Euclidean chord squared and categorical
  Fisher-Rao squared in that order. Learn a softmax logit vector per block/head;
  preserve a separate shared softplus gain per head and fixed positive scales.
- Two independent CPU-f32 Torch fixtures, seeds 1761 and 1863: one block with
  zero Q/K; two blocks with QKV content, external score biases and Topos.
  Both use B=2, T=4, width=4, two heads, geometry=4, curvature=-0.75.
- Fixed scales per block/head are [[1,.8,24],[1,1.2,48]] and
  [[.7,1.5,36],[1.3,.6,40]]. They are synthetic sensitivity controls, not learned
  values or corpus RMS calibration. No scale is fit to held-out performance.
- Compare every initial logit/parameter VJP, embedding/external-bias VJP and
  all 16 SGD updates at 0.125, including each geometry derivative and router
  mass. Preserve atol 3e-6 + rtol 5e-5 and geometry relative L2 <= 0.002 with
  reference norm > 1e-8 and nonzero actual gradient. Apply the relative gate
  separately to every router head and each component's gradient history.
- Preserve causality, off/detached-geometry sensitivity, immutable tapes,
  stale/foreign ownership rejection and whole-model atomic updates.
- Save after update 7 and restore a fresh owner. All 16 losses, weights,
  geometry/embedding VJPs and pre-update router masses must match the
  uninterrupted trajectory bit-for-bit within each runtime. Bind fixed
  scales, ordered metrics and learned logits in checkpoint v4.
- Make a finite component overflow while its softmax mass is exactly zero.
  It must guard the full forward/VJP family and reject even a zero-rate update
  without changing any parameter. A zero mass must not conceal failure.
- Run the same opaque fixture bytes through native WGPU and browser WASM;
  independent Python is an oracle/verifier, not production geometry semantics.
- Retain failures and raw weights/checkpoints locally. Publish only scalar
  verification evidence, hashes and reproduction instructions. A failing gate
  is not permission to widen tolerance or drop a component/head.

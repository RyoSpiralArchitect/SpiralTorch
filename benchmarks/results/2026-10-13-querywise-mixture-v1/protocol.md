# Query-Local Mixture Qualification

Declared before the native/browser qualification. The earlier CPU sensitivity
probe is exploratory, not a preregistered causal or quality study.

- Freeze fixture SHA-256 `cf340ca080983aa9669c72adb1070a83248484cef600b6c9e03e98a6c6c47e70`.
  It contains 24 first-training-batch probes from all twelve v7 learned-mixture
  arms, at initial and native-update-128 weights, plus three synthetic cases
  with distinct batch/head/query logits and 2, 3 and 8 components.
- Query-local components have shape `[B,H,Q,S,K]`, logits `[B,H,Q,K]`.
  Softmax is over components; logit pullbacks sum over keys only. The primitive
  neither creates a causal router nor replaces the byte model's shared router.
- Execute the same fixture on native WGPU and browser WASM/WebGPU. Compare
  forward, softmax masses, both VJPs, one SGD step at rate 0.05 and its forward
  with independent CPU-f32 Torch. Preserve `3e-6 + 5e-5 * abs(reference)`.
- Require a router-gradient reference L2 norm greater than `1e-8`, relative L2
  error at most `0.002`, and nonzero actual entries where reference magnitude
  exceeds `1e-8`. These are primitive-gradient gates, not quality criteria.
- Exercise strided operands, bit-exact repeated backward, retained forwards,
  and guarded component/logit/seed inputs. Invalid gradients must reject the
  entire two-parameter SGD update, including rate zero, without mutation.
- Keep fixture tensors, weights and executables local. Publish scalar results,
  all exploratory sensitivity rows, source/artifact hashes and reproduction.
- No input-conditioned LM, large-scale quality benefit, causality of arbitrary
  caller-supplied logits, equal-compute comparison or speedup is established.

## Review Reinforcement Before Remeasurement

The first 27-case reports passed the checks above but an independent review
found that absolute tolerances could hide a skipped tiny router update, and
two simultaneously guarded derivatives did not test a healthy sibling's atomic
rejection. Retain those original v1 reports, then use validation v2:

- Require relative L2 error <= `0.002` for each component/logit tensor's actual
  SGD delta, reference delta L2 > `1e-8`, and an actual nonzero update. Explicitly
  reject frozen component and router updates in every case.
- Bind a separate healthy parameter with a nonzero gradient alongside invalid
  derivatives, and require all three parameters to remain bit-identical. A
  standalone positive control must update that healthy parameter successfully.
- Reject BrowserWebGpu and unknown backend identity in the native snapshot
  probe. This is metadata consistency, not cryptographic provenance.
- Preserve the fixture, all earlier tolerances and raw original reports.

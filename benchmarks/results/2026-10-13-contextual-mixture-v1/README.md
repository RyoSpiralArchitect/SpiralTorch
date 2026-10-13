# Contextual Geometry Mixture: Wired, Not Fully Qualified

Status: **experimental; native passed, browser qualification failed** under the
same frozen gate. No threshold was relaxed and no formal PR is opened here.

The Rust byte decoder can now learn query-local mixture logits from its causal
wave coordinates. Projection, shared-router, metric, wave, embedding and QKV
parameters share the existing full-model update owner. Checkpoint v5 retains
the projection graphs; shared-only v4 remains unchanged. See the
[API and reproduction recipe](../../../docs/contextual_geometry_mixture.md).

## Executed Qualification

Four synthetic fixed fixtures combine zero/nonzero router initialization with
one/two blocks, using Poincare, Euclidean and Fisher-Rao components. Each runs
sixteen full-model SGD updates plus checkpoint/resume, causality, retained-tape
and late-query-failure checks. Two-block cases include ordinary QKV, external
biases and Topos; one-block cases isolate the geometric score with zero Q/K.

Native Metal passed all four. Browser WebGPU passed the ordinary output/VJP,
causality, guard and same-runtime exact-resume checks, but failed the additional
router-update-delta gate in the zero-initialized two-block case. The other three
cases passed their query checks. This is **not** an all-runtime qualification.

After independent-review repairs, 27 byte-decoder tests (including enabled native
GPU checks) and two validation-checker tests passed. Native and browser were both
remeasured; their respective raw reports are byte-identical to the pre-repair
reports, including the unchanged browser failure.

The failed step is revision 14, `geometry.mixture_logits.1`:

- Reference update norm: `6.812899059e-6`.
- Update-delta relative L2: `0.004374396586`, exceeding the fixed `0.002` gate.
- Gradient relative L2 at that step: approximately `8.7956e-6`.
- Maximum parameter difference from Torch at that step: one FP32 ULP.
- Recomputing separately-rounded FP32 SGD from actual browser gradients and
  prior values matches all 3,456 measured router/projection scalar updates bit
  for bit, including the failing step. No omitted update was detected.

This localizes the observed mismatch to the cross-reference rounded trajectory,
not a missing SGD application. It does not prove that every upstream numerical
difference is harmless. The original promotion failure remains in the record;
the read-only SGD diagnostic is not a replacement acceptance test.

Zero-initial projections recover the initial shared-mixture forward within
`7.451e-8` maximum error in the first native run. Nonzero cases expose the
router-to-coordinate pullback, and the detached-router negative control rejects.
An invalid last-block query projection guards all model derivatives and rejects
both zero-rate and nonzero-rate updates without partial parameter mutation.

## Boundaries And Next Work

This adds a real LM learning path, not a large-corpus quality result. The generic
linear projection has its own trainable bias as well as a shared intercept;
parameter counts and constant-direction update dynamics are therefore not
matched to the shared-only control. No general quality or speed claim is made.

Before a broad geometry sweep, separate pre-rounded gradient/step agreement from
the discontinuities of FP32 parameter rounding in a newly declared experiment,
while retaining this failed gate. Then declare an explicit projection-bias
policy and parameter budget for the shared/contextual corpus comparison.
Python exposure and new geometric families remain follow-up work.

Raw fixtures, tensors, model/checkpoint values and executables stay local.
Published artifacts contain scalar checks, fixed-gate failure details, hashes
and reproduction instructions only. Hashes bind retained bytes, not execution
attestation. See [protocol](protocol.md) and [review](independent-review.md).

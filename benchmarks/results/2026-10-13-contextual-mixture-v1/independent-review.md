# Independent Review

A read-only core reviewer inspected the new parameter-layout, causal projection,
VJP and checkpoint v5 additions relative to `4e2e4290`. No actionable P1/P2 issue
was found. The review traced shared-table rebinding, exact VJP separation,
router-to-coordinate accumulation and owner/checkpoint ordering.

The reviewer noticed that the parameterless-graph rejection test also had a
wrong output width. The test was changed to a two-component mixture with matching
input/output width, so it isolates parameterlessness rather than shape failure.

Core review did not execute builds, GPU, browser or Torch and is not runtime
attestation. Its initial runtime-evidence gaps are addressed separately by the
recorded native qualification and browser failure, not retroactively called
independently verified. A second measurement-source review is tracked below.

## Measurement Review

A second read-only reviewer found one P2 coverage gap: empty mixture-weight
collections could skip the initial/training/resume comparisons, while the
resume report still claimed exact weights. The executed reports contained all
snapshots; the finding concerns what a future regression could evade.

The shared harness now derives the expected snapshot count from the plan and
requires exactly one tensor per mixture block at initialization, every update
and every resume step. Regression checks reject missing-all, missing-some and
excess collections. The fixed numerical gate is unchanged.

The reviewer independently recomputed the stored browser scalar updates with
standard-library arithmetic: all 3,456 matched their own FP32 SGD candidates.
It located the sole delta-gate failure at step 14, shared router block 1,
component 4: one ULP (`2.9802322388e-8`). This diagnosis does not convert the
browser failure into a pass. No builds, Torch, GPU/browser or network execution
were performed by this reviewer.

The same reviewer then rechecked the count-enforcement repair and reported the
P2 closed with no new P1/P2 findings. It traced all three helper call sites and
inspected the negative-count tests without executing them. The parent separately
ran 27 byte-decoder tests with GPU runtime tests enabled and two checker tests.
The post-repair native report is byte-identical to the original native report.
Closing this coverage defect does not close the frozen numerical failure.

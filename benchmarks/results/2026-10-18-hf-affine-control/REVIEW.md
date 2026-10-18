# Independent Read-Only Review

Scope: new Rust affine snapshot/VJP, Python and WASM transport, portable recipe,
HF test wiring, matched-run orchestration, continuation checker and publication
claims. The reviewer did not edit files, build packages, train models, execute
tests or review unrelated PRs. All runtime qualification was performed separately
by the implementing agent.

Findings resolved before publication:

1. Native VJP copied an arbitrarily sized direction before rejecting its shape.
   Exact saved length is now checked before allocating; regression added.
2. A finite branch could overflow at the Python residual addition. The complete
   residual now must be finite; an extreme finite-f32 regression exercises this.
3. The shared fixture test constructed integer activations. It now explicitly
   uses float32 and reaches both reconstruction and invalid-recipe assertions.
4. A failed comparative run could leave no structured public-facing condition
   record. The driver now declares all six cases before execution, records each
   status by atomic replacement, retains completed/failed/not-started cases,
   sanitizes exception text and rethrows without retry. Four stdlib tests cover
   this path and the verifier's device selection. This is not crash recovery.
5. The continuation verifier always required an MPS RNG companion. It now
   validates matching supported devices and only requires that companion on
   MPS. The actual MPS comparison was rerun after this fix.

Each focused follow-up was independently inspected. Final review found no
remaining actionable publication inaccuracies or verifier bugs. The reviewer
confirmed the negative paired results and no-promotion boundary were accurately
described. This is static review closure, not an independent rerun of the
reported hardware or language experiments.

The original successful six-arm outputs remain private and unchanged. A full
fresh replay using the repaired manifest driver reproduced all six artifact IDs
and evaluation values. No previous result is retroactively attributed to the
new failure-recording implementation.

All 8,234 intentional tracked deletions were preserved and are outside this
change. CI success and parent-stack integration remain separate merge gates.

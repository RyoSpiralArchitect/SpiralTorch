# Independent Review

A read-only local reviewer inspected the new HF integration, layout identity,
bindings, tests and registration boundaries. No files, Git state, dependencies
or model data were changed by the reviewer. This was a review of source, not a
GitHub approval or evidence of language-quality improvement.

## Findings And Corrections

1. **P2: Unequal masked-label counts misweighted smoothed accumulated gradients.**
   HF's mean-per-microbatch smoothing did not match the combined batch. The
   integration now uses the existing HF causal smoother through its loss hook,
   weighting by valid shifted targets over the complete accumulation group.
   Regression tests include both ordinary and smoothed loss with unequal masks.
2. **P2: HF 4.57.6 omitted MPS device RNG in checkpoints.** The extended HF RNG
   hooks now persist and restore a required `spiraltorch-mps-rng.pt` companion,
   validated before geometric parameter copying. The test exercises actual MPS
   dropout continuation and rejects a missing companion.
3. **P2: Existing but unused insertion paths silently bypassed geometry.**
   `GeometryAdapterStack.attach(..., require_all=True)` tracks actual execution
   and rejects missing sites. Trainer enables it per training/evaluation scope;
   standalone optional branches retain their old default behavior. Failure
   cleanup and no-optimizer-update assertions cover both execution modes.

The reviewer independently re-probed masked/uneven microbatches, strict placement
execution/cleanup/backward compatibility, MPS RNG state/next-draw restoration,
and missing-companion rejection before copying parameters. All three findings
were closed with **no residual actionable findings in the inspected fixes**.
The causal counting fix and the parent's two optional-`nn` registration guards
also had no additional findings. Final rebuilt-wheel, full MPS training and
WASM fixture checks belong to the main qualification, not the independent probe.

Scope remains bounded: no claim about arbitrary models, distributed execution,
CUDA numerical behavior, untrusted HF checkpoints, complete crash-atomic saves,
or general-language quality. Source and binary identities are in `validation.json`.
## Later-Epoch Follow-Up Review

An independent read-only local reviewer examined the epoch-seeded sampler fix.
The first pass found that epoch-only stopping callbacks could leave a fractional
Trainer epoch and repeat the prior permutation. The fix rejects that unsupported
cursor transition. A second pass found that logging/evaluation callbacks could
request the stop after the initial guard but before saving. A checkpoint-entry
guard and a post-parent-hook guard now close those paths.

The final narrow pass found no actionable residual in these fixes. Eight
model-free callback probes covered log/evaluate, save/no-save and epoch-only/
whole-run stopping. The reviewer did not run training/builds or independently
repeat the final-wheel or real-GPT2 qualification. The main agent owns those
execution results; this is not an exhaustive repository review.

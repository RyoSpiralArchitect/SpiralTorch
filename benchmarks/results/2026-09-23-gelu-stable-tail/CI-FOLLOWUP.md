# Classification Regression After Publication

This archive measures `e97bba6454d374e7165632bbe906c9730edeaffc`, which
used the exp identity over the whole non-saturated GELU derivative domain.
Its recorded single-operation accuracy/performance results remain valid for
that source, but they are not acceptance of the complete training path.

CI run `35825014376`, WGPU job `107064747172`, passed the real-GPU GELU
checks and then failed the existing `resident_graph_training` integration
test. The full local copy of the job log has SHA-256
`91d4f6c908279305a612c3a719dee85333f71d23b339f531824364d8434fbc94`.
The same failure reproduced on the local Apple M4; it was not dismissed as
CI noise or resolved by retrying until green.

Diagnostics isolated classification seed 17, no label smoothing,
ModuleCompatible gain normalization, zero-based step 25. Input-gradient
element 0 was -0.020603502 locally (-0.020603504 in CI), against the ordinary
CPU Module reference -0.016700737. Existing inputs, rates, step count and
tolerance (2e-5 + 2e-4 * abs(reference)) were not changed. The diagnostic
stdout hash is `5e7cd8a92045695c5ba04ed987831f86b2949eb0625014019037ef924145bfb4`.

A controlled local substitution of only the old derivative expression made
the entire integration test pass; its stdout hash is
`1bed6301d23926bb2ecc7f7035074e8746b620fd0ccb59d4b62b118d4d413592`.
Thus a mathematically equivalent single-op expression was not sufficient to
preserve this independently evolving CPU/GPU training trajectory. A ReLU
branch crossing is a possible amplification mechanism, not a directly
observed intermediate in these logs.

The follow-up retains the established central evaluation at abs(x)<=3,
uses the stable exp identity outside it, and keeps the existing exact
abs(x)>=10 policy. The local graph integration passes with this hybrid.
Additional adjacent-f32 tests cover both sides of +/-3 without replacing
the original GELU domain fixtures or widening any tolerance. New numerical
and timing evidence must be collected for that new source, not attributed
to the intervals in this archive.

PR state and CI acceptance are separate from this historical evidence:
[PR #2118](https://github.com/RyoSpiralArchitect/SpiralTorch/pull/2118).

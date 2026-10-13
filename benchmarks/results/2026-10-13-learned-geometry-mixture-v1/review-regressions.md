# Post-Protocol Numerical Review Addendum

These controls were added after the first native full-model measurements and
independent source review. They do not retroactively alter the original frozen
Torch fixture, seed, tolerance, topology or 16-step qualification protocol.

The first implementation narrowed unweighted router reductions to float32.
For components `[1,2]`, equal masses and cotangent `f32::MAX`, the final router
VJP is finite `[-MAX/4,MAX/4]`, but the intermediate `2*MAX` was rejected.

Keeping the reductions wide fixed that case but left another: rows
`[[MAX,MAX],[1,2]]` with unit cotangents lose the small signal if centering
happens after reduction. The required router VJP is `[-0.25,0.25]`, not zero.
The corrected implementation takes component differences at each position
before accumulation, and remains wide until the final requested derivatives.

CPU and native/browser GPU regressions cover finite large VJPs, this mixed-scale
case across both positions and batches, and identical components with large
cotangents (exactly zero router VJP). `[1,10]` with cotangent `MAX` has genuinely
nonfinite final derivatives and must still guard the full VJP family.
The browser and native harness export actual observations; a separate Python
float64 analytic control validates them without relying on Torch float32
intermediate overflow. Existing full-model tolerance gates stay unchanged.

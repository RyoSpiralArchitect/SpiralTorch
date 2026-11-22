# Radius 4 recovers local margin sensitivity without rotating the gradient much

On the 24 training prefixes captured from GPT-2 small, R=4 was the **smallest
tested radius** to pass the predeclared rule on both hosts. This measures a
last-attention parameter derivative at fixed incoming adjoints, not an improvement
in the verb predictions. Nothing was trained, rescored or decoded for this sweep.

All 48 gate/bias pairs at R=1 reproduced the parent f32 VJPs byte-for-byte.
Rust then evaluated the five alternatives on those same 48 input/adjoint pairs:
288 site forwards and 288 parameter VJPs in total, zero decoder calls. The
real attempt took 0.630 s wall time (90 s bound) and was not retried.

Medians within each host, in **affine / WaveGate** order. The affine comparator
is the stored local affine gradient on the *same row and arriving adjoint*.
The forward shift is relative to the original WaveGate residual, not the full
site activation; it is the same for both hosts because their inputs match.

| Radius | Norm / affine | Norm / R=1 | Cosine to R=1 | Forward shift | Joint passes / 24 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | .0814 / .0819 | .500 / .500 | 1.00000 / 1.00000 | .500 | 0 / 0 |
| 1 | .1628 / .1638 | 1 / 1 | 1 / 1 | 0 | 0 / 0 |
| 2 | .3213 / .3253 | 1.985 / 1.985 | .999997 / .999997 | .985 | 0 / 0 |
| **4** | **.5678 / .5738** | **3.534 / 3.534** | **.999794 / .999796** | **2.533** | **23 / 23** |
| 8 | .7826 / .7868 | 4.817 / 4.817 | .999170 / .999175 | 3.809 | 24 / 24 |
| 16 | .8732 / .8818 | 5.361 / 5.361 | .998803 / .998812 | 4.346 | 24 / 24 |

At R=4 both blocks recovered: gate norms were .5068 / .5174 of local affine
and bias norms .6340 / .6340. All ratio/cosine and forward denominators were
defined. The single miss on both hosts was `doctor.00`: norm/affine was
.49369 / .49897, below .50, even though norm/R=1 was about 3.50 and cosine
exceeded .99997. It remains in the denominator. The minimum R=4 cosine was
.999225 / .999230.

The local output did change: the median shift at R=4 was 2.53 times the
original small residual, or **0.2195% of the input norm**. Keeping the old
upstream adjoint is therefore a counterfactual, not a claim that a decoder
run at R=4 would send back the same signal. Also, a training rule that normalizes
every update to length 1/256 discards a pure gradient scaling. The 3.53x norm
result alone is not a prediction of a 3.53x larger parameter step or an
accuracy gain. R=4 is a candidate for a matched full-model comparison, not
a new default, continuous optimum, or independent test-set result.

## Checks and provenance

- Four standard-library tests passed. Four disjoint synthetic central differences
  (gate and bias at R=.5 and R=4) had a maximum relative error of 5.37e-5.
  The artificial capture reproduced all 8 baseline VJPs; no radius passed its
  local selection rule. Both summaries replayed byte-for-byte without importing
  the extension or reading decoder weights.
- Parent archive manifest: `180c2c60b9dc5ee7c871fda336946af498324fec5199fa9235789025265a1201`.
  Original development extension: `1007bc32b7394142071a1d12ca1fdafa4311609839ece1f50f52e4ef358c50bc`.
- Frozen real protocol: `0f221f19642744303d5c96c339ab93dc3a11fdcd62471563a1fae71c7eef6bc2`.
  New raw observations: `f5428d8a6cce49ce5dfdba6a42542fb6b6da46b73409cc48a8208a08bbc2c7d2`.
  Replayed summary: `514fbd46d72323ab6eb8810b23dedebc60cf7d6550d586cc0711f35654bde81c`.
- Exact untrained R=4 adapter: `sha256:70b5ad5aed0966fbe99eb73fcf117c0a12f5314f943aeedc3a43384b62e92f27`.
  Its parent is `sha256:b95f692b12cb2d40a8a1eea9bee83c09c95cb97235ffa0b191fc0da628fb9c06`.
  Only the non-trainable log-radius buffer differs. For verification and
  reproduction, see [the frozen protocol and commands](README.md).

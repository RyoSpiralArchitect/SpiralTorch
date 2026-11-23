# Fixed channel rotation did not transfer across compound prompts

Changing the direction of a prefix gate at fixed delivered RMS moved the two verb logits,
but **the preregistered rotation passed neither the margin nor the decision criterion on
any of the three bases** (two were required). Within each template the correct two-verb
choices stayed at 8/16 before and after training for both teachers and all three maps.
There were no ties, wrong-to-right changes or right-to-wrong changes. Positive changes in a
mean signed margin are not improvements in classification.

## Where the rotation helped or hurt

Ranges below are over all three predeclared training orders. `h` is the head-teacher gain
in signed head margin; `S` compares head versus attractor supervision. The two seven-token
templates have the same length, token multiset and final token. Only the modifier's position
changes. These are new compound prompts, not unseen words. [All per-order values](orders.tsv)
include both teachers, both controls and the decision counts.

| Base | Modifier on | Head gain h, primary Q | Teacher selectivity S, primary Q | Comparison |
| --- | --- | ---: | ---: | --- |
| 557 | head | +0.01082 to +0.01175 | +0.00356 to +0.00388 | Identity has higher S in every order. |
| 557 | attractor | +0.00395 to +0.00410 | +0.00093 to +0.00110 | The inverse has higher S; identity has higher h. |
| 563 | head | +0.00018 to +0.00020 | +0.00082 to +0.00082 | Both absolute gains are below 0.001 despite beating both controls. |
| 563 | attractor | -0.00101 to -0.00100 | +0.00004 to +0.00004 | Inverse h is +0.00564 to +0.00572, but it also loses on the head template. |
| 569 | head | -0.00076 to -0.00075 | -0.00014 to -0.00014 | No route has positive mean head gain here. |
| 569 | attractor | -0.00193 to -0.00192 | -0.00045 to -0.00043 | Identity and inverse both beat Q. |

Numbers are rounded to five decimal places; the gate used unrounded values. On base 563
the largest fraction of an initial wrong-choice gap closed by the inverse on the
attractor-modified template was 14.7% (rounded). Its favorable slice is not a replacement primary result.
These data give no reason to make either rotation the default or to increase the radius.
A more informative next control would measure whether a direction estimated from **training
inputs only** beats the fixed rotations on a separately frozen split at the same parameter
budget and delivered RMS. That experiment was not run here.

## What actually ran

- The limited rehearsal used a separate base and examples: 12 synthetic updates and 72
  scoring forwards. The formal CPU run used 32 training prompts, 3 bases, 3 maps, 2 teachers,
  3 training orders and 32 updates per endpoint: 1,728 updates and 3,456 native gradient
  calculations. Both workers completed on their first attempt.
- All 576 identity updates and 18 identity endpoint artifacts matched the preceding study
  byte for byte. All 54 endpoints were sealed and audited before scoring. Nine origins plus
  54 endpoints each saw the same 32 new prompts once: 2,016 full-logit scoring forwards.
  These 32 prompts must not be treated as unused in another test.
- Independent verification covered 108 origin gradients and 6,912 native before/after
  residual rows. The maximum relative discrepancy from target RMS was 1.07e-6; the largest
  absolute discrepancy between the measured output RMS and the receipt was 1.14e-9.
  This tests delivery, not language quality.
- The local CPU build passed 1,366 Rust tests and five installed-wheel Python binding tests.
  CPython 3.8 and 3.12 independently replayed the journals into identical result JSON and TSV.
  No real GPT-2 calls, held-out natural-language prompts, GPU, network or publication were
  part of this experiment. No WASM run is claimed.

Source at execution: `555ba49e79ffcae3760d538aee9d79f7003c00e8`.
See the [recipe and replay instructions](README.md) and the [frozen protocol](PROTOCOL.md).

| SHA-256 identity | Value |
| --- | --- |
| Formal input manifest | `7e98f8143acddefe18bc6f3111d534631592cb97aa8761f33b8f65c9d61e8a00` |
| CPU wheel | `e2eebbd95f61d94c80fd9bd9d0fb4f34ea72db389bda818098c7f4fbf9178ea6` |
| Training manifest | `a353ec8146c663e8bc3d62b1770d0daff1196d012e6c08a197b3c07a13d18f36` |
| Raw evaluation logits | `f17abe883752f321717c2d3553e7f26210eaecef611576a55d1cb36c700be545` |
| Formal replay JSON | `0bbc30af848b76f0cb5b3bb473a9234df2065ec275d3661bbc03ce50a1d32d41` |
| All-order table | `d58e4f64ace61b20a14fa0225fe6aa8108584a392e9e0836746153e343f7b357` |

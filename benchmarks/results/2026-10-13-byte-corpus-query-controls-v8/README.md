# Contextual Geometry Mixtures: Byte-Corpus v8

This increment puts position-conditioned geometric mixtures into actual
document-held-out byte-LM learning. It extends the shared-router v7 screen
without changing the source data, order, seeds, units, gains or original arms.

**Learning comparison complete:** native Metal, browser WebGPU and independent
CPU-f32 Torch each completed all 75 cases and 9,600 updates. Both resident
engines pass the unchanged corpus numerical gates. Live pause/resume checks
for v8 remain pending; this does not claim the entire protocol is complete.
The earlier `checkpoint-validation.json` intentionally retains the historical
partial status from before the browser finished.

The [native](native-comparison.json) and [browser](browser-comparison.json)
qualifications both record maximum CE error `1.431e-6` and parameter error
`8.345e-7`. All twelve learned query-weight tensors move: smallest reference
delta L2 `4.912e-5`, largest relative delta error `3.022e-6` native and
`3.623e-6` browser, below the unchanged `0.002` gate. Every frozen query
bias/weight preserves its initial bits. All 51 inherited v7 case reports
[match exactly in each of the three engines](inheritance.json), including
signed-zero-sensitive float values.

**No quality advantage is established.** Maximum absolute contextual-versus-
shared final BPB difference is `6.450e-8` native, `8.600e-8` browser and
`8.600e-8` Torch; signs disagree across engines. All twelve contrasts per
engine are retained in [contrasts.json](contrasts.json). In the
[post-screen weight reconstructions](activity.json),
all learned routers vary by position, but their maximum probability departure
from their own shared router remains about `9.05e-5`. Nonzero parameter
updates and tiny score differences must not be presented as useful learning gains.

## Mechanism And Frozen Comparison

Each query selects weights for Poincare squared (P), origin-scaled Euclidean
chord squared (E), and categorical Fisher-Rao squared (F), using a projection
of causal-wave coordinates. The selected mixture contributes to causal QKV
attention scores. Gradients flow through both metric values and the query
selector, back into the wave, coordinate encoder and embeddings.
Because weights depend on the query, the resulting score construction is not
claimed to be a symmetric distance or a single new Riemannian metric.

Every P/E, P/F, E/F and P/E/F subset gains two arms: trainable query weights,
and the same weights frozen at zero. Their affine bias is always zero and
frozen; both still train one shared router/intercept and the coordinate path.
This avoids attributing a duplicated intercept update to contextual learning.

All 51 v7 arms remain, plus 24 new arms across seeds 11, 23 and 37. Each arm
uses 128 updates, 8,192 training target exposures and 2,048 held-out targets
at revisions 0/64/128. Width 16, hidden 32, two heads, one block, 32 byte
positions, batch 2, geometry width 4, curvature -0.75; Topos is off.
This remains a tiny fixed repository-prose corpus, not pretrained-LLM FT.

Two-/three-component shared mixtures have 11,294/11,296 parameter scalars.
Contextual versions have 11,314/11,326 stored scalars, including 4/6 frozen
bias scalars. Query weights add 16/24 trainable scalars in the learned arm.
Equal data and update counts are not equal parameter or computation budgets.
This is not a speed comparison with Torch.

Preparation runs in Rust without fitting, randomness, updates or held-out
access. Native and WASM preparation produced identical request bytes:
`fd40be093cee491f9cbd65b5ca4feefdbdb1d6a75dc7928d9ef913b981216da9`.
Every learner uses those bytes. Initial RMS still has to pass the inherited
training-window check on its actual execution device.

See the [predeclared protocol](pre-registered.md), the
[production recipe](../../../docs/byte_corpus_query_controls.md), and
[two independent source reviews](independent-review.md).

Source commit: `386966e747013b7c40c45c83f76746e94793ec02`.
Local validation: 44 Rust study tests and 91 Python corpus tests passed;
the three browser modules passed syntax checks. This does not assert formal CI.

## Evidence Boundary

The separate [contextual-model v1 browser screen](../2026-10-13-contextual-mixture-v1/README.md)
remains **failed** at its frozen per-step rounded-delta gate. That historical
failure and its one-ULP diagnostic are not overwritten by this study.
Here the corpus final-delta criteria are unchanged from v7. Completion,
numerical qualification, and improved held-out prediction are separate.

Raw requests, documents, weights, checkpoints, executables and full logs stay
local. Public projections omit weights and retain scalar learning/evaluation
trajectories. `summarize.py` reproduces those projections and comparisons
from the retained raw bundles; `activity.py` is a post-screen, CPU-Torch
reconstruction on declared training windows, not direct GPU activations or
a new qualification gate. Hashes identify artifacts, not execution attestation.

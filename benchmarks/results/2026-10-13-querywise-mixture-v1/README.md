# Query-Local Mixture Foundation And Router Sensitivity

The v7 corpus router moved but did not establish a quality advantage. This
follow-up examines that result without updating any model, then qualifies a
Rust-owned query-local mixture primitive on native Metal and browser WebGPU.
The byte model's existing shared router is unchanged. An input-conditioned
LM router is **not yet implemented or evaluated** by this experiment.

## What The Snapshot Probe Shows

The read-only CPU-f32 probe covers all twelve learned-mixture arms at initial
and retained native-update-128 weights, on all 128 declared training batches:
24 snapshots, 3,072 forwards. It neither reads held-out windows nor searches
learning rates. This is exploratory diagnosis, not a new efficacy experiment.

- Row-centered Poincare/Euclidean component cosine similarities range from
  approximately `0.99545` to `0.99699`: their patterns are very similar here.
- Within-batch query-gradient retention ranges from `0.1542` to `0.2460`.
  Summing the local gradient vectors before taking their norm is much smaller
  than summing their individual norms. The shared-parameter pullback is doing
  the mathematically correct sum; this is not a lost-gradient bug.
- Removing the geometric bias changes attention by mean total variation
  `0.07855` to `0.13930` per head/snapshot. Row-centered geometry/QK RMS ratios
  range from approximately `0.4893` to `1.0817`. The geometry is not simply absent
  from attention, but distribution shift alone is not useful language learning.
- Summed query-local gradients reproduce the shared-router gradient within
  `3.038e-9` absolute error in this CPU probe.

For each head, query retention is
`sum_batch ||sum_(sample,query) g||_2 / sum_(batch,sample,query) ||g||_2`.
`across_batch_retention` separately compares the sum over all batches with the
sum of batch-gradient norms. These ratios are not explained variance, causal
attribution, an estimate of recoverable quality, or independent quality samples.
All 24 rows, including every component pair and head, are in `sensitivity.json`.

## What Was Added And Verified

`querywise_mixture` accepts `[B,H,Q,S,K]` components and `[B,H,Q,K]` logits.
Unlike shared headwise routing, it preserves query identity in both softmax
weights and router derivatives. It reuses the checked wide-reduction kernels;
no Python runtime geometry implementation or host activation round trip is
introduced. See the [API and recipe](../../../docs/querywise_geometry_mixture.md).

The [frozen qualification protocol](protocol.md) retains 24 real LM-cotangent
fixtures plus three synthetic fixtures with distinct query logits and 2/3/8
components. Both runtimes pass all 27 cases against independent CPU-f32 Torch:
forward, weights, component/router pullbacks, a one-step SGD update of those
primitive tensors, and the resulting forward. This is not a full-LM update.

Both runtimes' largest forward error is `4.7684e-7`; maximum router-gradient
relative L2 error is `2.8450e-6`, below the unchanged `0.002` gate. All cases
exercise strided inputs, bit-exact repeated backward and retained tapes.
Guarded component, logit and seed operands reject all six tested three-parameter
updates atomically, including rate zero, without parameter mutation. Each
includes an independent healthy sibling, which separately passes a nonzero
update. Both component and router SGD deltas also pass a relative-L2 `0.002`
gate; frozen-update negative controls reject in every fixture.

Nine CPU contract tests, eight backend tests, two harness-checker tests and
seven snapshot-gate tests passed, with native GPU tests explicitly enabled.
Independent source reviews found no remaining actionable P1/P2 findings after
repairing three measurement gaps; their scope excludes runtime attestation.
Formal CI, input-conditioned LM learning, general quality and speed advantages
are not established. Neither the mixture primitive nor arbitrary router logits
provide a causal mask: the attention and logit producer must enforce causality.

The original validation-v1 reports passed weaker absolute-update/guard checks
and are retained as `native-original-v1.json` and `browser-original-v1.json`.
They are superseded by remeasured validation-v2 reports, not reinterpreted as
having passed the stronger checks. The native snapshot probe now rejects
partial, browser and foreign-engine reports; its reviewed rerun is byte-identical
to the original scalar diagnostics and cotangent fixture.

## Next Connection

A causal contextual projection can produce query logits, starting from zero
projection weights to retain the existing initial mixed function. Its VJP must
join the coordinate encoder and the same full-model parameter owner, with
prefix-causality and checkpoint/resume tests before a matched corpus study.
The current evidence motivates that controlled comparison; it does not predict
that query-local routing will win.

Raw tensors, model weights, fixtures and executables remain local. Published
JSON contains scalar diagnostics/checks only. `validation.json` and `SHA256SUMS`
identify source and retained artifacts; they are not cryptographic execution
attestations. See the recipe above to regenerate the fixtures from the exact
v7 source request and native report.

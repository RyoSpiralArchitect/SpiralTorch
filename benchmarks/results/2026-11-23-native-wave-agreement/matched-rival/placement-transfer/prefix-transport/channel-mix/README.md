# A fixed rotation after the prefix gate

The [completed comparison](RESULTS.md) found changes in verb margins, but no
decision flips or advantage across bases; the mixing recipe stays opt-in.
The subsequent [train-only route comparison](selection/RESULTS.md) tested
whether choosing among these same checkpoints changes that conclusion.

For temporal transport, m[t] is the average of earlier site features after
rotating each by its lag. The original adapter returns
`x[t] + strength * (gate * m[t] + bias)`. The optional v2 recipe instead
returns `x[t] + strength * Q(gate * m[t] + bias)`. This changes which output
channels receive each gated input. It is not a permutation of parameter
names. No extra weights, learned direction or cache slot is introduced.

For each nonoverlapping triple of channels, Q is
`[[15,-6,10],[10,15,-6],[-6,10,15]] / 19`; the inverse uses its transpose.
Leftover channels are not mixed. Widths below three are rejected for v2.
The common channel mean and Euclidean length are preserved in real arithmetic.
The residual and its stored output still round to f32. The input and parameter
backward paths apply the transpose; the raw upstream trace therefore need not
equal the bias gradient divided by strength. The existing per-channel calibration
continues to measure squared delivered length, but its certificate binds the
selected mix and cannot be reused on a different recipe.

Use the same placement, tensor names, shapes and `requires_grad` masks as
for a v1 prefix adapter. Rust callers select
`GeometryAdapterRecipe::MixedPrefixTransport`; clients using serialized
artifacts set the recipe explicitly, for example:

```json
{"schema":"spiraltorch.prefix_transport_adapter.v2","features":3,
 "strength":0.25,"max_values":64,"rotation":"quarter_turn",
 "output_mix":"cayley_triples"}
```

The inverse value is `inverse_cayley_triples`. Omit neither the version nor the
mix. Old v1 artifacts retain their original serialization, identity and
arithmetic. Once attached, training and full/cached inference use the same
ordinary model APIs; no Python-side rotation is needed.

## Reproduce and audit

The [protocol](PROTOCOL.md) fixes the templates, masks, time budget, comparisons and gates.
Use isolated CPU mode, single-threaded BLAS/Rayon, and the bounded launcher
from [the earlier study](../../../parameter-vjp/bounded.py). Do not run a second
attempt in a partly scored directory. Prepare both disjoint input lists with
`design.py KIND NEW_ROOT RESIDUAL_PARENT_ROOT` before loading a model.
`worker.py NEW_ROOT PREFIX_PARENT_ROOT RESIDUAL_PARENT_ROOT SIGNAL_ROOT KIND INPUTS_SHA EXTENSION_SHA`
takes only the training manifest into its training loop. All endpoints, native
residual witnesses and the historical control are checked **before** the
evaluation file is opened. The scored marker is durable even on partial failure.
For an existing journal run only `verify.py SPLIT_DIR PREFIX_PARENT_SPLIT
RESIDUAL_PARENT_SPLIT SIGNAL_SPLIT INPUTS_SHA NEW_JSON NEW_TSV` with the
standard library; this does not load a model.

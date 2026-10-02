# Frozen Context Interventions

The [gated study](elliptic_gated_study.md) found small mean improvements for
elliptic features, but all learned geometric gates finished negative and the
ordinary tangent control still won. Negative `g` makes the feature residual
`(1+abs(g))*local - abs(g)*context`. We therefore need to separate local gain,
constant centering, generic causal aggregation and content-dependent attention.

This experiment changes only the context term at inference. It does **not**
retrain, select a new checkpoint, alter the original study or establish which
alternative would learn better. All six trained gated checkpoints, every seed
and both original endpoint sets are retained. No speed claim is made.

## Fixed Conditions

For `f = phi(orientation)` and the saved scalar `g = tanh(raw_mix)`:

| Mode | Features sent to the unchanged readout | Interpretation |
| --- | --- | --- |
| native | original adapter, unchanged | exact replay gate |
| local | `f` | remove the entire learned correction |
| gain_only | `(1-g)*f` | keep local gain, remove contextual/constant term |
| anchor | `(1-g)*f + g*phi(1,0,0)` | fixed chart-anchor correction, no other tokens |
| prefix_mean | `(1-g)*f + g*mean(f[0:t+1])` | causal aggregation without attention scores |

The same modes apply to the ordinary tangent control. Rust owns the nonlinear
map and native attention; replacement formulas live only in this evaluation
example. The native condition calls the original implementation rather than a
reconstruction. Prefix means include the current token, never future tokens or
another batch member. They accumulate in f64 and round the context to f32 before
the blend, matching the native context output boundary.

All six native evaluations must reproduce the original per-block losses exactly
before any intervention starts. Saved gates, projections, readouts and adapter
metadata cannot change. A full base-weight digest verifies the restored model.
Atomic, hash-sealed progress allows interrupted evaluation to continue, and a
completed resume does not rerun conditions. Corrupted or unbound data fails closed.

## Run Offline

Use the original study's frozen native/client package and source helpers. The
example checks their hashes, the completed parent plan/results/checkpoints, model,
tokenized split and all evaluation blocks. Use a separate output directory:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
python bindings/st-py/examples/hf_elliptic_context_ablation.py \
  --parent-study-dir /path/to/completed-gated-study \
  --model-dir /path/to/gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e \
  --corpus /path/to/pride_and_prejudice.txt \
  --transfer-corpus /path/to/alices_adventure_in_wonderland.txt \
  --output-dir /path/to/new-ablation
```

Use `--resume` only with identical inputs and sources. The program has no training
mode and the intervention adapters reject gradient-enabled or training calls.
Publish numeric block losses, paired deltas, hashes and verification; keep model,
corpora and checkpoints local. Reused endpoints and three seeds are exploratory.
An inference intervention can expose the saved model's dependence on a term, but
its distribution shift does not establish the causal reason for a training gain.

# Directional Geometry With Ordinary Nonlinear Controls

The completed WaveGate study showed learning and exact continuation, but its
linear control won. This next experiment uses the existing public
`EllipticResidualAdapter`, whose learned coordinates feed the Rust elliptic/Lie
map and batched VJP. It does not introduce a Python replacement geometric map.

## Paired Arms

All three active arms use the same learned `F -> 2` affine projection and
zero-initialized `9 -> F` readout. For `F=768`, each has **8450 parameters**.
Within each seed, the initial parameter tensors and training batches are equal.
All residual outputs initially equal the frozen model exactly. Later updates
reach both projections, including the input projection through Rust's VJP.

Let `a` be the nine Rust features at chart orientation `(1,0,0)` and `J` their
Rust-derived Jacobian with respect to the last two coordinates. For learned
coordinates `u`, the arms are:

- `tangent`: `a + J u`.
- `tanh_control`: `a + tanh(J u)`, componentwise ordinary nonlinearity.
- `elliptic`: the full Rust feature map at `(1,u[0],u[1])`.

These share the anchor and local Jacobian, not feature distributions, effective
rank, global expressivity or compute cost. The tanh arm is a fixed nonlinear
feature control, **not an unconstrained learned MLP**. The geometric arm uses a
local hemisphere chart, not a global atlas, and remains tokenwise rather than
mixing tokens. It studies learned directions, not yet token-to-token relations.

## Fixed Budget And Evaluation

`bindings/st-py/examples/hf_elliptic_pride_nonlinear.json` fixes the experiment
before execution: cached GPT-2, frozen float32 CPU base, explicit placement after
`transformer.h.0.mlp`, seeds 41/43/47, 512 Adam updates per arm, batch two,
128-token blocks, lr 0.001 and residual strength 0.1. Each run consumes 130048
causal targets. No model/corpus download or native rebuild is necessary.

The same Pride training partition, 16-block descriptive development probe,
120-block endpoint partition and 32-block Alice set are reused. These endpoints
have already been inspected during WaveGate research: this is an **exploratory
reused benchmark**, not fresh confirmation on previously unseen data. Neither
book is claimed absent from GPT-2 pretraining. No best-checkpoint selection,
early stopping or endpoint-driven parameter change is permitted in this run.

The preceding WaveGate arms used only 1536/1537 parameters. Equal update budgets
do not make cross-family quality differences parameter-matched evidence.
The primary comparisons here are the three equally parameterized new arms.
Pure speed competition remains restricted to mathematically equivalent paths.

## Reused Execution Loop

The entry point is `bindings/st-py/examples/hf_elliptic_nonlinear_study.py`.
It supplies a seed-aware adapter factory to the existing long-horizon loop,
rather than copying its checkpoint, optimizer, cursor or evaluation logic.
The protocol binds the factory source, elliptic bridge, shared driver/helper,
native binary, configuration, model, tokens and batch schedules. Factory
construction explicitly uses CPU, restores its RNG, and does not seed accelerator
generators or change the caller's default device. The same initial-parameter hash is
stored in checkpoints and rechecked on resume before loading their learned state.

Each saved endpoint must preserve frozen base weights and reproduce its next
adapter/Adam update exactly. Evaluation opens only after every planned run
finishes. Tests interrupt the tanh arm after an earlier arm completes, then
compare the entire resumed state/history with an uninterrupted tiny HF run.

Run with the same offline environment and local data variables described in the
[long-horizon guide](wave_gate_long_horizon.md):

```bash
python bindings/st-py/examples/hf_elliptic_nonlinear_study.py --config bindings/st-py/examples/hf_elliptic_pride_nonlinear.json --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

Freeze these source files and the native runtime for a run. A later checkout or
code change is not a compatible resume; use the same frozen copies and `--resume`
only after confirming the original process ended. The existing summary tool
handles this protocol's schema and explicit comparison limitations while keeping
the previous WaveGate public summary byte-for-byte reproducible.

## Completed Outcome

The [complete nine-run record](../benchmarks/results/2026-10-02-elliptic-nonlinear-learning/README.md)
contains all 4608 primary updates, exact adapter/Adam continuation checks and
fixed-endpoint per-block losses. Every active arm improves the frozen baseline.
Elliptic beats the tanh control in every seed on both sets, but tangent wins all
paired comparisons. Mean elliptic-minus-tangent CE is +0.150571 on Pride and
+0.085750 on Alice; elliptic-minus-tanh is -0.037138 and -0.011554.

Thus the directional Rust map learns in a real HF loss path, but it does not yet
justify preferring geometry over the simpler linear adapter. Local differential
matching did not equalize the maps' global optimization behavior. Token-to-token
geometry remains a distinct, untested mechanism rather than an inference from
this tokenwise result. These reused endpoints must not become a tuning oracle
while being described as fresh confirmation.

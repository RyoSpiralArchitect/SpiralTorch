# Elliptic Chart-Direction Steps

This explicit optimizer operation follows the
[training-only conditioning probe](elliptic_directional_learning.md). It does not
change the geometric forward map, VJP, JVP or production adapter backward rule.

## Rust Contract

`EllipticLearningBatch::chart_step(proposal, relative_damping)` uses columns 1
and 2 of the saved native Jacobian, holding orientation coordinate 0 fixed:

```text
G = mean_rows(J_chart^T J_chart)              # 2 by 2
M = G / (trace(G) / 2) + relative_damping * I
raw_step = solve(M, proposal)                # proposal has shape [2, C]
step = raw_step * norm(proposal) / norm(raw_step)
```

The global Frobenius/L2 norm is preserved up to f32 rounding, not each column's
norm. The positive determinant cancels in the normalization, so Rust uses the
2 by 2 adjugate without an unstable inverse division. Accumulation and norm
calculation use f64. Relative damping must be finite in `[1e-6, 1]`; proposals
must have two nonempty finite rows and at most 131072 values. Empty snapshots,
zero-trace metrics and unrepresentable outputs fail. A zero proposal remains
exactly zero and its undefined cosine is `None` / `undefined`.

The owned `EllipticChartStep` contains values, mean metric, damped condition,
input/output norms and direction cosine. Python exposes the same method and
properties. WASM exposes `chartStep`, `dampedCondition`, `proposalL2`, `stepL2`
and the corresponding arrays. Clients do not reconstruct the geometric map or
its metric.

This is a **separable mean chart metric**: it ignores hidden-input covariance,
the readout and language-model loss curvature. It is not the full parameter
pullback, Fisher information or Hessian. Averaging may hide token-specific
anisotropy; whether this correction helps training is an experimental question.

## Matched Training

The executable `bindings/st-py/examples/hf_elliptic_chart_step_study.py` compares
ordinary anchored tangent and native anchored elliptic, each with ordinary Adam
and with chart-corrected Adam proposals. Each arm has 8451 trainable parameters.
All four start from identical projection/readout/raw-gate tensors within a seed,
with the zero readout preserving the frozen model initially.

Adam always consumes the true loss gradient and maintains its normal moments.
The chart arms correct only the orientation weight/bias proposal, packed as
`[2, features + 1]`. Gate and readout updates remain ordinary Adam. Each corrected
proposal keeps the norm of that arm's current Adam proposal. This does **not**
equate cumulative step lengths after trajectories diverge. Both native and
actually applied f32 displacement norms are recorded.

The read-only study validator requires the applied displacement norm to match
the proposal within `2e-5` relative tolerance, with no absolute floor. A vanished
nonzero update therefore fails rather than being reported as budget-preserving.
It also checks positive semidefiniteness of the symmetric mean Gram matrix
(normalized determinant tolerance `64 * ulp(1.0)` for f64 accumulation), then
checks the damped condition against that matrix and the configured damping
within `1e-6` relative tolerance, accounting for the native f32 damping boundary.
These are receipt consistency checks, not a new geometric execution rule.

The affine control uses the native Jacobian at its constant chart anchor.
For both maps the anchored gate multiplies the local Jacobian by a shared scalar;
that scalar cancels in relative damping and step normalization when nonzero.
The operation therefore uses the bare local metric, not a newly invented
gate-gradient rule. It does not add token mixing or new parameters.

The experimental optimizer requires exactly one fresh gradient-enabled forward
per step, rejects closures/gradient accumulation and currently requires CPU f32.
It stores the recipe in checkpoints and checks it before restoring Adam. Native
rejection after proposal creation rolls back all parameters and Adam state.
Production optimizer defaults are unchanged. The long-horizon driver accepts an
explicit optimizer factory for the same initialization and exact-resume checks.

The fixed protocol is `hf_elliptic_pride_chart_step.json`: three seeds, four arms,
512 updates each, identical batches, Adam learning rate 0.001, relative damping
0.1. No hyperparameter or checkpoint is selected from development scores. All
training and next-update continuation checks must finish before endpoint scoring.
Pride/Alice are previously inspected exploratory endpoints, not untouched tests.
Extra geometry computation and host transport are not a speed comparison.

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 \
  python bindings/st-py/examples/hf_elliptic_chart_step_study.py \
  --config bindings/st-py/examples/hf_elliptic_pride_chart_step.json \
  --model-dir "$LOCAL_GPT2" --corpus "$LOCAL_PRIDE" \
  --transfer-corpus "$LOCAL_ALICE" --output-dir "$NEW_STUDY_DIRECTORY"
```

Resume with the same frozen code/native package and `--resume`. The practical
reference remains ordinary Adam with the tangent map. Primary diagnostic
contrast is chart elliptic minus Adam elliptic; also report the effect on the
tangent control, both geometry gaps and the paired interaction. Keep all losing
seeds. Successful numerical/continuation tests do not establish LM improvement.

## Completed Comparison

The [frozen three-seed study](../benchmarks/results/2026-10-03-elliptic-chart-step-study/README.md)
completed all twelve 512-update runs with finite saved states and exact next
updates after restore. Chart elliptic minus Adam elliptic gives +0.001270 mean
Pride CE and -0.003290 Alice CE, with only one improving seed on each set. The
ordinary tangent map remains better in every seed under either optimizer.

Mean proposal-direction cosine of 0.531-0.561 confirms the elliptic correction
really changes learning updates. Its limited/mixed effect does not establish
that directional chart conditioning causes or resolves the remaining quality
gap. The mean-metric approximation stays an explicit experimental operation;
ordinary Adam remains the default. This result neither rejects geometry in
general nor supports promoting this particular correction as a quality win.

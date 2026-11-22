# A one-proposal geometry update for declared choices

The [training-log comparison](../prequential/RESULTS.md) separated better
within-pair ranking from retaining the pair's total vocabulary probability.
Commit `d3b069c457bd` implements that distinction in the tensor gradient,
native decoder and Python binding. See the [API example](../../../../../bindings/st-py/examples/declared_set_geometry.md).

## Fixed artificial cases

The decoder starts with uniform logits; only a geometry bias is trainable.
Gold and rival head rows are zero and `(1,-1,0,0)`. Each outside head row
can vary along that axis and a perpendicular axis. The score check computes
probabilities independently from the actual output logits.

| Case | Trainable L2 step | Initial P(set) | Within-only proposal | Returned P(set) |
| --- | ---: | ---: | ---: | ---: |
| 3 classes, set can be protected | 1/65536 | .666667 | .666288 | **.667628**, repaired |
| 3 classes, gradients exactly opposed | 1/65536 | .666667 | .666288 | No feasible direction; original remains .666667 |
| 4 classes, two outside competitors | 1/65536 | .500000 | .500390 | **.500390**, accepted |
| Same 4-class model and input | 1/128 | .500000 | .003200 | **.500000**, measured and rejected |

In the first row P(gold) rose from .333333 to .333712 even under the leaking
proposal. The ordinary full-CE plus within-set first-order guard therefore
passed it unchanged. With the direct set-coverage gradient instead, a repair
at the same distance raised P(gold) to .334164 **and** raised P(set).

In the last row the **actual f32 parameter displacement** had a negative
dot product with both initial gradients (-.873464 within and -.436732
coverage); the first-order guard passed it. In a fresh forward on that
candidate, one outside class dominated. The finite check returned the
original model, not the candidate and not a smaller retried step.

These synthetic examples distinguish repaired, infeasible, accepted, and
finite-rejected behavior. They do **not** establish a GPT-2 quality gain or a
good radius. This run loaded no pretrained model and no evaluation prompts.

## Reproduce

With a local CPU wheel containing the source commit installed, run
`python -IB synthetic_step.py /path/to/new-report.json` from this directory.
It uses only in-memory artificial weights. The guarded Rust operation uses
one shared training forward for its two VJPs and at most one extra scoring
forward. Proposal errors are not retried. Both individual handles and the
forward-only scores are independently usable.

Validation: 21 Rust tensor classification tests, 300 native causal-LM tests
and 70 installed Python causal-LM tests. The first Python check of the
lower-level primitives had a wrong test expectation for a pass-through guard
(`mix` is zero, not null); the original failure was kept. The criterion and
native implementation were not changed to make it pass.

Private CPU wheel SHA-256: `06a0aad1d09a00c3694ee851d53867f87a2750d08ad3a1426a7b107ef3fb5fc7`.
Its extension and the installed bytes both hash to
`ac7120c27d914423c4496fefb2d2adf0110ea0d904f7e0f5692a9842c06a32e7`.
Four-case JSON SHA-256: `cd05f97a13047a326721b759ee07d14644c230d1cbb971d886b1beb817df53eb`.
This development wheel is neither a release candidate nor a replacement
for the separately pinned, unrun real-model radius comparison.

# A fixed-length update from several sealed geometry gradients

Use `step_geometry_cohort_gradient_at_distance` when the experiment calls for
an equal vote per objective and a fixed Euclidean distance in trainable
parameter coordinates. All handles must belong to the same model snapshot
and readout policy. A repeated handle votes again; target counts are metadata.

```python
next_model, r = model.step_geometry_cohort_gradient_at_distance(
    [first_geometry_gradient, second_geometry_gradient], 1 / 256
)
print(r["step"]["gradient_l2"], r["step"]["realized_l2"], r["member_loss_dots"])
```

Unlike `step_geometry_cohort_sgd(..., target_l2=...)`, this does not rescale
an already rounded SGD proposal. Rust computes each coordinate of the mean
with compensated f64 summation, normalizes it in f64, then rounds the new
weights to f32. For example, `[1, 2^-149]` and `[-1, 0]` have a nonzero
mean that would round to the all-zero f32 vector. True cancellation rejects
a positive distance; zero distance is an identity after input validation.
The existing f32 representability threshold (1e-4 relative vector and length
error) still applies. This is opt-in: a tiny mean can reflect noise.

The member dots use the *stored* displacement. Negative predicts a local
loss decrease for that member; it is not a measured post-update loss.
Average descent need not be descent for all members. The Rust group
common-descent API remains separate.

## Synthetic check

Each of two synthetic WaveGate models (R=1 and R=4) took two consecutive
updates of length 1/256, with two freshly differentiated prefixes per step.
The first pair required opposite verbs; the next pair, at the updated
weights, required the same verb. Do not treat this sequence as a
matched-state comparison of label choice.

| Radius | Pair | Norm of mean / mean member norm | Signs of member loss dots |
| --- | --- | ---: | --- |
| 1 | Opposite | 0.17456 | negative, positive |
| 1 | Same | 0.99976 | negative, negative |
| 4 | Opposite | 0.17487 | negative, positive |
| 4 | Same | 0.99976 | negative, negative |

All 128 stored coordinates (32 gate/bias entries per step) matched an
independent Python `math.fsum` and f32-rounding calculation byte for byte.
Frozen state, identity, the mean loss and each member dot were also checked.
The installed binding passed 65 whole-decoder tests; 305 selected Rust tests
passed. For the pre-existing single-gradient API, 8 toy updates and 16 toy
predictions reproduced all 19 historical numerical artifacts byte for byte.
No pretrained model or evaluation split was used in the new cohort run.
This validates the update contract, not language quality or a best radius.

Reproduce from this directory with a CPU-only development wheel:

```sh
"$VENV/bin/python" -I -B check_cohort_ray.py "$PINNED_TOY/plan.json" "$NEW_OUTPUT" "$EXTENSION_SHA256"
```

Only toy plan `e7a486aadc12503cbaa6828aa9ced4ea575900c51de470a1f0abf9c1ae2ab636`
is accepted. The inspected CPU wheel SHA-256 is
`4d03cd564f2417d00dcc78f439847048af9752787b448e61de5cfd4f22f22fb6`; its installed
extension SHA-256 is `22cb7b91ff88212e844895cdf34704ca6b7e0a783e726dfd7e37766620346687`.
This development wheel does not replace the one pinned to the separate
GPT-2 radius-training protocol.

# Matching a geometry step by its residual effect

A final-MLP prefix-transport gate acts on the running mean of the MLP output.
Equal-sized gate vectors need not move the residual stream equally. This
training-only API measures what a gate displacement delivers on a chosen set of
training prompts. It does not change token scoring, the adapter file format or
the inference path.

```python
# model: NativeCausalLm with a single trainable prefix-transport gate
# at the last MLP. The base, bias and any numerical LoRA are frozen.
metric = model.calibrate_prefix_residual_metric(train_prefixes)
gradients = [
    model.geometry_token_menu_gradients(ids, [7, 8], gold_index=label, mix=1.0)
    for ids, label in batch
]
model, receipt = model.step_prefix_residual_rms(
    metric, gradients, "diagonal", target_rms=0.001, max_parameter_l2=16.0
)
print(metric.report()["second_moments"], receipt["realized_rms"])
```

Rust uses the same two methods on `NativeCausalLm`, with `PrefixMetricRule`
rather than string choices. Each prompt contributes one last-position row with
equal weight. No target label or vocabulary projection is needed to calibrate.
The metric is reusable as only the gate moves; gradient handles are not reusable
after a step. The native implementation rejects an unsupported placement,
changed owner, frozen-only direction, unrepresentable displacement, or a step
above the caller's cap. It does not silently clip or fall back.

`uniform` matches a raw gradient's residual length. `diagonal` corrects for
each channel's training-set second moment; `cyclic` deliberately puts the
same correction on the wrong channels as a control. Matching is for the
pre-addition residual branch on those inputs, not the logits, probabilities,
held-out rows or the physical parameter distance. For a calibrated metric
`report()` returns a description, not a mutable or importable optimizer state.

See the [frozen comparison](PROTOCOL.md) and its [results](RESULTS.md).

## Reproduce or audit

Use an isolated CPU wheel built from this source, and the verified private
output directories of [prefix training](../) and the [signal study](../prefix-signal/).
Each input directory has separate `rehearsal/` and `formal/` children.
`run_exact.py` accepts the split, a private output root, the two input roots,
and the SHA-256 of the installed extension. Run rehearsal first; stop on
failure. The bounded launcher requires 4 GiB plus 128 MiB free before launch,
stops below 4 GiB, and refuses to reuse a receipt directory.

```sh
# WORKER_PY must import only the intended locally verified wheel.
# CAMPAIGN points to this directory's run_exact.py.
export SPIRALTORCH_BACKEND=cpu RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python3 -ISB "$BOUNDED" "$OUT/rehearsal-cpu" 120 "$WORKER_PY" -IB "$CAMPAIGN" rehearsal "$OUT" "$PARENT" "$SIGNALS" "$EXT_SHA256"
python3 -ISB "$BOUNDED" "$OUT/formal-cpu" 300 "$WORKER_PY" -IB "$CAMPAIGN" formal "$OUT" "$PARENT" "$SIGNALS" "$EXT_SHA256"
```

The sibling `verify_exact.py` needs only the standard library to re-read all step,
weight, native residual and logit records. Its five arguments are the split
output directory, the corresponding prefix-training and signal split directories,
and new JSON and TSV paths. It never reruns a model. The historical
`campaign.py` and `verify.py` remain hash-pinned: the latter rounded two
coordinates differently. Do not train again to recover an already saved run.

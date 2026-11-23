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

See the [frozen comparison](PROTOCOL.md) for the controlled study.

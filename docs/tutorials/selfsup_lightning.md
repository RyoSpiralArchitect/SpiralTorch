# Self-supervised Lightning Stages

The `SelfSupStage` bridge ties the self-supervised objectives exposed by
`spiral-selfsup` into the standard `ModuleTrainer` loop. Each stage bundles
self-supervised batches, the lightning configuration, and the objective
configuration so it can be scheduled alongside conventional stages.

## Running the InfoNCE example

A minimal InfoNCE training workflow is available as an executable example:

```bash
just selfsup-info-nce
```

The recipe compiles and runs `crates/st-nn/examples/selfsup_info_nce.rs`. It
creates a lightweight encoder, synthesises anchor/positive pairs, and executes a
self-supervised lightning plan. The program prints the mean InfoNCE loss per
stage epoch and publishes telemetry into the Atlas stream for downstream
consumers.

> **Note:** The example (and the `SelfSupStage` APIs) live behind the
> `st-nn` crate's `selfsup` feature. When invoking the example manually, enable
> the feature: `cargo run -p st-nn --example selfsup_info_nce --features
> selfsup`.

## Using `SelfSupStage` manually

1. Build a `SelfSupBatch` for each anchor/positive pair and bundle them into a
   `SelfSupEpoch`.
2. Instantiate a `SelfSupStage` with a `LightningConfig` and a
   `SelfSupObjective` (currently InfoNCE via `InfoNCEConfig`).
3. Register the stage with `SpiralLightning::fit_selfsup_plan` alongside the
   module you want to train. This method is also gated behind the `selfsup`
   feature.

The plan API reuses the full roundtable schedule machinery, so autopilot,
telemetry, and softlogic feedback remain active during self-supervised runs.

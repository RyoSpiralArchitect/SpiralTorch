# Self-Supervised Model Exports

This guide explains how to convert checkpoints produced by the `spiral-selfsup`
crate into the `st-model-hub` format and how to reuse the exported weights for
downstream experiments within `st-vision` and `st-nn`.

## Exporting checkpoints

The `tools/export_selfsup.py` utility ingests a JSON checkpoint containing the
encoder, projector, and optional linear probe head learned during
self-supervised pre-training.  Run the exporter with the desired variant and
output directory:

```bash
python tools/export_selfsup.py \
  path/to/checkpoint.json \
  artefacts/spiral-selfsup-resnet50 \
  --variant resnet50-contrastive \
  --objective info_nce
```

Additional flags allow you to document downstream defaults such as the number
of classes, recommended linear-probe learning rate, and batch size.  When the
command succeeds it produces:

- `manifest.json`: top-level metadata including the variant, objective,
  compatible downstream tasks, and suggested fine-tuning hyperparameters.
- `encoder.json`: a `ModuleSnapshot` containing the encoder weight matrix.
- `projector.json`: the projection head stored with the same snapshot schema.
- `linear_head.json` (optional): an initialised linear probe compatible with
  `st-nn::layers::linear::Linear`.

Each snapshot follows the `st-nn::io::load_json` schema so modules can restore
their parameters without manual tensor wiring.

## Directory layout

```
spiral-selfsup-resnet50/
├── manifest.json
├── encoder.json
├── projector.json
└── linear_head.json
```

Every matrix is exported row-major and includes explicit shape metadata, making
it straightforward to materialise as `st_tensor::Tensor` instances in Rust or
Python.  The manifest’s `downstream.compatible` array lists the model families
that understand the artefact, starting with `st-vision/classification`,
`st-nn/linear-probe`, and `st-nn/fine-tune`.

## Downstream usage

The Rust example in `examples/fine_tune_with_selfsup.rs` demonstrates how to:

1. Load the manifest and tensor snapshots.
2. Materialise the encoder and projector as `Tensor`s.
3. Restore the linear probe via `st_nn::io::load_json` when present.
4. Compare the exported probe against a randomly initialised head trained with
   Euclidean updates.

Invoke the example with the artefact directory:

```bash
cargo run --example fine_tune_with_selfsup -- artefacts/spiral-selfsup-resnet50
```

The script prints the pre-trained probe accuracy alongside a scratch baseline,
illustrating how quickly downstream heads benefit from the self-supervised
initialisation.

## Recommended hyperparameters

The exporter embeds fine-tuning defaults in `manifest.json`.  The provided
values match the settings used during internal benchmarks:

| Setting            | Default | Notes                                         |
| ------------------ | ------- | --------------------------------------------- |
| `batch_size`       | 256     | Scale linearly with available accelerator RAM |
| `learning_rate`    | 0.005   | Suitable for linear probes on ImageNet-1k     |
| `weight_decay`     | 5e-4    | Works well for both probes and full fine-tune |
| `warmup_epochs`    | 5       | Helps stabilise large-batch linear probing    |
| `total_epochs`     | 90      | Mirrors ImageNet schedules for comparability  |

Adjust these recommendations based on dataset scale—larger target corpora
benefit from proportionally longer schedules while smaller tasks often converge
sooner.

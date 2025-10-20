# Hyper-Parameter Search with `spiral-train`

This tutorial walks through the new `spiral-train` command-line utility that ships with the
SpiralTorch Python bindings. It demonstrates how to define a search space, plug in the Rust
strategies implemented in the `spiral-hpo` crate, and guarantee deterministic resume behaviour
with resource-aware scheduling.

## 1. Preparing a configuration file

Create a configuration file in JSON or YAML format. The file describes four components:

- `space`: a list of parameter definitions (`float`, `int`, or `categorical`).
- `strategy`: which search strategy to use (`bayesian` or `population`) and any
  strategy-specific knobs (e.g., exploration rate or population size).
- `objective`: Python callable that receives the suggestion dictionary and returns a metric.
- `resource` (optional): resource scheduling hints (maximum concurrency and minimum interval).

```yaml
space:
  - { name: lr, type: float, low: 0.0001, high: 0.1 }
  - { name: layers, type: int, low: 2, high: 6 }
  - { name: activation, type: categorical, choices: [relu, gelu, tanh] }
strategy:
  name: bayesian
  exploration: 0.2
  seed: 42
objective:
  callable: examples.objectives:train_and_validate
  maximize: false
resource:
  max_concurrent: 1
  min_interval_ms: 0
max_trials: 12
```

The objective callable may return any numeric metric. Set `maximize: true` if the metric should be
maximised; the CLI converts it internally so that the Rust strategies always minimise.

## 2. Running the search loop

From the repository root (or any environment where `spiraltorch` is installed), run:

```bash
spiral-train search --config configs/hpo.yml --checkpoint runs/checkpoint.json --output runs/best.json
```

The CLI will:

1. Instantiate a `spiraltorch.hpo.SearchLoop` via the new PyO3 bindings.
2. Request trial suggestions from the Rust strategy and feed them to your Python objective.
3. Persist checkpoints after each observation, enabling deterministic resume.

To resume, pass `--resume runs/checkpoint.json`. The search loop state and strategy RNG position are
restored from the checkpoint exported by the Rust crate, so subsequent suggestions match the
pre-resume execution exactly.

## 3. Resource scheduling

The `resource` stanza propagates to the Rust scheduler, enforcing concurrency limits and optional
cool-down intervals. If `max_concurrent` suggestions are already active, `spiraltorch` raises an
error instead of over-subscribing the device pool. This behaviour is covered by integration tests
that simulate constrained resources and verify that resumed runs continue to respect the limits.

## 4. Tracking integrations

The CLI can emit events to experiment trackers using the connectors located under `tools/tracking/`.
Enable them with repeated `--tracker` flags:

```bash
spiral-train search --config configs/hpo.yml --tracker console --tracker mlflow:experiment=spiral
```

- `console` prints concise updates via the standard logger.
- `mlflow` attaches parameter and metric logs to the current MLflow run (if the `mlflow`
  package is installed).
- `wandb` reports to Weights & Biases with optional `project`, `entity`, and `tags` arguments.

All connectors are optional; if the dependency is missing the CLI continues without failing.

## 5. Integration tests

The repository now includes end-to-end tests under `bindings/st-py/tests/` that exercise the CLI in
three phases:

1. Run a short search and capture the checkpoint after a subset of trials.
2. Resume from the checkpoint and confirm that future suggestions match the deterministic baseline.
3. Configure a low `max_concurrent` value and assert that the CLI surfaces the Rust scheduler error
   when it would oversubscribe resources.

These tests ensure that the CLI, the PyO3 bindings, and the `spiral-hpo` strategies stay in sync,
providing confidence that future changes preserve resumability and resource-awareness.

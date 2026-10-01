# Resident Vision Z-Space Control

The resident vision trainer accepts the existing Rust-validated Z-space
parameter control. Rust, Python and WASM use the same `st-core` report
projection and `st-nn` replay planner. No policy arithmetic is recreated in a
Python callback or JavaScript training loop.

This first connection applies an **absolute learning-rate multiplier** to
plain resident SGD. It is not resident Adam, geometric parameter updates or
an automatic closed-loop optimizer. The meta-optimizer producer retains its
own latent state; a trainer checkpoint does not contain that producer state.

An optional [resident loss-feedback gate](resident_vision_feedback.md) now
consumes accepted-step loss in Rust and gates the next proposal toward identity.
Its own history is checkpointed; this does not migrate the proposal producer's
latent state or geometric updates.

## Apply A Control

Rust accepts `apply_zspace_parameter_control(&control)` or
`apply_zspace_meta_optimizer_report(&report)`. Both bindings send the complete
report through Rust's bounded JSON entry and receive a serialized receipt:

```python
import json
import spiraltorch as st

# trainer is the settled ResidentVisionTrainer from the client guide.
meta = st.zspace_meta_optimizer_init({"dimension": 2, "topos_control_gain": 1.0})
report = st.zspace_meta_optimizer_step(
    config=meta["config"], state=meta["state"],
    observation={"gradient": [0.1, -0.2],
                 "telemetry": {"topos.training_hints.learning_rate_scale": 0.5}},
)
receipt = json.loads(trainer.apply_zspace_meta_optimizer_report_json(json.dumps(report)))
submitted = trainer.submit_next()
outcome = trainer.settle()
```

The gradient and hint above are a prescribed example, not measurements of the
image model or a recommended training policy. Actual policy evaluation must
define the observations and preserve/restart its producer independently.

```javascript
// reportJson is the complete Rust-produced report, not a hand-edited scale.
const receipt = JSON.parse(trainer.applyZSpaceMetaOptimizerReportJson(reportJson));
const submitted = trainer.submitNext();
const outcome = await trainer.settle();
submitted.free();
outcome.free();
```

## Clocks And Failure Semantics

- The effective rate is `nominal_schedule_rate * absolute_scale`, once. A
  repeated report with the same source step and applied scale never compounds.
- The source meta-step and accepted model-update count are different clocks.
  Stale controls and conflicting scales at the same source step are rejected.
- Control application requires a settled owner and does not consume input or
  advance the schedule. A pending update also blocks duplicate controls.
- The source projection checks the report contract and recomputes its Topos
  rate control. It is not writer authentication or a proof of all latent math.
- A numerical model rejection consumes the batch, retains all weights and
  leaves the accepted-update schedule and applied control unchanged.
- Non-finite/overflowed effective rates and positive rates rounded to zero
  fail before input consumption. Explicit zero-rate probes remain zero.
  Submission checks again because a schedule may later reach another rate;
  a settled checkpoint remains capturable so a caller can retune and resume.

`state_json()` / `stateJson()` exposes the applied scale and source meta-step
under `parameter_control`. Rust exposes the same validated state through
`ResidentVisionTrainingState::parameter_control()`.

## Checkpoint Compatibility

Uncontrolled trainers retain `spiraltorch.vision.training_checkpoint.v1` and
omit the default control field, preserving previous plain-SGD payload bytes.
Once a control has been applied, the checkpoint is explicitly `v2`, with the
scale and source clock included in the trainer hash. Resetting the scale to
one does not erase its replay guard or downgrade it to `v1`. Old readers must
reject controlled checkpoints rather than silently lose that state.

Restore prepares a complete replacement before mutating the live owner.
Control corruption, invalid state or a mismatched schema is rejected.

## Exercise The Public Clients

Build a fresh wheel and the `webgpu` WASM module as described in
[the trainer client guide](resident_vision_trainer_clients.md). The controlled
fixture uses the same real GPU model, inputs, schedules and browser server:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 \
SPIRALTORCH_VISION_CONTROL_HANDOFF=/tmp/vision-control-native.json \
python -I bindings/st-py/tests/test_vision_trainer_control.py -v
node tools/serve_vision_trainer_fixture.cjs MODULE_DIR \
  /tmp/vision-control-native.json /tmp/vision-control-browser-new 8769
```

Run `/?schedule=constant&mode=control`, then `prefix`, close that document,
and open `resume` in a fresh document. `mode=python` instead continues the
native prefix. Repeat with `schedule=cosine`. This is a five-report prescribed
replay over 100 attempts, split 37/63, with deliberate rejected updates. It
tests the consumer's control state, not adaptive-policy quality or producer
restart. Browser-to-native continuation can then be checked with:

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 \
SPIRALTORCH_VISION_CONTROL_BROWSER_DIR=/tmp/vision-control-browser-new \
python -I bindings/st-py/tests/test_vision_trainer_control.py \
  Gpu.test_browser_controlled_checkpoints_continue_in_native -v
```

The Rust regression also compares every actual updated parameter with an
explicit half-rate SGD control, and checks that an unscaled run differs.
Neither this replay nor a learning-rate response demonstrates a Z-space
quality, throughput or memory advantage. Those require matched real-data
policy-on/off measurements, including a fixed-rate control.

The [bounded native/browser result](../benchmarks/results/2026-10-01-vision-zspace-control/README.md)
records the two schedules, all eight browser phases and both handoff directions.

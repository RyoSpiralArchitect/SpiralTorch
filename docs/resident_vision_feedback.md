# Resident Vision Loss Feedback

The resident trainer can opt into the existing
`st-core::runtime::zspace_optimizer_feedback` loss gate. The Rust trainer
observes its actual frozen pre-update cross-entropy after an accepted update,
then gates the next external Z-space rate proposal toward identity. Python
and WASM only supply the configuration and transport proposals; they do not
observe loss or reconstruct the gate to make the trainer work.

```python
config = json.loads(st.ResidentVisionTrainer.default_config_json())
# Set the model/input/rate configuration as in the trainer client guide.
config["optimizer_feedback"] = {}  # validated core defaults
trainer = st.ResidentVisionTrainer.create(
    device, dataset, dataset_sha256, json.dumps(config), pipeline)
# Existing Rust-produced reports remain the source of rate proposals.
trainer.apply_zspace_meta_optimizer_report_json(report_json)
submission = trainer.submit_next()
outcome = trainer.settle()  # Rust observes accepted loss and updates gate history
```

The same optional `optimizer_feedback` configuration is accepted by WASM
`ResidentVisionTrainer.createWithPipeline`. Await `settle()` in the browser.
Rust callers may instead call `enable_zspace_optimizer_feedback(config)` once
on a fresh trainer, before the first submission. No mid-run reconfiguration
or implicit history reset is allowed.

## Clocks And Ownership

- The applied multiplier is the existing core rule
  `1 + effective_gate * (proposed_scale - 1)`, multiplied by the nominal
  scheduler rate once. It is not multiplied by the proposal a second time.
- `parameter_control` and its application receipt describe the ungated proposal
  and source meta-step. `submission.learning_rate` is the actual gated rate.
- A preview does not mutate the live gate. Settlement advances its control
  clock once per attempted update; its observation count advances only on
  accepted updates. The nominal scheduler still advances only on acceptance.
- Rejected attempts consume the batch without observing its loss. This leaves
  the prior loss/EMA intact and lets the existing core staleness rule suppress
  an unsupported proposal on the next attempt.
- The observed loss belongs to the forward that produced the accepted gradient,
  not a second post-update forward or a development-set evaluation.
- Failed input preparation does not advance the gate. Mapping failure or async
  cancellation keeps the update pending and prohibits reuse/checkpointing;
  retrying settlement does not duplicate an observation.

The opt-in gate needs one scalar loss snapshot in addition to the acceptance
receipt already observed by this trainer. It does not read all parameters,
images or gradients. This is an explicit synchronization cost, not a claim of
readback-free training. Disabled trainers retain their previous path.

## Optional Observation Windows

Set `optimizer_feedback.loss_window_observations` to a positive integer before
training to compare non-overlapping, equal-weight means of accepted losses.
The default is one, preserving the existing adjacent-loss rule and JSON bytes.
For example, `{"loss_window_observations": 80}` compares 80 accepted batch means
at a time. This is not a sample-weighted mean, a moving window, or an automatic
epoch detector. Rejected updates consume input without joining the window, so
80 accepted observations need not cover exactly one 80-batch input pass.

Partial windows return `await_window` and preserve the gate, streaks and
relative-delta EMA. Raw loss telemetry, absolute-loss EMA and the accepted
observation clock still advance. Only a completed window changes the gate;
the first complete window establishes a reference. Consequently a larger window
delays both opening the gate and detecting real regression. Warmup is still
specified in accepted observations, checked at completed-window boundaries;
staleness still refers to the last accepted observation, not the last boundary.
All accepted updates still map their scalar loss: windowing does not remove
the trainer's existing readback cost.

Window width, partial count/mean and previous completed mean are Rust-owned
checkpoint state. They cannot be silently changed or discarded on restore.
This opt-in additive extension retains the feedback v1 / trainer v3 identifiers;
older strict readers reject its unknown fields. Default width-one payloads
omit the extra fields and remain readable by existing clients.

The fixed-model probe motivates a coverage-derived width of 80 for its specific
1,280-image, batch-16 task. This is not a recommended general default or evidence
of improved training. Evaluate real-regression latency and matched learning
controls before adoption. Python/WASM use the same Rust aggregation, including
partial-window restart; no client-side averaging is required.

## Restart

Feedback-enabled checkpoints use `spiraltorch.vision.training_checkpoint.v3`.
The configuration, loss history, EMA, gate and control clock are bound into
the trainer hash. Restore validates the core state and requires the control
clock to equal attempted updates and observation count to equal accepted
updates. Uncontrolled v1 and external-control-only v2 encodings are unchanged.

This captures the **feedback gate**, not the latent state of an external
meta-optimizer that supplies proposals. Applications must still preserve that
producer separately. A gate checkpoint must not be advertised as full producer
restart or geometric optimizer state.

## Evidence And Limits

The Rust regression compares each actual gate transition against the existing
core functions, feeds the resulting rates to an independent plain resident
SGD owner, and checks every resulting parameter. Constant and cosine cases
also compare uninterrupted 100-attempt training with a 37/63 restart, including
rejected updates. The public Python fixture runs those phases in distinct
processes and records every feedback state as well as inputs/rates/acceptance.

```bash
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 cargo +1.98.0 test -p st-vision \
  --features wgpu --lib resident_trainer -- --test-threads=1
SPIRALTORCH_RUN_WGPU_RUNTIME_TESTS=1 \
SPIRALTORCH_VISION_FEEDBACK_HANDOFF=/tmp/vision-feedback-native.json \
python -I bindings/st-py/tests/test_vision_trainer_feedback.py -v
```

Install a fresh wheel with `nn,wgpu` enabled before the Python run. Build the
WebGPU WASM module as in the [trainer replay guide](resident_vision_trainer_clients.md#portability-and-replay),
then serve the feedback fixture instead of the plain-SGD fixture:

```bash
node tools/serve_vision_trainer_fixture.cjs MODULE_DIR \
  /tmp/vision-feedback-native.json /tmp/vision-feedback-browser-new 8770
```

Run the guide's constant/cosine control, prefix, resume and Python-handoff modes
on port 8770, closing the prefix document before opening resume. Use a new
output directory for every run; the server refuses to replace recorded phases.
Compilation alone does not establish browser execution or cross-runtime replay.

The first actual browser probe passes uninterrupted and fresh-document restart
for both schedules, but the strict Python-to-browser continuation probe fails:
some GPU cross-entropy observations differ by up to two f32 ULPs and therefore
change the loss EMA. On these two fixtures the applied rates and all final weights
still match; that does not establish bitwise feedback-state replay or behavior
near a gate threshold. The failed receipts are retained, and the strict
comparison is not relaxed. Replaying the browser's **observed** losses through
native Rust separately checks whether the control math itself agrees.
See the [measured results and retained failures](../benchmarks/results/2026-10-01-vision-feedback/README.md).

The [rounding follow-up](../benchmarks/results/2026-10-01-vision-ce-rounding/README.md)
isolates this to CE's `log_one_plus` correction in the measured cases. It now
uses the existing integer-defined f32 addition boundary for `1 + tail` and
the rounded subtraction, instead of allowing their cancellation back to `tail`.
The correction formula, loss observation and gate thresholds are not replaced
by a tolerance or by quantizing the observed loss. This also fixes a genuine
small-loss accuracy regression, independently of feedback.

Fresh native/browser builds from the same source now pass the unchanged strict
continuation comparison in both directions, including all feedback state and every weight.
This is evidence for the tested devices and fixtures, not universal cross-GPU
bitwise determinism. In addition, `st-core` enables exact JSON float roundtrips
itself: loss-history restoration must not depend on a binding crate enabling
the parser feature. A standalone Rust regression checks the actual f64 state,
not just approximate equality. Existing checkpoint histories are preserved;
reproducing an older build's future trajectory still requires its original
loss kernel and numerical behavior.

The subsequent macOS CI run found that preserving the rounding correction alone
does not guarantee small-loss accuracy on another runtime: at logit gap 12 the
relative error was `2.65e-4`, above the unchanged `2e-5` regression bound. The
first small-tail correction used `x * (1 - x/2)` for `x <= 2^-10`.
A second CI run passed that range but exposed `2.37e-5` relative error at gap
`6.0078125`, outside it. The shared kernel now uses
`x * (1 + x * (-1/2 + x/3))` for `x <= 2^-6`, with explicit rounding at both
Horner sums. Its analytic truncation error is below `9.7e-7` relative; device
arithmetic still requires the GPU regression. The unchanged `2e-5` test now
covers gaps 0 through 22 in increments of 1/256 and neighboring f32 values at
both old and new branch boundaries. The earlier results remain tied to their
recorded builds and are not proof of this follow-up.

The gate guards an external intervention, not the baseline SGD itself: a halted
gate returns to the nominal rate and does not stop training. For a proposal
below one, closing the gate increases the rate back toward nominal. Batch-to-
batch loss changes also include sampling noise. No convergence, quality,
stability or speed advantage follows from wiring correctness. Real-data
policy-on/off and integrated-rate-matched comparisons remain necessary.

The shared `ZSpaceParameterFeedbackState` adapter validates its finite f32 loss
domain before any parameter update. A positive but extremely small f64
`loss_floor` could otherwise overflow the relative delta after a GPU update was
already accepted, leaving settlement pending. Such configurations, and restored
histories outside the adapter's arithmetic domain, now fail before submission.
This adds headroom for relative-delta/EMA arithmetic without changing ordinary
configurations, clipping observed losses, or changing the standalone f64 core.

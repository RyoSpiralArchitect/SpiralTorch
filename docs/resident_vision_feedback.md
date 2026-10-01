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
small-tail path now uses `x * (1 - x/2)` through the same explicit subtraction
boundary for `x <= 2^-10`. Its analytic truncation error is below `3.2e-7`
relative; device arithmetic still requires the GPU regression. This avoids the
near-one logarithm in that range rather than weakening the test. The earlier
results remain tied to their recorded builds and are not proof of this follow-up.

The gate guards an external intervention, not the baseline SGD itself: a halted
gate returns to the nominal rate and does not stop training. For a proposal
below one, closing the gate increases the rate back toward nominal. Batch-to-
batch loss changes also include sampling noise. No convergence, quality,
stability or speed advantage follows from wiring correctness. Real-data
policy-on/off and integrated-rate-matched comparisons remain necessary.

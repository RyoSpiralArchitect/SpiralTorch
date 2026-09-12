# Resident Momentum Preparation Fusion

The resident learner fuses effective-gradient normalization, optional clip
scaling, Topos EMA and weight-candidate preparation. Python and WASM use the
same Rust implementation without a new optimizer surface or routing policy.
This removes an intermediate GPU-buffer round trip, not a CPU readback: the
previous learner already kept this computation on GPU.

For `N` parameter tensors, the update phase excluding VJP composition changes
from `3N + 1` to `2N + 1` dispatches without clipping, and from `4N + 2` to
`3N + 2` with clipping. Global norm reduction, the whole-update validity decision
and the weight/history commit retain their boundaries. No invalid parameter
may leave a partially updated model or advance only its history.

The clip-only and EMA paths share the normalization/scaling WGSL functions.
Both retain the ordered normal-f32 scale factors for very large finite norms
and tiny clip limits. Clipped bindings are prepared once when needed, regardless
of whether clipping or momentum is enabled first. Later updates allocate no
new bindings or tensor-sized buffers. Disabling either option preserves its
documented lifecycle; ordinary SGD does not execute the EMA kernel.

## Matched Measurement

The existing training benchmark now accepts `--learner-optimizer topos_ema` or
`--learner-optimizer clipped_topos_ema`. Both lanes and eager PyTorch use the
same choice, zero initial history and damping `0.5`; the clipped mode uses a
global norm limit of `1/1024`. These are fixed benchmark settings, not recommended
training hyperparameters. Omitting the option preserves the plain-SGD control.

The standard matrix is three seeds and three mixed graphs: shape `[2,16,32]`
at depth 2, `[2,129,32]` at depth 4, and `[4,32,64]` at depth 8. All use the same
quadratic/quartic objective, two exact VJPs, weights `0.75/0.25`, learning rate
`0.01`, eight updates per interval and immediate/deferred observation cadences.
Every interval resets weights and history outside timing. Each cadence has
two discarded warmups and eight retained blocks, with rotated lane order.

Rust reads every requested update-acceptance receipt; PyTorch only synchronizes
completion and has no equivalent stage guards or transactional rollback.
Terminal losses, weights, both VJPs and momentum history are compared outside
timing. History participates in reset-state fingerprints and admission checks;
an interval cannot be relabelled as a different optimizer. This is neither a
peak-GPU benchmark nor a comparison with `torch.optim.SGD` heavy-ball momentum.

Build the same benchmark harness on separate immutable baseline/candidate
sources, preserving each native binary and generated WASM module. For example:

```bash
python -I tools/bench_resident_training_vs_torch.py \
  --baseline /path/to/baseline-binary --baseline-source BASELINE_COMMIT \
  --candidate /path/to/candidate-binary --candidate-source CANDIDATE_COMMIT \
  --graph --learner --learner-optimizer clipped_topos_ema \
  --device mps --output /tmp/new-native-ema.json

node tools/bench_resident_training_browser.cjs \
  /path/to/baseline-module /path/to/candidate-module /path/to/chromium \
  /tmp/new-browser-ema.json learner standard none clipped_topos_ema

python -I tools/validate_resident_training_bench.py \
  --native /tmp/new-native-ema.json --browser /tmp/new-browser-ema.json \
  --browser-progress /tmp/new-browser-ema.json.progress.jsonl \
  --baseline-source BASELINE_COMMIT --candidate-source CANDIDATE_COMMIT \
  --browser-harness-source HARNESS_COMMIT --output /tmp/new-ema-validation.json
```

Use fresh outputs, serialize owned GPU work, retain all timings and failures,
and keep numerical correctness distinct from performance. Readback costs and
host contention must remain explicit; no CUDA, FT quality or generalization
claim follows from dispatch-count reduction.

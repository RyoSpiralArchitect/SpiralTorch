# SpiralTorch Ecosystem Roadmap

SpiralTorch already offers a rich Rust-first runtime, a shared hypergrad tape for the Python bindings, and a TypeScript-powered collaboration canvas. This document captures the near-term ecosystem priorities so contributors can converge on the same themes while the core crates continue to evolve.

## Learning Geometry Rail

Return the main research effort to language-model learning, while retaining the
vision correctness fixtures as regression controls. Keep the pure speed race
limited to PyTorch-equivalent computation. Novel geometry earns its place via
actual gradients/updates, stability and matched learning controls, with extra
compute and transfers reported separately.

The first [geometric learning bridge](geometric_learning_bridge.md) connects the
existing Rust Topos recurrence and VJP to a zero-initialized Torch residual gate
and to a browser forward/VJP client. A random tiny HF loss updates that gate;
the three-seed control experiment does **not** establish a quality advantage.
The [elliptic/Lie connection](elliptic_learning_bridge.md) now repairs near-pole
derivatives and large finite norms, owns batched VJPs in Rust, and reaches both
trainable projections in a bounded pretrained GPT-2 run. Its tangent-linear
control performs better on the tiny authored corpus; this is not a quality win.
The [WaveGate pullback repair](wave_gate_learning.md) now includes the missing
parameter saturation derivative, separates raw VJPs from training-policy rewrites,
and hardens shared CPU/WGPU projection arithmetic. Its Rust/Python Tensor API is
available, together with an owned Rust forward snapshot consumed by a Torch
residual adapter and a WASM learning client. Isolated off/tangent/WaveGate
controls precede combinations. Rust owns each geometric
rule; Python/WASM own transport and orchestration, not replacement mathematics.
Pretrained FT quality, mixed precision and resident geometric execution remain
open, rather than being inferred from the wiring tests.

The [paired novel conditioning pilot](../benchmarks/results/2026-10-02-wave-gate-pride-conditioning/README.md)
now exercises 12 conditions and actual adapter/Adam continuation. WaveGate learns,
but the tangent control is slightly better in all three minibatch schedules.
No logged training input reaches elementwise saturation; relaxing its threshold
changes nothing. Rust-owned projection-gain observations instead motivate testing
an explicitly parameterized, learnable projection radius with a preserved initial
gain. The [radius pilot](../benchmarks/results/2026-10-02-wave-gate-pride-radius/README.md)
now connects that scalar through Rust/Python/WASM and completes 18 conditions.
All 15 active adapter/Adam continuations match exactly. Learning radius improves
its fixed-radius controls very slightly, but tangent remains best in all seeds.
Widening radius approaches the linear map, not proof of geometric advantage.
The [fixed 512-update study](../benchmarks/results/2026-10-02-wave-gate-long-horizon/README.md)
now completes all nine runs, exact continuations and delayed evaluation on 120
unused within-book blocks and 32 transfer-book blocks. Every active arm improves
versus the frozen baseline, but tangent remains best for every seed on both sets.
Learned radius grows from 4 to about 9.2-9.3 and improves fixed radius without
closing that gap. Nonlinear controls and directional/relational geometry remain
open; this is an implemented learning mechanism, not a geometric quality or
speed win. CI now exercises the geometric clients and exact tiny-HF continuation
instead of leaving that coverage solely in local experiments.

The [causal factorial](elliptic_causal_study.md) and
[learned-context comparison](elliptic_gated_study.md) retain ordinary tangent
controls as the stronger learning path. A signed context gate modestly improves
the elliptic mean, but geometry still loses in every seed on both reused sets.
The [frozen-checkpoint intervention](elliptic_context_ablation.md) then separates
local gain, fixed-anchor correction and true context; its outcomes are not
retraining evidence. The resulting [Rust-owned anchored operator](elliptic_anchored_learning.md)
connects the same input/shared-gate VJPs to Python/HF and WASM without adding a
sequence cache or quadratic anchor attention.

The [anchored training study](../benchmarks/results/2026-10-03-elliptic-anchored-study/README.md)
now completes four equally parameterized arms across three seeds and 512 updates
each. Fixed-anchor elliptic improves over gated-context elliptic in all seeds on
both sets (-0.028651 mean Pride CE, -0.011172 Alice), but ordinary anchored tangent
still wins every seed (+0.116164 and +0.064956 primary geometry gaps). All saved
states and exact continuations are verified; six old-control replays are not
independent confirmations. Before adding another mixer, inspect nonlinear chart
feature/gradient conditioning against the ordinary control. The experiment does
not establish that conditioning causes the remaining gap, nor any speed advantage.

## Restarted execution slice: vision across Rust, Python, and browser

Current scope of the vision execution path:

| Milestone | Rust | Python | Browser/WASM |
| --- | --- | --- | --- |
| Seeded resident geometry batch | Public API | Public API | Public API, async snapshots |
| Model-owned resident ConvNeXt forward | `Module` API | Via resident classifier handle | Via resident classifier handle |
| Full-backbone resident VJP | `ConvNeXtBackbone::vjp_resident` | Via classifier gradients, including head | Via classifier gradients, including head |
| ConvNeXt-owned resident parameter update | `compile_resident_training`, plain SGD | Via classifier, explicit receipt | Via classifier, async receipt |
| ConvNeXt plain-SGD model checkpoint/resume | Public API, explicit host handoff | Classifier JSON, resident restart and host handoff | Classifier JSON, resident restart and async mapping |
| Real ConvNeXt classifier common inference entry | `ConvNeXtClassifier`, `create_classification_model` | Ordinary factory with `nn`; resident handle with `nn,wgpu` | Resident handle; no ordinary host factory |
| Classifier-owned resident CE/VJP/SGD/checkpoint | One owner for backbone and head | Thin Rust-owned public client | Thin Rust-owned public client |
| Resident image normalization and DataLoader handoff | Public API, homogeneous NCHW | Public API, same Rust loader | Public normalization/batch/continuation API; caller-supplied batches, no JS DataLoader |
| Portable input checkpoint | Order/cursor, shuffle and transform RNG | Same Rust loader/transform state | Transform state; integrated trainer also owns order/cursor/shuffle |
| Unified resident training boundary | Model/input/schedule owner, explicit settlement and bound checkpoint | Thin Rust-owned trainer client | Same Rust owner, async settlement; fresh-document and cross-runtime restart exercised |
| Matched real-image learning correctness | Rust/WGPU execution via Python client | Bounded CIFAR-10 / PyTorch comparison; shared-trainer native process restart | Open |
| Transfer-inclusive training throughput | Rust/WGPU execution via Python client | Bounded matched native timing; large-batch gap remains | Open |
| Training peak GPU memory | Open | Open | Open |

The compiled ConvNeXt training path now connects model-owned gradients to
resident parameter updates with version checks, stale-gradient rejection and
all-parameter acceptance. Native and browser fixtures cover consecutive steps,
rejected steps preserving all weights, and a valid retry. Portable plain-SGD
model checkpoints now preserve weights, architecture and attempted-update
revision, with explicit mapping and fresh restored owner identities. Model-only
checkpoints exclude data cursors, RNGs, rate schedules and trainer policy; the
resident trainer below joins the supported components at one settled boundary.
The common public ConvNeXt entry now owns the real backbone, channel-preserving
global-average pooling, and Linear classification head. Classifier learning uses
one resident parameter owner and one acceptance/revision clock for all weights.
Its checkpoint can return the learned model to the ordinary inference interface.
Python's `nn` feature enables this same Rust factory. See
[the classifier contract result](../benchmarks/results/2026-10-01-convnext-classifier-contract.md).

The [thin training clients](resident_vision_training_clients.md) now expose the
Rust classifier through Python and JavaScript, including versioned forward and
gradient handles, frozen update receipts, and portable model checkpoints. They
do not duplicate training rules or equate a submitted update with acceptance.
The browser-to-Python replay uses the recorded normalized input; it does not
claim DataLoader/RNG continuation or real-data learning quality.

The [resident input path](resident_vision_input.md) now connects Normalize,
geometry and Rust/Python DataLoader batches to the existing classifier on one
device, without intermediate image readbacks. Browser clients expose the same
Rust transforms and accept caller-supplied batches. A shared native/browser
fixture checks four normalized-input classifier updates and all-weight rejection
of invalid normalization. This remains synthetic correctness evidence, not
real-image quality or a throughput claim. Resident submission advances input
cursor/RNG before deferred GPU validity is observed; model checkpoints still do
not capture that input state. See the
[input contract result](../benchmarks/results/2026-10-01-resident-vision-input/README.md).

The [input checkpoint](vision_input_checkpoint.md) now preserves order/cursor
and augmentation RNG independently of the model. A native GPU fixture matches
100 classifier update attempts against a restart after attempt 37, including
two rejected updates. Python-to-wasm32 transform replay is exact under Node.
These payloads do not form one integrity-bound model/input/trainer checkpoint
by themselves; the unified trainer below supplies that boundary.

The [resident trainer](resident_vision_trainer.md) now owns those components and
the update settlement boundary. It reuses the existing Rust warmup/cosine scheduler
and checkpoint hashing, rather than reconstructing either in clients. Its native
process-restart fixture exercises constant SGD and scheduled rates, including
rejected updates. The [integrated trainer clients](resident_vision_trainer_clients.md)
now expose that same owner in Python/WASM. Both fixed and scheduled learning
exercise a 100-attempt control versus a 37/63 restart, with 90 accepted and ten
rejected updates. Actual browser documents are closed and recreated, and native
and browser checkpoints continue in both directions. This caught and fixed a
one-ULP platform-cosine rate difference in the shared Rust scheduler; no client
math or relaxed rate comparison was added. The bounded synthetic replay is not
a browser real-image or throughput measurement.
See the [client replay result](../benchmarks/results/2026-10-01-vision-trainer-clients/README.md).

### Next Rails And Exit Gates

The [matched real-image result](../benchmarks/results/2026-10-01-vision-matched-learning/README.md)
now covers three seeds against PyTorch CPU/MPS, including 10,000 training
images and 1,600 development-evaluation images on the expanded MPS comparison.
All five epochs complete with matching evaluation accuracies. This is a small
shared-architecture learning baseline, not a speed, full-dataset quality or
Z-space policy advantage. Every step observes loss/acceptance on the host.

The [shared-trainer real-image replay](../benchmarks/results/2026-10-01-vision-trainer-realdata/README.md)
now carries that Rust-owned boundary into 1,280 training / 320 development images,
three seeds and five epochs. Both fixed SGD and flip plus warmup/cosine runs
match their uninterrupted trajectory after a fresh-process 37/363 restart,
including input/schedule state and every weight. Torch remains an independent
model under the Rust-selected inputs/rates; both frameworks' final weights are
retained and checked. This closes native real-image trainer continuation, not
browser real-image restart, throughput or a Z-space policy advantage.

1. **Extend matched real-data evidence.** Keep the completed numerical/learning
   baseline as the control while measuring transfer-inclusive throughput,
   memory, and restart equivalence. Carry the same task into the browser,
   recording that environment separately. Neither synthetic loss decrease nor
   kernel timings close these remaining gates; new model families or a model
   hub are not prerequisites.
   The [first transfer-inclusive timing sweep](../benchmarks/results/2026-10-02-vision-training-throughput/README.md)
   now completes 180 workers across three seeds, batches 1/16/64 and both plain
   and identity-feedback modes. All final weights match the eager Torch
   reference within the existing bound; all 90 baseline/candidate Rust
   checkpoints are identical. Combining receipt and feedback-scalar readback
   preserves correctness but does not establish a general speedup. The
   large-batch throughput gap remains around 21x on this small Apple M4 model.
   Profile GPU passes, then optimize the measured dominant reductions; do not
   label host settlement time as mapping-only cost. Peak GPU memory and browser
   real-image throughput remain open gates.
2. **Restartable input and trainer state.** Keep model snapshots distinct from
   data order/cursor, augmentation RNG and schedule state, then connect their
   restart contracts. Specify whether rejected updates retry or consume a batch.
   Compare uninterrupted and resumed batch identities, transforms and updates.
   The Rust resident trainer now connects this boundary for full fixed-size
   classifier batches. Thin clients, synthetic fresh-browser-instance restart,
   and native real-image process continuation are now exercised. Carry the
   real-image comparison into the browser without duplicating input/trainer
   policy in JavaScript; cross-device and crash-durability guarantees remain open.
3. **Shared resident optimizer control.** Connect existing Rust optimizer and
   Z-space policy to the resident parameter owner rather than reimplementing it
   in Python/JavaScript. Preserve the plain-SGD control case and all-parameter
   acceptance, then measure policy-on/off quality and stability on the same data.
   The [first shared connection](resident_vision_zspace_control.md) now applies
   validated absolute rate control with replay guards and checkpointed consumer
   state. Producer latent state, geometric parameter updates and demonstrated
   real-image policy benefit remain open; rate modulation alone is not those gates.
   The optional [loss-feedback connection](resident_vision_feedback.md) now
   feeds accepted resident losses into the existing Rust gate and checkpoints
   its history. Rejections preserve the prior observation and trigger the core
   staleness rule. This is not a new policy, full producer-state migration, or
   a measured policy advantage.
   A [shared CE rounding fix](../benchmarks/results/2026-10-01-vision-ce-rounding/README.md)
   closes the measured feedback continuation mismatch without relaxing the
   comparison. Standalone Rust also owns exact JSON float restoration. The next
   policy gate remains matched real-image learning, not more reporting surfaces.
   The [first four-arm ablation](../benchmarks/results/2026-10-01-vision-feedback-ablation/README.md)
   now completes three seeds, five epochs and fresh-process restarts. The gate
   is active and suppresses a harmful prescribed half-rate proposal, but does
   not demonstrate an advantage over nominal SGD or an integrated-rate-matched
   control. Preserve that negative result. Next, examine whether the shared
   Rust observation rule distinguishes batch noise from actual regression, then
   rerun the same controls; do not count more reports or tuned seeds as progress.
   The [fixed-model probe](../benchmarks/results/2026-10-01-vision-feedback-stationarity/README.md)
   now shows substantial gate activity on all six frozen initial/final models,
   despite stable full-pass loss. An opt-in, checkpointed
   [Rust observation window](resident_vision_feedback.md#optional-observation-windows)
   now preserves partial aggregates and the original default. Native/Python/WASM
   checks cover windowed transitions and restart. The
   [window comparison](../benchmarks/results/2026-10-01-vision-feedback-window/README.md)
   now completes matched learning and latency probes: frozen-model gate activity
   disappears, but regression detection is delayed and learning does not improve
   over the relevant controls. Retain the default, preserve this negative result,
   and advance transfer-inclusive execution measurements rather than tune until
   these development seeds favor a window. Model-derived proposal production and
   its latent-state restart remain separate future optimizer gates.

### Focus Handoff: Back To Language Models

The [bounded convolution diagnostic](../benchmarks/results/2026-10-02-vision-vjp-profile/README.md)
closes this vision execution slice without claiming the remaining performance
gap is solved. All 54 cases preserve both routes' complete reference checkpoints.
At batch 64 the existing convolution VJP passes total about 3.2 ms against a
roughly 63 ms profiled host step. The rest is unattributed; neither convolution
dominance nor a mapping-only explanation follows. Shared NN graph backward and
execution waits remain candidates to examine on language-model workloads.
Browser real-image restart, peak memory and useful vision policy remain open,
but are not prerequisites for returning the main development effort to LLMs.

The next learning slice is the existing Rust-owned repetition-unlikelihood
objective used by HF/PEFT, not another reporting surface or a complete HF
backend replacement. The retained
[GPT-2 long-horizon result](benchmarks/hf_periodic_gpt2_pride_full_corpus_256step_20260823.json)
records early benefit followed by later reversal and unequal auxiliary-loss
magnitudes at equal nominal strength. This motivates testing an explicit Rust
objective budget/normalization or update-time schedule, not declaring cumulative
over-application proven. Preserve the current default as a frozen control;
include gradient accumulation, masked tokens and resumed optimizer-update clocks
in the new contract. Then compare ordinary FT, the frozen intervention and one
prespecified candidate over the same longer horizon with fresh seeds, held-out
causal-LM loss and generated-text assessment. Decoding-only controls must not
stand in for a learning improvement. HF retains its differentiable model graph;
resident Linear/normalization/adapter execution can migrate in separately tested
slices rather than delaying this learning question for a complete decoder port.

The [aligned schedule comparison](../benchmarks/results/2026-10-02-llm-schedule-v2/README.md)
now completes nine 256-update runs and all 432 continuations. Ordinary FT learns,
but decay loses to both constant intervention and ordinary FT on the final loop
score in all three seeds. The held-out-loss safety margin passes, not a claim of
equivalent loss or better language. Keep this negative result and the separately
invalidated v1 attempt; do not extend schedule search until these seeds favor it.

The next language rails preserve Z-Space integration and independent PyTorch
controls rather than choosing between them:

1. **Improve the learning mechanism.** Inspect teacher-forced candidate exposure
   and generated-context repetition using the shared Rust periodicity rules.
   Candidate selection, objective budget and control semantics stay in Rust;
   Python/HF and WASM expose the same contracts. A generated-context intervention
   is a hypothesis, not an established fix. Freeze new quality controls and
   account for extra compute before testing it; never turn held-out prompts into
   training data.
2. **Connect resident causal attention and Z-bias.** Carry Q/K/V projections,
   causal attention and output projection without intermediate host readback.
   Compare both zero-bias and the same nonzero geometric bias with PyTorch, so
   the numerical reference computes the same operation. Masking, sequence/head
   layout and cached-query offsets require explicit shared semantics before KV
   cache expansion; do not implement a separate client-side interpretation.
   The [first resident forward slice](resident_zspace_attention.md) now covers
   causal offsets and both score biases, with matched PyTorch fixtures on native
   WGPU and browser WebGPU. Frozen QKV/output projection lowering now connects
   existing `Linear` and `ZRBFAttention` parameters and geometric bias without
   activation readback, with 12 matched full-chain controls on both clients.
   This is mean-only inference; uncertainty outputs, backward, KV-cache ownership
   and complete decoder integration remain open.
3. **Measure the complete language path.** Require numerical agreement before
   transfer-inclusive throughput measurements, with resident-only timings
   labeled separately. Browser WebGPU, native WGPU and PyTorch CPU/MPS/CUDA are
   distinct observations, not interchangeable performance evidence. Learning
   quality remains a separate matched-model, matched-budget gate.
   The [first full-chain comparison](../benchmarks/results/2026-10-02-attention-key-tiling/README.md)
   now retains all plain/zero/Z-RBF controls, including a rejected unconditional
   key-tiling rollout. A shape-selected tile improves the measured 128/256-token
   regimes versus the previous ST kernel, but still trails PyTorch MPS; short
   samples remain sensitive to warmup. The next
   [projection-only comparison](../benchmarks/results/2026-10-02-attention-projection-kernels/README.md)
   connects existing Register2x2 kernels to both projections, with native/WASM
   parity. The explicit 16x16 preset improves the measured larger inputs, while
   Scalar remains the default because short/device behavior is not settled.
   Attention now reads Q/K/V and broadcast biases directly from N-D views,
   removing their temporary materializations while preserving inherited guards.
   The [direct-view comparison](../benchmarks/results/2026-10-02-attention-strided-inputs/README.md)
   finds short-input gains but mixed/noisy wide-input timings; negative results
   and a post-hoc sensitivity run are retained. Output head merging and
   submission costs remain optimization targets. These are inference
   measurements, not a reason to claim LLM quality gains.

Other model kinds still route through legacy `SimpleCnn`; this slice changes
only ConvNeXt. Model hub, more model families, and broader interop follow the
working end-to-end path rather than multiplying disconnected entry points.

<details>
<summary>Execution history and bounded measurements</summary>

The candidate arithmetic is now shared by the existing dense, graph, clipped,
and EMA update routes through `st-kernel-contracts::sgd`. The Rust CPU oracle
and generated WGSL check the gradient, multiplication, and subtraction before
the existing all-parameter commit decision. This supplies a common update rule;
the compiled model connection below supplies parameter ownership and versioned
gradient handoff.
The bounded native and browser checks are recorded in the
[shared SGD contract result](../benchmarks/results/2026-09-30-shared-sgd-contract.md).

The next ownership slice supplies model-independent
[`ResidentParameters`](resident_parameters.md): immutable GPU parameter versions,
explicit derivative binding, and one all-parameter SGD decision without host
mapping. The native/browser fixture feeds its updated weights into consecutive
Conv2d forward/loss/VJP steps. That fixture alone is not a ConvNeXt model update,
and checkpoint/optimizer state must not be inferred from weight snapshots.
The bounded checks are recorded in the
[resident parameter ownership result](../benchmarks/results/2026-09-30-resident-parameter-owner.md).

[`ConvNeXtBackbone::compile_resident_training`](resident_convnext_training.md)
now snapshots host weights into a separate GPU-owned model. Stem, every block,
downsampling and final normalization consume the updated resident values on the
next forward. Convolution geometry comes from the original Rust layers; the
Linear/LayerNorm/GELU subgraphs reuse graph-autograd with explicit GPU weight
rebinding. The host model remains independent rather than silently replacing
device updates through its inference caches. The same Rust fixture executes
eight full-backbone MSE/VJP/SGD steps in native and browser GPU runtimes without
readbacks inside that loop. It compares every gradient, weight version and next
prediction with CPU, including rejection/retry and retained snapshots.
See the [bounded learning result](../benchmarks/results/2026-09-30-convnext-resident-learning.md).
This is plain SGD, not ModuleTrainer policy migration, a JavaScript/Python model
API, or demonstrated real-image quality/throughput.

The next completed slice adds explicit model checkpoint capture, portable JSON,
host handoff, and resident restart. A shared native/browser fixture checks
bitwise uninterrupted-versus-resumed predictions and all 22 weights; a checkpoint
actually emitted by Chrome also resumes in native Rust within the existing f32
parity bound. See [the checkpoint result](../benchmarks/results/2026-09-30-convnext-resident-checkpoint.md).

The classifier slice closes the earlier split between the ordinary
`create_classification_model()` ConvNeXt descriptor and the real backbone.
Batch preparation/normalization and thin training bindings remain the next
integration work, followed by real-image and matched PyTorch measurements.

The following narrative records the successive implementation and measurement
slices; the table above describes their current combined scope.

The first concrete cross-surface milestone is the image-transform path in
`st-vision`, not a general claim that every backend is interchangeable.
Contiguous resize, crop, and sampled horizontal-flip stages now use one WGPU
geometry sequence with one upload and terminal readback on native platforms.
The seeded CPU route remains the behavioral reference; Python exposes an
explicit `TransformPipeline.enable_wgpu()` opt-in and `disable_wgpu()` fallback.

The browser geometry slice now has the same Rust-owned semantics. Transform
WGSL is embedded in the WASM build; `VisionTransformPipeline.createGpu(seed)`
uses the `st-vision` planner and runs adjacent resize, center-crop, and sampled
horizontal-flip stages with one upload and one **async terminal readback**.
`createCpu(seed)` is the in-browser Rust CPU reference. The real-Chrome parity
fixture compares 12 seeded frames, rejects invalid geometry, and records the
actual Rust runtime adapter. A failed geometry run leaves the image and flip
seed unchanged on both CPU and GPU, including a subsequent valid retry.
Native Rust and fresh-wheel Python tests cover
the same sequence. The bounded result and replay instructions live in
[`benchmarks/results/2026-09-25-vision-wasm-async/`](../benchmarks/results/2026-09-25-vision-wasm-async/README.md).

The next slice now hands a transformed CHW image directly to the existing
guarded `ResidentTensor`. Rust `apply_geometry_resident`, Python
`apply_resident(image, device)`, and browser `applyResident(...)` make the
handoff explicit. A browser `Sequential` forward consumes the image without
an intermediate readback; only its output is mapped. The GPU values are
checked for non-finite results before they enter the tensor/NN contract.
The bounded browser result is in
[`benchmarks/results/2026-09-25-vision-resident-handoff/`](../benchmarks/results/2026-09-25-vision-resident-handoff/README.md).

The next slice processes a homogeneous NCHW batch through the same Rust-owned
geometry planner, with one image-data upload, one transform submission, per-image
seeded flip choices, and a guarded resident output. Python
`apply_resident_batch(images, device)` and browser
`applyResidentBatch(n, c, h, w, data)` expose the same route. The browser
fixture connects it both to NN inference and to one supervised `GraphLearner`
update without a geometry-to-learner readback. A six-condition shape/batch
sweep compares this route with per-image resident execution; full conditions
and limits are in
[`benchmarks/results/2026-09-25-vision-resident-batch/`](../benchmarks/results/2026-09-25-vision-resident-batch/README.md).

On one 3x128x128 Chrome case, the full WebGPU geometry route with terminal
readback was slower than WASM CPU. For geometry followed by a GPU NN layer,
the matched resident handoff was faster than the GPU readback/reupload route
on that same host and shape. The batch sweep probes a different comparison:
one batch versus N separately resident images, including all NN readbacks.
Neither establishes general superiority or full-model training throughput.
Normalize and ColorJitter are not in the browser GPU geometry interface and
must not silently fall back. Real image-model training, mixed-size batches,
and matched cross-framework comparisons remain open gates.

The next Rust-only model slice makes the ConvNeXt-style backbone trainable:
block residuals, stage downsampling, final normalization, and the stem now
propagate gradients into their parameters. A two-image classification example
exercises the full backbone and head; finite differences check input and
parameter gradients through a downsampling stage. Host convolutions
(`Conv1d` through `Conv4d`, plus `Conv6da`) and the four affine-normalization
variants now use the supplied loss gradient without another per-layer batch
average. Mean-loss VJP and duplicated-batch tests cover this migration.
`st-vision/wgpu` now enables `st-nn/wgpu` when the optional NN dependency is
present. ConvNeXt blocks now use a channel-wise `DepthwiseConv2d` with compact
weights, a CPU reference VJP, and numerical gradient checks. This changes the
old dense block weight shape, so old checkpoints require an explicit migration
rather than a silent reload. Its dedicated WGPU forward kernel has CPU parity
on a native adapter and emits the selected backend or CPU fallback in operation
metadata. It still uploads inputs and weights and reads back each result; the
backward pass is CPU-only. A matched Apple M4 host CPU/WGPU timing sweep is
recorded in `benchmarks/results/2026-09-26-vision-depthwise-m4.md`: the small
shape is slower on WGPU, while larger shapes improve on that one host. The
exploratory Auto threshold therefore avoids the measured small-workload range.
WASM compilation is checked; the synchronous host-Tensor WGPU route explicitly
rejects browser execution before dispatch because browser readback is async.
The browser CPU reference remains available. A separate `ResidentTensor`
depthwise route now accepts NCHW images with resident `[C, KH, KW]` weights
and `[C]` bias. Native Rust, Python `WgpuTensor`, and browser `WgpuTensor`
use the same GPU operation and deferred validity guard. It connects the
existing image-transform resident handoff to depthwise and an activation
without an intermediate readback; browser snapshot mapping remains async.
The bounded browser parity and paired handoff timings are in
`benchmarks/results/2026-09-27-vision-resident-depthwise.md`. The next Rust
slice connects the original `DepthwiseConv2d` parameter owner to this
primitive and composes one `ConvNeXtBlock` forward entirely on the resident
device: depthwise, token layout packing, LayerNorm, MLP, and residual. The
parameter cache refreshes after host-side updates, and native GPU tests match
the CPU block and the vision-transform handoff. The next slice extends this
to the Rust `ConvNeXtBackbone`: a guarded NCHW dense `Conv2d` primitive powers
its model-owned stem and stage downsampling, and the blocks and final norm
stay resident through the final snapshot. The primitive is also exposed as
Python and browser `WgpuTensor.conv2d`, with the same Rust shape and validity
checks; this does not expose the model-owned ConvNeXt API. A tiny native GPU
test checks whole-backbone CPU parity, non-contiguous input, and cache refresh after
parameter updates. A Rust example connects the existing image-transform
handoff to this full forward. This is **not** a Python/WASM ConvNeXt model,
GPU backward, or a demonstrated full-model speedup. A bounded Apple M4
[synthetic backbone probe](../benchmarks/results/2026-09-27-vision-resident-convnext-m4.md)
found a narrow LayerNorm WGPU bottleneck and improved it with a 32-lane
dispatch, but WGPU remains slower than CPU on the tested full-model shapes.
The browser WebGPU LayerNorm fixture passes widths 16, 32, and 33, but does
not measure browser ConvNeXt throughput.
Next gates are resident training, finishing the VJP migration for specialized
`st-nn` layers, and matched real-dataset accuracy/throughput.
The first backward slice now provides GPU-resident depthwise input, weight,
and bias VJPs with one shared validity guard. The model-owned `DepthwiseConv2d`
exposes them without an implicit host update; native CPU parity and a
[Chrome WebGPU check](../benchmarks/results/2026-09-28-vision-depthwise-vjp-m4.md)
cover the bounded contract. A second slice composes that primitive with the
existing Rust graph-autograd VJPs for LayerNorm, Linear, and GELU, plus the
residual branch. `ConvNeXtBlock::vjp_resident` now returns resident input and
eight parameter gradients in `Module` order, rebuilding its frozen tail graph
when host parameters change. Native GPU and
[Chrome WebGPU parity](../benchmarks/results/2026-09-28-vision-convnext-vjp-contract.md)
cover a bounded block. The next slice adds a deterministic dense `Conv2d` VJP
and composes it with the block VJPs, stem, stage downsampling, and final norm.
`ConvNeXtBackbone::vjp_resident` now returns an input gradient and every
parameter gradient in `Module` order. Native GPU and
[Chrome WebGPU parity](../benchmarks/results/2026-09-30-vision-convnext-backbone-vjp-contract.md)
cover one small two-stage backbone. There is still no GPU-owned optimizer
state, model update, or measured real-dataset training advantage. Those remain
the next gates, not inferred from isolated backward correctness.

</details>

## Documentation & Learning
- **Curated entry points.** Expand the README "Quick Start" into a set of versioned walkthroughs that mirror the typical paths: Rust-only, Python wheel, and the collaborative canvas. Each walkthrough should end with a runnable example and explicit troubleshooting steps.
- **Concept glossaries.** Promote the existing conceptual notes in `docs/` (e.g. _Quantum Reality Acceleration_) into a structured glossary. Call out how each concept maps onto concrete crates such as `st-core`, `st-tensor`, or `st-kdsl`.
- **Migration stories.** Publish recipes that translate common PyTorch training loops into SpiralTorch idioms. Start with supervised trainers and checkpoint loading; follow up with RL env bridges once telemetry hooks stabilize.

## Tutorials & Samples
- **Device-aware notebooks.** Author Jupyter/Polars notebooks that demonstrate switching between CPU, WGPU, MPS, and CUDA via feature flags. Surface the performance and observability differences using the telemetry APIs.
- **Canvas-first demos.** Capture the collaborative canvas flows with scripted recordings. Pair them with the TypeScript bindings to show how live annotations or metric overlays are powered.
- **Edge-ready bundles.** Produce lightweight binaries that highlight Rust's zero-runtime-cost deployment story. Target single-board ARM devices first, then progressively integrate GPU backends where available.

## Community & Contribution
- **Guides for contributors.** Maintain `CONTRIBUTING.md` with style, testing, and security expectations. Reference the AGPL obligations, including how derivative works should publish sources.
- **Discussion rituals.** Use scheduled office hours in GitHub Discussions to triage backend regressions and feature proposals. Publish outcomes as short changelog snippets.
- **Issue labeling.** Introduce labels for backend coverage (`backend:cuda`, `backend:wgpu`, etc.), documentation, and canvas/UI work so that contributors can filter effortlessly.

## Integrations & Distribution
- **ONNX/interop matrix.** Track ONNX export/ingest progress in [onnx_interop.md](onnx_interop.md) and enumerate the missing operators for parity with PyTorch 2.x. Track the maturity per backend for both forward and backward passes.
- **Model hub prototype.** Sketch a minimal artifact registry backed by object storage. Ensure AGPL license metadata and reproducibility manifests are embedded.
- **Tuning pipelines.** Bundle sample configurations for Optuna and Ray Tune. Show how the hypergrad tape and planner can be wired in without Python-side tensor copying.
- **Compatibility playbook.** Maintain the [Compatibility Strategy](compatibility_strategy.md) as a living guide for PyTorch/TensorFlow migrations, including API diff tables, operator coverage, and hybrid deployment recipes.
- **Language expansion.** Land the shared `spiraltorch-sys` ABI crate and pilot Julia and Go bindings following the [integration strategy](julia_go_integration.md). Target inference-first workflows, document ownership semantics, and formalize support levels once telemetry and CI coverage land.

## Measuring Progress
- Establish a living changelog that calls out which roadmap items advanced in each release.
- Track documentation coverage by counting the number of walkthroughs with validated code samples.
- Define a backend matrix (see `docs/backend_matrix.md`) to ensure the supported feature set stays transparent for each device stack.

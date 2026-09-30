# SpiralTorch Ecosystem Roadmap

SpiralTorch already offers a rich Rust-first runtime, a shared hypergrad tape for the Python bindings, and a TypeScript-powered collaboration canvas. This document captures the near-term ecosystem priorities so contributors can converge on the same themes while the core crates continue to evolve.

## Restarted execution slice: vision across Rust, Python, and browser

Current scope of the vision execution path:

| Milestone | Rust | Python | Browser/WASM |
| --- | --- | --- | --- |
| Seeded resident geometry batch | Public API | Public API | Public API, async snapshots |
| Model-owned resident ConvNeXt forward | `Module` API | Not exposed | Exercised by Rust WASM VJP fixture; no model binding |
| Full-backbone resident VJP | `ConvNeXtBackbone::vjp_resident` | Not exposed | Rust WASM parity fixture; no model binding |
| ConvNeXt-owned resident parameter update | `compile_resident_training`, plain SGD | Not exposed | Rust WASM learning fixture; no model binding |
| ConvNeXt plain-SGD model checkpoint/resume | Public API, explicit host handoff | Not exposed | Rust WASM resume fixture and portable JSON; no model binding |
| Real ConvNeXt classifier common inference entry | `ConvNeXtClassifier`, `create_classification_model` | Ordinary factory with `nn`; inference only | Rust WASM common-entry fixture; no JS model binding |
| Classifier-owned resident CE/VJP/SGD/checkpoint | One owner for backbone and head | Not exposed | Shared Rust WASM learning/resume fixture; no JS model binding |
| Resident image normalization and DataLoader handoff | Open | Open | Open |
| Matched real-image training quality and throughput | Open | Open | Open |

The compiled ConvNeXt training path now connects model-owned gradients to
resident parameter updates with version checks, stale-gradient rejection and
all-parameter acceptance. Native and browser fixtures cover consecutive steps,
rejected steps preserving all weights, and a valid retry. Portable plain-SGD
model checkpoints now preserve weights, architecture and attempted-update
revision, with explicit mapping and fresh restored owner identities. Data
cursors, RNGs, rate schedules and trainer policy remain caller-owned state.
The common public ConvNeXt entry now owns the real backbone, channel-preserving
global-average pooling, and Linear classification head. Classifier learning uses
one resident parameter owner and one acceptance/revision clock for all weights.
Its checkpoint can return the learned model to the ordinary inference interface.
Python's `nn` feature enables this same Rust factory; the WASM fixture exercises
the same Rust model, not a JavaScript model binding. See
[the classifier contract result](../benchmarks/results/2026-10-01-convnext-classifier-contract.md).

### Next Rails And Exit Gates

1. **Resident real-image input.** Connect batching and Normalize to the existing
   geometry/classifier/loss/update path. Preserve seed/retry semantics and avoid
   hidden intermediate readbacks. `VisionBatch::stack()` remains a host matrix;
   the resident geometry API still rejects non-geometry transforms.
2. **Thin training clients.** Expose the Rust-owned classifier, explicit
   checkpoint mapping, and restart to Python and JavaScript without duplicating
   training rules or implying that a submitted update has been accepted.
3. **Matched real-data evidence.** Check numerical parity against the same
   architecture/weights/data/loss in PyTorch, then measure held-out quality,
   transfer-inclusive throughput, memory, and restart equivalence. Synthetic
   loss decrease and kernel timings are not substitutes for this gate.

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

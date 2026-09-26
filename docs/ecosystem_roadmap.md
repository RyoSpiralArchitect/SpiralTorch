# SpiralTorch Ecosystem Roadmap

SpiralTorch already offers a rich Rust-first runtime, a shared hypergrad tape for the Python bindings, and a TypeScript-powered collaboration canvas. This document captures the near-term ecosystem priorities so contributors can converge on the same themes while the core crates continue to evolve.

## Restarted execution slice: vision across Rust, Python, and browser

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
`benchmarks/results/2026-09-27-vision-resident-depthwise.md`. This is a
forward primitive, **not** an automatically resident ConvNeXt graph or GPU
backward pass. Next gates are resident model composition and training,
finishing the VJP migration for specialized `st-nn` layers, and matched
real-dataset accuracy/throughput.

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

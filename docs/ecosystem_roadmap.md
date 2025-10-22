# SpiralTorch Ecosystem Roadmap

SpiralTorch already offers a rich Rust-first runtime, a shared hypergrad tape for the Python bindings, and a TypeScript-powered collaboration canvas. This document captures the near-term ecosystem priorities so contributors can converge on the same themes while the core crates continue to evolve.

## Documentation & Learning
- **Curated entry points.** Expand the README "Quick Start" into a set of versioned walkthroughs that mirror the typical paths: Rust-only, Python wheel, and the collaborative canvas. Each walkthrough should end with a runnable example and explicit troubleshooting steps.
- **Concept glossaries.** Promote the existing conceptual notes in `docs/` (e.g. _Quantum Reality Acceleration_) into a structured glossary. Call out how each concept maps onto concrete crates such as `st-core`, `st-tensor`, or `st-kdsl`.
- **Migration stories.** Publish recipes that translate common PyTorch training loops into SpiralTorch idioms. Start with supervised trainers and checkpoint loading; follow up with RL env bridges once telemetry hooks stabilize.

## Tutorials & Samples
- **Device-aware notebooks.** Author Jupyter/Polars notebooks that demonstrate switching between CPU, WGPU, MPS, and CUDA via feature flags. Surface the performance and observability differences using the telemetry APIs.
- **Canvas-first demos.** Capture the collaborative canvas flows with scripted recordings. Pair them with the TypeScript bindings to show how live annotations or metric overlays are powered.
- **Edge-ready bundles.** Produce lightweight binaries that highlight Rust's zero-runtime-cost deployment story. Target single-board ARM devices first, then progressively integrate GPU backends where available.

## Community & Contribution
- **Guides for contributors.** Add `CONTRIBUTING.md` with style, testing, and security expectations. Reference the AGPL obligations, including how derivative works should publish sources.
- **Discussion rituals.** Use scheduled office hours in GitHub Discussions to triage backend regressions and feature proposals. Publish outcomes as short changelog snippets.
- **Issue labeling.** Introduce labels for backend coverage (`backend:cuda`, `backend:wgpu`, etc.), documentation, and canvas/UI work so that contributors can filter effortlessly.

## Integrations & Distribution
- **ONNX/interop matrix.** Document the current ONNX export/ingest path and enumerate the missing operators for parity with PyTorch 2.x. Track the maturity per backend for both forward and backward passes.
- **Model hub prototype.** Sketch a minimal artifact registry backed by object storage. Ensure AGPL license metadata and reproducibility manifests are embedded.
- **Tuning pipelines.** Bundle sample configurations for Optuna and Ray Tune. Show how the hypergrad tape and planner can be wired in without Python-side tensor copying.
- **Compatibility playbook.** Maintain the [Compatibility Strategy](compatibility_strategy.md) as a living guide for PyTorch/TensorFlow migrations, including API diff tables, operator coverage, and hybrid deployment recipes.
- **Language expansion.** Land the shared `spiraltorch-sys` ABI crate and pilot Julia and Go bindings following the [integration strategy](julia_go_integration.md). Target inference-first workflows, document ownership semantics, and formalize support levels once telemetry and CI coverage land.

## Measuring Progress
- Establish a living changelog that calls out which roadmap items advanced in each release.
- Track documentation coverage by counting the number of walkthroughs with validated code samples.
- Define a backend matrix (see `docs/backend_matrix.md`) to ensure the supported feature set stays transparent for each device stack.


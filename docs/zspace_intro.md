# Navigating Z-space in SpiralTorch

Z-space is SpiralTorch's shared manifold for tensors, gradients, and telemetry. Every
kernel, scheduler pass, and learning loop interprets data relative to a Z-frame so
spectral structure, topology, and temporal echoes stay coherent across the stack.
This document surveys how the implementation assembles that manifold, the APIs you
use to work with it, and the surrounding tooling that keeps Z-space sessions
inspectable and portable.

## Core building blocks

### Atlas, frames, and indices

The foundational types that describe the manifold live in the core crates:

- `ZAtlas`, `ZFrame`, and `ZIndex` in `crates/st-core` define the spectral bands,
  sheet topology, and temporal echoes that constitute a session's coordinate
  system.
- The Python bindings forward those descriptors automatically. Tensors moving
  through `spiraltorch` operations carry their frame metadata, so projecting back
  into plain layouts (for example via `spiraltorch.nn.Conv2d` or
  `ZSpaceProjector`) preserves alignment without manual bookkeeping.

### Gradient roundtables

Gradients and metric streams are partitioned into Above/Here/Beneath bands with
`st_core::ops::zspace_round`. The `SpectralFeatureSample` helper extracts sheet
confidence, curvature, spin, and energy from Z-space slices before
`RoundtableAssignment` feeds the cooperative scheduler. This keeps the banded
roundtable that drives cooperative optimisation in sync with the latest spectral
statistics.

### Coherence and sequencing

`st-nn` extends the core descriptors with Z-space-aware layers and sequencers.
`StableZSpaceProjector`, `ZSpaceMixer`, and `ZSpaceCoherenceSequencer` expose
forward/backward pipelines that understand warped lattices, while
`zspace_coherence::run_zspace_learning_pass` powers PSI synchroniser learning
from Rust and Python call-sites.

## Runtime state and caching

Each SpiralTorch session owns a Z-space atlas. When kernels lift activations into
that space the runtime consults the frame signature to reuse tuned kernels,
allocate residency, and emit telemetry. The cache keys incorporate Z-frame
metadata so code generation, autotuning artefacts, and checkpointing remain
compatible even as the atlas warps for a new task.

Python callers can inspect the active state with `st.zspace.session()`:

```python
import spiraltorch as st

with st.zspace.session() as session:
    frame = session.frame
    print("bands", frame.bands)
    print("sheets", frame.sheets)
    print("echoes", frame.temporal.echo_count)
```

`st.zspace.render_atlas()` renders a Matplotlib snapshot of the current atlas and
`st.zspace.sample_field()` lets you probe spectral density and homology labels for
individual cells, making warp changes visible during experiments.

## Working with Z-space metrics

### Encoding and decoding

The convenience namespace `st.z[...]` routes text directly through the
`LanguageWaveEncoder` so you can materialise Z-space tensors with one-liners. For
structured results, `spiraltorch.decode_zspace_embedding` and
`spiraltorch.zspace_eval` convert latent vectors or Mellin projections back into
named metrics and diagnostic points.

### Metrics, partials, and telemetry

`spiraltorch.z_metrics` normalises speed, memory, stability, drift-response, and
gradient series into canonical Z-space metrics. You can collect partial
observations with `st.z.partial(...)` which emits `ZSpacePartialBundle`
instances carrying weights, origin tags, and optional telemetry payloads.

`blend_zspace_partials`, `infer_from_partial`, and `infer_with_partials` fuse
partials with latent states, while `ZSpaceTrainer` and `stream_zspace_training`
provide an optimiser-friendly wrapper that incrementally adapts a posterior to
incoming metrics.

The inference helpers accept structured telemetry. Pass dicts, `ZSpaceTelemetryFrame`
instances, or the telemetry captured inside a `ZSpacePartialBundle`; everything
is flattened and merged automatically before the posterior update so PSI health
metrics and canvas energy stay aligned with the atlas.

### Importing external weights

Warm-starting from other ecosystems hinges on the DLPack bridges:

- `spiraltorch.weights_partial_from_dlpack` and
  `spiraltorch.weights_partial_from_compat` derive partial bundles from imported
  tensors.
- `spiraltorch.infer_weights_from_dlpack` projects those partials through the
  posterior, blending optional PSI telemetry in the same step.

These helpers integrate with the compat adapters advertised in the README so a
PyTorch, TensorFlow, or JAX model can contribute to a Z-space session without
rewriting the training loop.

## PSI synchroniser learning

The PSI module exposes multi-branch learning bundles tuned for Z-space coherence.
`st.psi.run_zspace_learning(...)` funnels Atlas fragments, heatmaps, ZPulse
snapshots, and Golden directives into the Rust sequencer. The resulting summary
and telemetry keep distributed learners and `golden` retrievers aligned without
leaving the cooperative runtime.

## Canvas and vision feedback

SpiralTorchVision exposes `CanvasProjector::emit_zspace_patch` so WebGPU canvases
can fold their state back into the fractal scheduler. The projector returns both
the RGBA buffer and Z-space-compatible patches, letting browser or Rust call-sites
ship telemetry into the same atlas used by language and PSI modules. Pair it with
`spiraltorch.z_space_barycenter` to blend canvas chart priors directly into the
roundtable before handing gradients to `Hypergrad` or `Realgrad` tapes.

## Further reading

- [Z-space drift-response linguistics](docs/drift_response_linguistics.md)
- [General relativity couplings inside Z-space](docs/general_relativity_zspace.md)
- [Z-space inference autopilot](docs/quantum_reality_acceleration.md)
- Rust sources around Z-space operators:
  - `crates/st-core/src/ops/zspace_round.rs`
  - `crates/st-nn/src/layers/zspace_projector.rs`
  - `crates/st-nn/src/zspace_coherence/`
- Python inference helpers in `bindings/st-py/spiraltorch/zspace_inference.py`

With these components in view you can inspect, extend, and stabilise Z-space
pipelines while keeping imported checkpoints, PSI telemetry, and Canvas feedback
aligned with SpiralTorch's cooperative scheduler.

## Quick-start: Z-RBA uncertainty head

```python
import spiraltorch as st
from spiraltorch.nn import ZRBA, ZRBAConfig
from spiraltorch.nn import ZMetricWeights
from spiraltorch.nn import ZTelemetryBundle

with st.zspace.session() as sess:
    x, y = load_structured_batch()
    cfg = ZRBAConfig(
        d_model=128,
        n_heads=4,
        cov_rank=8,
        metric=ZMetricWeights(w_band=1.0, w_sheet=0.5, w_echo=0.2),
        ard=True,
        gate_momentum=0.05,
        gate_seed=42,
        gate_use_expected=True,
    )
    model = ZRBA(cfg)

    z_tensor = st.z.embed(x)
    stats = st.zspace.sample_field()
    yhat, cov, telemetry = model.forward(z_tensor, sess.frame, stats)

    metrics = telemetry.metrics(yhat.mu, yhat.sigma, targets=y, pin=0.95, indices=yhat.indices)
    bundle = telemetry.bundle_metrics(metrics)
    st.zspace.telemetry.log_uq(bundle.to_json())
```

`gate_use_expected=True` keeps the residual path deterministic by default, while
setting it to `False` switches the gate to draw Beta samples per forward pass.
The `gate_momentum` hyper-parameter tunes how fast the running mean/variance of
the gate adapt to new spectral statistics, which is useful when Roundtable bands
shift under rapid Atlas warps.

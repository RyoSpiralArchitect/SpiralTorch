# Temporal Z-dynamics video pipelines

SpiralTorch Vision now includes a streaming video stack that lifts FFmpeg
streams into Z-space dynamics with atlas telemetry baked in. The workflow is
anchored by three new building blocks:

- [`ZSpaceStreamFrame`](../../crates/st-vision/src/lib.rs) accepts planar
  tensors plus optional [`ChronoSnapshot`] metadata and keeps the shapes aligned
  for volumetric fusion.
- [`ChronoSnapshot`](../../crates/st-vision/src/lib.rs) records the latest
  `ChronoSummary` along with the timestep so temporal transforms can blend
  weights deterministically.
- [`VideoPipeline`](../../crates/st-vision/src/video/mod.rs) manages FFmpeg
  decoding, motion-aware filtering, resonance envelopes, and atlas emission in a
  single iterator-style API.

## Decoding and streaming

Implement [`FfmpegBinding`](../../crates/st-vision/src/video/mod.rs) inside your
`bindings/` crate to expose a `decode_next()` method that yields `(timestamp,
Tensor)` pairs. Wrap it with [`FfmpegDecoder`] and hand the decoder to a
[`VideoPipeline`]:

```rust
use st_vision::{FfmpegDecoder, VideoPipeline, VideoPipelineConfig};
use my_bindings::FfmpegSession; // implements st_vision::FfmpegBinding

let session = FfmpegSession::open("clip.mov")?;
let decoder = FfmpegDecoder::new(session);
let mut pipeline = VideoPipeline::new(decoder, VideoPipelineConfig::default());
```

Each `pipeline.next()?` call returns a [`VideoPipelineOutput`] containing the
latest [`AtlasFrame`], motion embedding, resonance envelope, and the fused
[`StreamedVolume`]. The atlas metrics automatically include:

- `z.motion_energy` — mean absolute delta between consecutive frames.
- `z.resonance_energy` — temporal envelope energy across the flattened volume.
- `z.weight_entropy` — entropy of the smoothed depth attention profile.

These metrics live in the `temporal` district so dashboard consumers can plot
or threshold them alongside other atlas telemetry.

## Temporal transforms

The pipeline performs several Z-space transforms using `st-tensor` operators:

1. `MotionEmbeddingFilter` computes frame deltas with `Tensor::sub()` and scales
   the result to emphasise motion.
2. `ResonanceEnvelope` flattens each [`ZSpaceVolume`] into a tensor, blends it
   with an exponential moving average via `Tensor::scale()`/`Tensor::add()`, and
   exposes the accumulated envelope for downstream analysis.
3. `TemporalResonanceBuffer` smooths per-depth energy before collapse so the
   resulting weights capture persistent motion as well as transient spikes.

[`SpectralWindow::hann()`](../../crates/st-vision/src/lib.rs) is applied to the
volume on every iteration to provide a frequency-oriented view of the depth
axis. The resulting spectrum, depth energy, and smoothed weights are surfaced in
[`ZDynamicsAnnotation`] for any consumer that needs raw telemetry.

## Integration notes

- The first frame seeds motion and resonance buffers; subsequent frames apply
  the configured `temporal_alpha` and `resonance_decay` coefficients.
- `ChronoSnapshot` timestamps are propagated into the atlas frame, so any
  existing maintainer tooling can continue to rely on atlas chronology.
- When the incoming depth changes, the temporal buffers reset automatically to
  prevent stale weights from contaminating the new stack.

See `crates/st-vision/tests/video_pipeline.rs` for a synthetic clip example that
covers motion embeddings, entropy metrics, and chrono propagation end-to-end.

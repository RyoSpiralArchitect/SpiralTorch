# SpiralTorchVision Guide

SpiralTorchVision extends SpiralTorch's native Z-space capabilities while staying compatible with the standard TorchVision stack. This guide inventories the core features that TorchVision ships today and outlines how SpiralTorchVision builds on top of them.

## TorchVision feature overview
- **Datasets**: Canonical tasks for image classification, detection, semantic/instance segmentation, optical flow, stereo matching, image pairs, captioning, video classification, and video prediction through `torchvision.datasets`.
- **Models**: Classification architectures ranging from AlexNet to Vision Transformers, quantization-ready classifiers, semantic segmentation, detection/instance segmentation/keypoint estimation, video classification, and optical flow modules with pre-trained weights.
- **Transforms v2**: A unified preprocessing pipeline under `torchvision.transforms.v2` that handles images, videos, bounding boxes, masks, and keypoints with a consistent API while retaining v1 compatibility and higher throughput.
- **TVTensors**: Tensor subclasses such as Image, Video, and BoundingBoxes that preserve metadata and enable automatic dispatch.
- **Utilities**: Visualization and persistence helpers in `torchvision.utils`, including `draw_bounding_boxes`, `draw_segmentation_masks`, `make_grid`, and `save_image`.
- **Custom operators**: TorchScript-friendly primitives in `torchvision.ops` for NMS, RoI ops, box algebra, detection losses, convolution/DropBlock/SE blocks, and more.
- **IO**: Decoding for JPEG/PNG/WEBP/GIF/AVIF/HEIC, encoding for JPEG/PNG, and (deprecated) video IO with fast tensor conversion.
- **Feature extraction utilities**: Tools like `create_feature_extractor` for intermediate feature capture, visualization, transfer learning, FPN assembly, and other advanced uses.

## SpiralTorchVision expansion points
- **Spectral Z-attention**: `SpectralWindow` functions (Hann, Hamming, Blackman, Gaussian) now modulate how `VisionProjector` collapses `ZSpaceVolume` slices, letting you pre-emphasize perceptual frequencies before feeding tensors to TorchVision models.
- **ZSpaceVolume / VisionProjector**: A volumetric representation that accumulates resonant features along the Z-axis and collapses them into tensors. It ingests intermediate activations from TorchVision models and bridges them into SpiralTorch's Z-space analyzers.
- **Differential Resonance integration**: Combining with `st_tensor::DifferentialResonance` to reproject spatiotemporal features gathered from TorchVision networks back into SpiralTorch's resonant frames.
- **Temporal resonance accumulation**: `ZSpaceVolume::accumulate` and `TemporalResonanceBuffer` perform exponential moving averages across frames so `VisionProjector::project_with_temporal` can mix historical attention with new resonance in real time.
- **Multi-view Z-fusion**: Register camera descriptors with `MultiViewFusion` so `VisionProjector::project_multi_view` can weight Z slices as viewpoints, modulating attention with orientation-aware biases before collapse.
- **Generative resonance coupling**: Feed `ZSpaceVolume` slice statistics into a `ResonanceGenerator` backed by `SpiralRnn` so synthetic `DifferentialResonance` fields can drive projection without an external conductor.
- **Z-space super-resolution & diffusion**: Upsample volumes with `InterpolationMethod`, `ZSpaceVolume::interpolate`, and `ZSpaceVolume::upscale`, then refine or hallucinate detail with `ZDiffuser` and the stochastic `ZDecoder` latent bridge.
- **Video stream projection**: Pipe temporally ordered volumes through `VideoStreamProjector` to mix diffusion, super-resolution, generative resonance, and temporal smoothing while tracking previous feedback.
- **Long-term integrations**: Leverage TorchVision datasets/transforms as inputs while adding Z-space-native losses, visualisation tools, and SpiralTorch-specific model heads.
- **Long-term integrations**: Leveraging TorchVision datasets/transforms as inputs while adding Z-space-native losses, visualization tools, and SpiralTorch-specific model heads.
- **Modular vision backbones**: `st_vision::models` ships ergonomic ResNet, ViT, and ConvNeXt backbones built on the `st-nn` module trait. They expose configuration structs, state-dict interop, and forward passes tuned for SpiralTorch tensors.

### Backbone quickstart

Instantiate and run a pretrained-friendly backbone directly from Rust. The modules implement `st_nn::module::Module`, so optimizers, telemetry, and serialization utilities interoperate out of the box:

```rust
use st_tensor::Tensor;
use st_vision::models::{ResNetBackbone, ResNetConfig};

let config = ResNetConfig {
    input_hw: (224, 224),
    stage_channels: vec![64, 128, 256, 512],
    block_depths: vec![2, 2, 2, 2],
    ..Default::default()
};
let backbone = ResNetBackbone::new(config)?;
let input = Tensor::random_normal(1, 3 * 224 * 224, 0.0, 1.0, Some(0))?;
let embedding = backbone.forward(&input)?;
assert_eq!(embedding.shape().1, backbone.output_features());
```

The `ViTBackbone` exports `load_weights_json`/`load_weights_bincode` helpers, accepts patch/grid tweaks, and emits CLS-token embeddings by default. `ConvNeXtBackbone` mirrors ConvNeXt-T style stage depths, returning flattened feature maps ready for detection heads or Z-space projection.

Need a CIFAR-style network? `ResNetConfig::resnet56_cifar(true)` wires a 56-layer backbone with SpiralTorch's learnable skip scalers and a default **slip schedule** that eases each residual bridge in before letting it run at full strength. Override the schedule to taste by swapping in your own `SkipSlipSchedule`:

```rust
use st_vision::models::{ResNetConfig, ResNetBackbone, SkipSlipSchedule};

let mut config = ResNetConfig::resnet56_cifar(true);
config.skip_slip = Some(
    SkipSlipSchedule::linear(0.25, 1.0)
        .per_stage()
        .with_power(1.2),
);
config.skip_init = 0.95; // softly dampen residuals at init
let mut resnet56 = ResNetBackbone::new(config)?;
let logits = resnet56.forward(&input)?;
// gradients propagate into each skip gate when calling backward(...)
```

### Temporal resonance accumulation

Temporal continuity lets SpiralTorchVision respond to motion and lingering cues without reprocessing an entire video buffer. Each frame updates a `ZSpaceVolume` in-place with `accumulate`, applying an exponential moving average (`alpha` near `0.2` preserves the past, `alpha` near `1.0` chases the latest frame). The resulting volume feeds a `TemporalResonanceBuffer`, which smooths the depth attention profile before collapse:

```rust
let mut ema_volume = ZSpaceVolume::from_slices(&frame_zero_slices)?;
let mut temporal = TemporalResonanceBuffer::new(0.25);

for frame in sequence {
    let next_volume = ZSpaceVolume::from_slices(&frame.slices)?;
    ema_volume.accumulate(&next_volume, 0.25)?;

    let projection = projector.project_with_temporal(
        &ema_volume,
        &frame.resonance,
        &mut temporal,
    )?;
    // feed projection into downstream TorchVision modules here
}
```

`TemporalResonanceBuffer::apply` keeps track of how many frames influenced the profile, so you can reset when a scene cut happens or when sensors disagree. Pairing it with spectral windows gives a dual attention sweep—frequency over Z, inertia over time.

### Multi-view Z-fusion

Multi-camera rigs treat the Z-axis as a viewpoint stack. Describe each camera with a `ViewDescriptor`, build a `MultiViewFusion`, and project the stacked volume with view-aware weights. The fusion helper normalises alignment biases (dot product between a view's forward vector and the configured focus direction) and combines them with baseline weights so attention favours cameras that cover the region of interest:

```rust
let views = vec![
    ViewDescriptor::new("front", [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ViewDescriptor::new("right", [0.0, 0.0, 0.0], [1.0, 0.0, 0.25]).with_baseline_weight(1.2),
    ViewDescriptor::new("up", [0.0, 0.0, 0.0], [0.0, 1.0, 0.8]),
];
let fusion = MultiViewFusion::new(views)?
    .with_focus_direction([0.2, 0.1, 1.0])
    .with_alignment_gamma(1.5);

let multi_view_volume = ZSpaceVolume::from_slices(&view_slices)?; // Z = view index
let weights = projector.depth_weights_multi_view(&fusion, &multi_view_volume, &resonance)?;
let fused_projection = projector.project_multi_view(&fusion, &multi_view_volume, &resonance)?;

let mut temporal = TemporalResonanceBuffer::new(0.35);
let fused_temporal = projector.project_multi_view_with_temporal(
    &fusion,
    &multi_view_volume,
    &resonance,
    &mut temporal,
)?;
```

Multi-view temporal buffers reuse the same decay logic, so you can smooth viewpoint attention as sensors hand off dominance (e.g., during turns or occlusions). Downstream TorchVision modules receive a fused 2D tensor while retaining inspectable per-view weights.

### Generative resonance coupling

With spectral, temporal, and multi-view conditioning in place, the next step is to let SpiralTorchVision **synthesize** its own resonance fields. `ZSpaceVolume::slice_profile` summarises every slice with means, standard deviations, and energy estimates, while `ZSpaceVolume::total_energy` reports the global activation budget. Feed those statistics into a `ResonanceGenerator`—a thin wrapper around `st_nn::SpiralRnn`—and you receive a fully populated `DifferentialResonance` without querying an external conductor:

```rust
let mut generator = ResonanceGenerator::new("vision-loop", 12, volume.depth())?;
let chrono_summary = build_chrono_summary(); // produce a ChronoSummary from your telemetry window
let mut atlas = AtlasFrame::new(chrono_summary.latest_timestamp);
atlas.z_signal = Some(0.7);
atlas.collapse_total = Some(1.05);

let synthetic = generator.generate(
    &volume,
    &projector,
    Some(&chrono_summary),
    Some(&atlas),
    last_resonance.as_ref(),
)?;
let fused = projector.project(&volume, &synthetic)?;
```

Passing the previous `DifferentialResonance` back into `generate` closes the loop, allowing the RNN to refine its latent state over time. Because `ResonanceGenerator` exposes `rnn_mut`, you can attach hypergrads or otherwise fine-tune the SpiralRNN as part of a larger training run. The resulting projections inherit spectral, temporal, and multi-view behaviour from `VisionProjector`, but now the resonance driving that collapse is born from the same pipeline.

### Z-space super-resolution, diffusion, and decoding

`InterpolationMethod` controls how intermediate slices are synthesised while traversing depth. Call `ZSpaceVolume::interpolate`
to densify the Z-axis and `ZSpaceVolume::upscale` to bilinearly expand each slice's spatial resolution. Couple the output with a
`ZDiffuser` to bloom sparse activations without erasing structural cues:

```rust
let hi_res = volume
    .interpolate(InterpolationMethod::Cubic)?
    .upscale(2)?;

let diffuser = ZDiffuser::new(2, 0.35);
let softened = diffuser.diffuse(&hi_res)?;
```

For generative workflows, `ZDecoder` turns latent tensors into volumetric canvases using deterministic seeds. Optional refinement
stages let you reuse the same diffusion/super-resolution path immediately after decoding:

```rust
let latent = Tensor::from_vec(1, 16, latent_vec)?;
let mut decoder = ZDecoder::new(4, 32, 32, 0xDEADBEEF)?;
let generated = decoder.decode_with_refinement(
    &latent,
    Some(&diffuser),
    Some(InterpolationMethod::Linear),
    Some(2),
)?;
```

### Video-aware projection pipeline

`VideoStreamProjector` orchestrates diffusion, super-resolution, temporal smoothing, and SpiralRNN-driven resonance generation
for sequential inputs. It keeps a `TemporalResonanceBuffer`, calibrates the internal `VisionProjector` with live `AtlasFrame`
telemetry, and threads the previous resonance back into `ResonanceGenerator::generate` for continuity:

```rust
let generator = ResonanceGenerator::new("video", 12, 5)?;
let projector = VisionProjector::new(0.55, 0.35, 0.1);
let mut stream = VideoStreamProjector::new(projector, generator, 0.25)
    .with_diffuser(ZDiffuser::new(1, 0.2))
    .with_super_resolution(InterpolationMethod::Linear, 2);

let chrono_frames = vec![Some(summary_for(frame0)), Some(summary_for(frame1))];
let atlas_frames = vec![Some(atlas0), Some(atlas1)];
let projections = stream.project_sequence(&[volume0, volume1], &chrono_frames, &atlas_frames)?;
let latest_resonance = stream.last_resonance();
```

You can also process frames incrementally with `step`, which returns both the projection and an `Arc<DifferentialResonance>` for downstream logging, feedback, or closed-loop modulation.

## Resonant roadmap
- **Z-space as a perceptual frequency domain**: Use the new spectral windows and `ZSpaceVolume::spectral_response` helper to treat collapse as a frequency-aware attention sweep that preconditions downstream ConvNets.
- **Temporal resonance layering**: Extend the new `TemporalResonanceBuffer` into multi-scale stacks that model short- and long-term attention simultaneously.
- **Dynamic multi-camera orchestration**: Extend `MultiViewFusion` with calibration from SLAM/IMU pipelines and learnable alignment gammas so attention follows moving rigs in autonomous capture setups.
- **Generative resonance coupling**: Train the new `ResonanceGenerator` alongside SpiralRNN/`ZConductor` stacks so resonance synthesis becomes adaptive rather than purely heuristic.
- **Learnable super-resolution in Z**: Pair the new interpolation/diffusion primitives with trainable weights so Z-aware GAN/VAE stacks can decide when to refine, denoise, or hallucinate volumetric detail instead of relying on heuristics.

This guide will evolve over time to map TorchVision ecosystems to the expanding SpiralTorchVision feature set.

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
- **Long-term integrations**: Leveraging TorchVision datasets/transforms as inputs while adding Z-space-native losses, visualization tools, and SpiralTorch-specific model heads.
- **Modular vision backbones**: `st_vision::models` ships ergonomic ResNet, ViT, and ConvNeXt backbones that reuse the core `st-nn` layers. Each module accepts a concise configuration, reports stage shapes, and exposes state-dict helpers for checkpoint I/O.

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
assert_eq!(backbone.stage_shapes().len(), 4);
```

The `ViTBackbone` applies a stride-equals-patch convolution to form tokens, averages them into a single embedding, and reports the number of patches via `patches()`. `ConvNeXtBackbone` mirrors the ConvNeXt-T stage schedule with lightweight MLP-style blocks, returning flattened feature maps alongside per-stage shape metadata.

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

## Resonant roadmap
- **Z-space as a perceptual frequency domain**: Use the new spectral windows and `ZSpaceVolume::spectral_response` helper to treat collapse as a frequency-aware attention sweep that preconditions downstream ConvNets.
- **Temporal resonance layering**: Extend the new `TemporalResonanceBuffer` into multi-scale stacks that model short- and long-term attention simultaneously.
- **Dynamic multi-camera orchestration**: Extend `MultiViewFusion` with calibration from SLAM/IMU pipelines and learnable alignment gammas so attention follows moving rigs in autonomous capture setups.
- **Generative resonance coupling**: Feed resonance fields generated by SpiralRNN or future ZConductors back into the projector, creating a closed-loop "generative visual consciousness" cycle.
- **Super-resolution and synthesis in Z**: Add interpolation, upscaling, diffusion, and decoding helpers so Z-aware GAN/VAE stacks can both enhance and generate volumetric inputs.

This guide will evolve over time to map TorchVision ecosystems to the expanding SpiralTorchVision feature set.

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
- **Temporal resonance accumulation**: `ZSpaceVolume::accumulate`, `ZSpaceVolume::blend_sequence`, and `VisionProjector::project_sequence` blend multi-frame sequences with Gaussian temporal attention, enabling video/timeseries perception before collapse.
- **ZSpaceVolume / VisionProjector**: A volumetric representation that accumulates resonant features along the Z-axis and collapses them into tensors. It ingests intermediate activations from TorchVision models and bridges them into SpiralTorch's Z-space analyzers.
- **Differential Resonance integration**: Combining with `st_tensor::DifferentialResonance` to reproject spatiotemporal features gathered from TorchVision networks back into SpiralTorch's resonant frames.
- **Long-term integrations**: Leveraging TorchVision datasets/transforms as inputs while adding Z-space-native losses, visualization tools, and SpiralTorch-specific model heads.

### Temporal resonance accumulation

Temporal resonance treats each volume in a sequence as a waypoint along a perceptual trajectory. The combination of EMA updates and Gaussian-temporal fusion keeps the sequence coherent while still respecting depth attention.

```rust
use st_vision::{VisionProjector, ZSpaceVolume};
use st_tensor::DifferentialResonance;

fn collapse_sequence(
    projector: &VisionProjector,
    resonance: &DifferentialResonance,
    frames: &[ZSpaceVolume],
) -> st_tensor::PureResult<st_tensor::Tensor> {
    // Streaming EMA to keep a running latent ready for low-latency consumers.
    let mut state = frames
        .first()
        .cloned()
        .ok_or_else(|| st_tensor::TensorError::EmptyInput("z_space_sequence"))?;
    for frame in &frames[1..] {
        projector.accumulate_temporal(&mut state, frame)?;
    }

    // Full-sequence fusion before Z collapse for high-quality output.
    let fused = projector.fuse_sequence(frames)?;
    projector.project(&fused, resonance)
}
```

`VisionProjector::with_temporal_attention` lets you steer the Gaussian center (0 = oldest frame, 1 = most recent) and the decay width. The same parameters drive both the streaming EMA and the offline fusion so your temporal footprint stays consistent across inference modes.

## Resonant roadmap
- **Z-space as a perceptual frequency domain**: Use the new spectral windows and `ZSpaceVolume::spectral_response` helper to treat collapse as a frequency-aware attention sweep that preconditions downstream ConvNets.
- **Temporal resonance evolution**: Build on the EMA/Gaussian stack with learned temporal kernels from SpiralRNN so attention shifts can be predicted rather than hand-tuned.
- **Multi-camera Z-fusion**: Treat each Z slice as a distinct viewpoint index, fuse multi-perspective tensors inside `ZSpaceVolume::from_slices`, and drive viewpoint-specific attention maps for stereo and birds-eye tasks.
- **Generative resonance coupling**: Feed resonance fields generated by SpiralRNN or future ZConductors back into the projector, creating a closed-loop "generative visual consciousness" cycle.
- **Super-resolution and synthesis in Z**: Add interpolation, upscaling, diffusion, and decoding helpers so Z-aware GAN/VAE stacks can both enhance and generate volumetric inputs.

This guide will evolve over time to map TorchVision ecosystems to the expanding SpiralTorchVision feature set.

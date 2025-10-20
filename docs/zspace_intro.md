# What is Z-space?

SpiralTorch revolves around a shared state-space called **Z-space**. When kernels
emit tensors, gradients, or metadata they are always interpreted relative to this
space. Understanding how Z-space is constructed helps you reason about
SpiralTorch's scheduling, spectral operators, and topology-preserving buffers.

## Motivating intuition

Z-space is a **hybrid spectral–topological manifold**. It begins as a
frequency-aligned lattice (think Mellin wavelets stretched along golden-ratio
intervals) and is then warped by the current task's resonance profile. The warp
preserves orientation so that operators can compose safely, yet it keeps the
spectrum locally stationary for fast FFT- and Mellin-family primitives.

At runtime Z-space is represented as a stack of cells:

- **Spectral cells** hold band-limited activations. They are indexed by
  `(band, orientation, resonance)`.
- **Topological cells** track adjacency in the warped manifold. They store
  winding numbers, branch cuts, and cross-link hints used by homotopy-aware
  optimisers.
- **Temporal cells** capture time-shifted echoes. These are optional but power
  features such as differential resonance replay and multiframe fusion.

## Coordinate systems

All tensors flowing through SpiralTorch carry a **Z-frame** that defines how to
interpret their axes.

| Axis | Meaning | Typical range |
| ---- | ------- | ------------- |
| `μ` (mu) | Spectral band index | `0 … N_bands` |
| `ν` (nu) | Orientation / spin | `-K … +K` |
| `τ` (tau) | Temporal echo index | `0 … N_echoes` |
| `χ` (chi) | Topological sheet selector | `0 … N_sheets` |

The core libraries expose helpers (`ZFrame`, `ZIndex`, `ZAtlas`) to negotiate
between frames. Downstream APIs—Python, Rust, and TypeScript—propagate Z-frames
implicitly so user code can stay in tensor-first thinking.

## Relationship to tensors

Most modules expose familiar tensor signatures. Internally, operators lift
inputs into Z-space, run topology-aware schedules, and then **project back** to
your requested layout. The README callout about GPU-first convolution describes
this flow: the WGPU backend expands the 5D activation volume on-device,
performs batched GEMMs, then collapses the result back through the active
Z-frame before returning a standard tensor.

When you request a `spiraltorch.nn.Conv2d` with `layout="NCHW"`, the following
happens:

1. Activations and filters are lifted into Z-space via the current `ZAtlas`.
2. Spectral–topological kernels execute over the warped lattice.
3. The result is projected into the layout you requested.

## Persistence and caching

Z-space shapes the runtime caches:

- **Kernel caches** are keyed by Z-frame signatures. If two kernels operate over
  identical Z-frames, codegen artifacts and tuning results are shared.
- **Checkpointing** stores Z-frame metadata next to tensor values. This ensures
  resuming a training run restores the same spectral orientation.
- **Streaming** features pre-allocate Z-space volumes for upcoming frames so
  multi-camera or sensor fusion jobs keep continuity.

## Inspecting Z-space at runtime

SpiralTorch includes lightweight probes to inspect the active Z-space atlas:

```python
import spiraltorch as st

with st.zspace.session() as session:
    print("active bands:", session.frame.bands)
    print("sheet topology:", session.frame.sheets)
    print("echo count:", session.frame.temporal.echo_count)
```

`st.zspace.render_atlas()` produces a quick Matplotlib overview of the current
manifold. For deeper inspection, `st.zspace.sample_field()` lets you peek at the
spectral density and homology classes assigned to each cell.

## Further reading

- [SpiralTorchVision overview](docs/spiraltorchvision.md)
- [Cloud Integration Guide](docs/cloud_integration.md)
- `crates/st-core/src/zspace/` for Rust implementations of the atlas, frames,
  and topological planners.

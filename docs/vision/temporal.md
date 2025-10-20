# Temporal Z-Space Workflows

Temporal Z-space extends `ZSpaceVolume` with per-voxel temporal harmonics and
resonance decay parameters. The additions allow SpiralTorch to replay a
4-dimensional resonant field without re-running heavy inference pipelines.

## Volume authoring

1. Allocate a volume with harmonics using `ZSpaceVolume::zeros_with_temporal`.
2. Populate spatial voxels (`volume.voxels_mut()`) and their harmonic spectra
   (`volume.temporal_harmonics_mut()`), storing resonance damping inside
   `volume.resonance_decay_mut()`.
3. Use `st_logic::temporal_dynamics::integrate_volume` to blend live input with
   historical state. This function respects harmonic channels and applies
   exponential damping derived from the decay buffer.

## Rendering and export

* `st_backend_wgpu::render::TemporalRenderer` evaluates the harmonic series for
  each voxel across time, producing `[frames, depth, height, width]` sequences.
* The `visualize_z_volume` utility (built into `st-backend-wgpu`) generates a
  textual dump. Run it with:

```bash
cargo run -p st-backend-wgpu --bin visualize_z_volume -- --depth=4 --height=4 --width=4 --frames=32 --output=temporal.csv
```

The CSV contains per-frame amplitudes that can be plotted to create animations
or debug resonance propagation.

## Regression testing

Temporal interpolation and integration are exercised through unit tests in
`st-vision`, `st-backend-wgpu`, and `st-logic`. Together they ensure:

* Harmonically enriched volumes preserve metadata when accumulated.
* Rendered animations respect harmonic oscillations and exponential decay.
* Temporal integration blends drive signals into historical state without shape
  regressions.

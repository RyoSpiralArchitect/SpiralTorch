use std::env;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

use rand::{rngs::StdRng, Rng, SeedableRng};
use st_backend_wgpu::render::{
    TemporalRenderOutput, TemporalRenderer, TemporalRendererConfig, TemporalVolumeLike,
};
use st_vision::ZSpaceVolume;

struct VisionVolume<'a>(&'a ZSpaceVolume);

impl<'a> TemporalVolumeLike for VisionVolume<'a> {
    fn depth(&self) -> usize {
        self.0.depth()
    }

    fn height(&self) -> usize {
        self.0.height()
    }

    fn width(&self) -> usize {
        self.0.width()
    }

    fn harmonic_channels(&self) -> usize {
        self.0.harmonic_channels()
    }

    fn voxels(&self) -> &[f32] {
        self.0.voxels()
    }

    fn temporal_harmonics(&self) -> &[f32] {
        self.0.temporal_harmonics()
    }

    fn resonance_decay(&self) -> &[f32] {
        self.0.resonance_decay()
    }
}

fn parse_arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let key = format!("--{}=", name);
    for arg in env::args().skip(1) {
        if let Some(value) = arg.strip_prefix(&key) {
            if let Ok(parsed) = value.parse::<T>() {
                return parsed;
            }
        }
    }
    default
}

fn build_volume(
    depth: usize,
    height: usize,
    width: usize,
    harmonic_channels: usize,
    seed: u64,
) -> ZSpaceVolume {
    let mut volume = ZSpaceVolume::zeros_with_temporal(depth, height, width, harmonic_channels)
        .expect("invalid dimensions for volume");
    let mut rng = StdRng::seed_from_u64(seed);
    for (idx, voxel) in volume.voxels_mut().iter_mut().enumerate() {
        let z = idx / (height * width);
        *voxel = (z as f32 + rng.gen_range(0.0..1.0)) / depth as f32;
    }
    for coeff in volume.temporal_harmonics_mut().iter_mut() {
        *coeff = rng.gen_range(-0.5..0.5);
    }
    for decay in volume.resonance_decay_mut().iter_mut() {
        *decay = rng.gen_range(0.2..1.2);
    }
    volume
}

fn write_output<P: AsRef<Path>>(output: &TemporalRenderOutput, path: Option<P>) -> io::Result<()> {
    let mut writer: Box<dyn Write> = match path {
        Some(p) => Box::new(File::create(p)?),
        None => Box::new(io::stdout()),
    };
    writeln!(
        writer,
        "# frames={} depth={} height={} width={}",
        output.frames, output.depth, output.height, output.width
    )?;
    for slice in &output.slices {
        writeln!(
            writer,
            "frame={} time={:.6} values={}",
            slice.frame_index,
            slice.time_seconds,
            slice
                .data
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(",")
        )?;
    }
    Ok(())
}

fn main() -> io::Result<()> {
    let depth = parse_arg("depth", 4usize);
    let height = parse_arg("height", 4usize);
    let width = parse_arg("width", 4usize);
    let frames = parse_arg("frames", 16usize);
    let harmonics = parse_arg("harmonics", 3usize);
    let seed = parse_arg("seed", 13u64);
    let output_path: Option<String> = env::args()
        .skip(1)
        .find_map(|arg| arg.strip_prefix("--output=").map(|s| s.to_string()));

    let volume = build_volume(depth, height, width, harmonics, seed);
    let renderer = TemporalRenderer::new(TemporalRendererConfig {
        frames,
        ..Default::default()
    });
    let render_output = renderer
        .render(&VisionVolume(&volume))
        .expect("unable to render temporal volume");
    if let Some(path) = output_path {
        write_output(&render_output, Some(path))?;
    } else {
        write_output(&render_output, Option::<&str>::None)?;
    }
    Ok(())
}

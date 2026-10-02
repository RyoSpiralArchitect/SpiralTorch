//! Shared test/benchmark presets, not a device-independent performance policy.
use st_backend_wgpu::resident_matmul::{MatmulKernel, MatmulTile};

pub fn options(name: &str) -> Result<(MatmulTile, MatmulKernel), Box<dyn std::error::Error>> {
    Ok(match name {
        "scalar" => (MatmulTile::default(), MatmulKernel::Scalar),
        "register8" => (MatmulTile::default(), MatmulKernel::Register2x2),
        "register16" => (MatmulTile::new(16, 16, 16)?, MatmulKernel::Register2x2),
        _ => return Err("unknown projection mode".into()),
    })
}

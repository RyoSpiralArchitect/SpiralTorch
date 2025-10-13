use wgpu::*;

/// Loads the ND indexer shader that materialises strided indices and segment ids.
pub fn create_pipeline(device: &Device, shader_dir: &str) -> ComputePipeline {
    let src = std::fs::read_to_string(format!("{shader_dir}/nd_indexer.wgsl"))
        .expect("missing nd_indexer.wgsl");
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("nd_indexer"),
        source: ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("nd_indexer"),
        layout: None,
        module: &module,
        entry_point: "main",
    })
}

use wgpu::*;

pub struct CompactionPipelines {
    pub p_1ce: ComputePipeline,
}

pub fn create(device:&Device, shader_dir:&str)->CompactionPipelines{
    let src = std::fs::read_to_string(format!("{shader_dir}/wgpu_compaction_1ce.wgsl")).unwrap();
    let sm = device.create_shader_module(ShaderModuleDescriptor{
        label: Some("compaction_1ce"),
        source: ShaderSource::Wgsl(src.into())
    });
    let p = device.create_compute_pipeline(&ComputePipelineDescriptor{
        label: Some("compaction_1ce"),
        layout: None, module:&sm, entry_point:"main_cs"
    });
    CompactionPipelines{ p_1ce: p }
}

use wgpu::*;

pub struct Compaction2Pipelines {
    pub p_scan: ComputePipeline,
    pub p_apply: ComputePipeline,
}

pub fn create(device:&Device, shader_dir:&str)->Compaction2Pipelines{
    let scan_src = std::fs::read_to_string(format!("{shader_dir}/wgpu_compaction_scan.wgsl")).unwrap();
    let apply_src= std::fs::read_to_string(format!("{shader_dir}/wgpu_compaction_apply.wgsl")).unwrap();
    let sm_scan = device.create_shader_module(ShaderModuleDescriptor{
        label: Some("compaction_scan"),
        source: ShaderSource::Wgsl(scan_src.into())
    });
    let sm_apply = device.create_shader_module(ShaderModuleDescriptor{
        label: Some("compaction_apply"),
        source: ShaderSource::Wgsl(apply_src.into())
    });
    let p_scan = device.create_compute_pipeline(&ComputePipelineDescriptor{
        label: Some("compaction_scan"),
        layout: None, module:&sm_scan, entry_point:"main_cs"
    });
    let p_apply = device.create_compute_pipeline(&ComputePipelineDescriptor{
        label: Some("compaction_apply"),
        layout: None, module:&sm_apply, entry_point:"main_cs"
    });
    Compaction2Pipelines{ p_scan, p_apply }
}

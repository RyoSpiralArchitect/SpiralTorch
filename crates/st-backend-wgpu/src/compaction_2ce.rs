// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use wgpu::*;
pub struct Compaction2CE {
    pub p_scan: ComputePipeline,
    pub p_apply: ComputePipeline,
}
pub fn create(device:&Device, shader_dir:&str)->Compaction2CE{
    let scan_src = std::fs::read_to_string(format!("{shader_dir}/wgpu_compaction_scan_pass.wgsl")).unwrap();
    let apply_src = std::fs::read_to_string(format!("{shader_dir}/wgpu_compaction_apply_pass.wgsl")).unwrap();
    let sm_scan = device.create_shader_module(ShaderModuleDescriptor{ label:Some("scan"), source: ShaderSource::Wgsl(scan_src.into()) });
    let sm_apply= device.create_shader_module(ShaderModuleDescriptor{ label:Some("apply"), source: ShaderSource::Wgsl(apply_src.into()) });
    let p_scan = device.create_compute_pipeline(&ComputePipelineDescriptor{ label:Some("scan"), layout: None, module:&sm_scan, entry_point:"main_cs"});
    let p_apply= device.create_compute_pipeline(&ComputePipelineDescriptor{ label:Some("apply"),layout: None, module:&sm_apply, entry_point:"main_cs"});
    Compaction2CE{ p_scan, p_apply }
}

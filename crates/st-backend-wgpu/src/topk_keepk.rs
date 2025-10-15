// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Minimal driver for WGSL keep‑k kernels (subgroup/workgroup).
//! This is a scaffold; integrate with your existing WGPU backend device/queue management.

use wgpu::*;

pub enum MergeKind { Bitonic=0, Shared=1, Warp=2 }

pub struct Pipelines {
    pub keepk_subgroup: Option<ComputePipeline>,
    pub keepk_workgroup: ComputePipeline,
}

pub fn create_pipelines(device:&Device, shader_dir:&str, supports_subgroup:bool)->Pipelines{
    let wgsl_work = std::fs::read_to_string(format!("{shader_dir}/topk_keepk_workgroup.wgsl")).unwrap();
    let mod_work = device.create_shader_module(ShaderModuleDescriptor{
        label: Some("keepk_workgroup"),
        source: ShaderSource::Wgsl(wgsl_work.into())
    });
    let pl_work = device.create_compute_pipeline(&ComputePipelineDescriptor{
        label: Some("keepk_workgroup"),
        layout: None,
        module: &mod_work, entry_point: "main_cs"
    });

    let keepk_subgroup = if supports_subgroup {
        let wgsl_sub = std::fs::read_to_string(format!("{shader_dir}/topk_keepk_subgroup.wgsl")).unwrap();
        let mod_sub = device.create_shader_module(ShaderModuleDescriptor{
            label: Some("keepk_subgroup"),
            source: ShaderSource::Wgsl(wgsl_sub.into())
        });
        Some(device.create_compute_pipeline(&ComputePipelineDescriptor{
            label: Some("keepk_subgroup"),
            layout: None,
            module: &mod_sub, entry_point: "main_cs"
        }))
    } else { None };

    Pipelines{ keepk_subgroup, keepk_workgroup: pl_work }
}

//! Composable GPU consensus over softmax probabilities and an all-peak mask.
//! This is the raw GPU operation, not the Tensor API's CPU/statistics blend.

use crate::runtime::Shared;
use bytemuck::{Pod, Zeroable};
use wgpu::{BindGroup, BindGroupLayout, Buffer, Device};

/// The shader's unchanged 64-byte ABI. Strides count f32 elements. Flag bit 0
/// selects Chimera addressing; metrics contain four f32 values per row.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Params {
    pub rows: u32,
    pub cols: u32,
    pub soft_stride: u32,
    pub mask_stride: u32,
    pub spiral_stride: u32,
    pub chimera_tile: u32,
    pub chimera_stripes: u32,
    pub flags: u32,
    pub phi: f32,
    pub phi_conjugate: f32,
    pub phi_bias: f32,
    pub leech_scale: f32,
    pub ramanujan_ratio: f32,
    pub inv_cols: f32,
    pub entropy_epsilon: f32,
    pub _pad: f32,
}

pub fn bind_layout(device: &Device) -> BindGroupLayout {
    let entries: Vec<_> = (0..5)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 4 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: binding < 2,
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("st.softmax.consensus.layout"),
        entries: &entries,
    })
}

pub fn bind(
    device: &Device,
    layout: &BindGroupLayout,
    softmax: &Buffer,
    mask: &Buffer,
    spiral: &Buffer,
    metrics: &Buffer,
    params: &Buffer,
) -> BindGroup {
    let entries: Vec<_> = [softmax, mask, spiral, metrics, params]
        .into_iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.softmax.consensus.bind"),
        layout,
        entries: &entries,
    })
}

/// Low-level native/WASM pipeline. Callers own valid uniforms, non-aliasing
/// buffers, row/stride bounds, device limits and the explicit observation point.
pub struct Pipeline {
    pub bind_layout: BindGroupLayout,
    pub compute: Shared<wgpu::ComputePipeline>,
}

impl Pipeline {
    pub fn from_embedded(device: &Device) -> Self {
        let bind_layout = bind_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("st.softmax.consensus.pipeline_layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("st.softmax.consensus.shader"),
            source: wgpu::ShaderSource::Wgsl(
                crate::shader_sources::SOFTMAX_SPIRAL_CONSENSUS_WGSL.into(),
            ),
        });
        let compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("st.softmax.consensus.pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Self {
            bind_layout,
            compute: Shared::new(compute),
        }
    }

    /// Encode without submitting or reading back, so preceding softmax and
    /// following consumers can share the caller's command buffer.
    pub fn encode_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        binding: &BindGroup,
        rows: u32,
    ) -> bool {
        if rows == 0 {
            return false;
        }
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.softmax.consensus.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compute);
        pass.set_bind_group(0, binding, &[]);
        pass.dispatch_workgroups(rows, 1, 1);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uniform_abi() {
        assert_eq!(std::mem::size_of::<Params>(), 64);
        assert_eq!(std::mem::align_of::<Params>(), 16);
        assert_eq!(std::mem::offset_of!(Params, phi), 32);
        assert_eq!(std::mem::offset_of!(Params, entropy_epsilon), 56);
    }
}

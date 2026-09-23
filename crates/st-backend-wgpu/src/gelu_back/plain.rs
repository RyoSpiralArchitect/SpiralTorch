//! Derivative-only GELU: one write, no residual/bias work and no implicit readback.
use super::portable::{self, PlanError};
use crate::{runtime::Shared, util::create_inline_pipeline, ShaderLoadError};
use wgpu::{Buffer, Device};

#[derive(Debug, Clone, Copy)]
pub struct Plan {
    params: [u32; 4],
    grid: [u32; 3],
}

impl Plan {
    pub fn new(rows: usize, cols: usize, limits: &wgpu::Limits) -> Result<Self, PlanError> {
        let len = rows.checked_mul(cols).ok_or(PlanError("shape overflow"))?;
        let len = portable::storage_len(
            u64::try_from(len).map_err(|_| PlanError("length overflow"))?,
            limits,
        )?;
        if limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroups_per_dimension == 0
        {
            return Err(PlanError("device cannot dispatch plain GELU"));
        }
        let groups = (len as u32).div_ceil(256);
        let x = groups.min(limits.max_compute_workgroups_per_dimension);
        let y = groups.div_ceil(x);
        if y > limits.max_compute_workgroups_per_dimension {
            return Err(PlanError("dispatch exceeds device limits"));
        }
        Ok(Self {
            params: [len as u32, x, groups, 0],
            grid: [x, y, 1],
        })
    }
    pub fn len(&self) -> usize {
        self.params[0] as usize
    }
    pub fn is_empty(&self) -> bool {
        false
    }
    pub fn uniforms(&self) -> [u32; 4] {
        self.params
    }
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }
}

pub fn source() -> String {
    include_str!("../shaders/gelu_backward_plain.wgsl").replace(
        "// GELU_DERIVATIVE",
        crate::shader_sources::GELU_DERIVATIVE_WGSL,
    )
}

pub fn bind_layout(device: &Device) -> wgpu::BindGroupLayout {
    portable::layout(device, &[true, true, false], "st.gelu.plain.layout")
}

/// Ordered bindings: z, upstream, gradient output, plan uniforms. Callers own
/// valid, nonaliasing buffers and finite-data policy; WGPU checks device ownership.
pub fn bind(
    device: &Device,
    layout: &wgpu::BindGroupLayout,
    buffers: [&Buffer; 4],
) -> wgpu::BindGroup {
    portable::bind(device, layout, &buffers)
}

pub struct Pipeline {
    pub bind_layout: wgpu::BindGroupLayout,
    pub compute: Shared<wgpu::ComputePipeline>,
}

impl Pipeline {
    pub fn from_embedded(device: &Device) -> Result<Self, ShaderLoadError> {
        let bind_layout = bind_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("st.gelu.plain"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let compute = create_inline_pipeline(device, "st.gelu.plain", source(), "main", &layout)?;
        Ok(Self {
            bind_layout,
            compute,
        })
    }

    pub fn encode_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        binding: &wgpu::BindGroup,
        plan: &Plan,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.gelu.plain"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compute);
        pass.set_bind_group(0, binding, &[]);
        let [x, y, z] = plan.grid();
        pass.dispatch_workgroups(x, y, z);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plain_plan_handles_tails_and_two_dimensional_dispatch() {
        let mut limits = wgpu::Limits::downlevel_defaults();
        limits.max_compute_workgroups_per_dimension = 2;
        let p = Plan::new(1, 513, &limits).unwrap();
        assert_eq!(p.uniforms(), [513, 2, 3, 0]);
        assert_eq!(p.grid(), [2, 2, 1]);
        for (r, c) in [(0, 4), (4, 0), (usize::MAX, 2), (1, 1025)] {
            assert!(Plan::new(r, c, &limits).is_err());
        }
        let module = naga::front::wgsl::parse_str(&source()).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

use super::*;
use crate::util::{apply_overrides, create_inline_pipeline};

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("invalid GELU backward plan: {0}")]
pub struct PlanError(pub(crate) &'static str);

impl Geometry {
    pub fn validate(&self, limits: &wgpu::Limits) -> Result<(), PlanError> {
        let tile = self
            .wg_rows
            .checked_mul(self.wg_cols)
            .ok_or(PlanError("tile overflow"))?;
        if tile == 0
            || self.reduce_wg == 0
            || self.wg_cols > limits.max_compute_workgroup_size_x
            || self.wg_rows > limits.max_compute_workgroup_size_y
            || tile > limits.max_compute_invocations_per_workgroup
            || u64::from(tile) * 4 > u64::from(limits.max_compute_workgroup_storage_size)
            || self.reduce_wg > limits.max_compute_workgroup_size_x
            || self.reduce_wg > limits.max_compute_invocations_per_workgroup
        {
            return Err(PlanError("workgroup geometry exceeds device limits"));
        }
        Ok(())
    }
}

pub(crate) fn storage_len(elements: u64, limits: &wgpu::Limits) -> Result<usize, PlanError> {
    let bytes = elements
        .checked_mul(4)
        .ok_or(PlanError("byte count overflow"))?;
    if elements == 0
        || elements > u64::from(u32::MAX)
        || bytes > isize::MAX as u64
        || bytes > limits.max_buffer_size
        || bytes > u64::from(limits.max_storage_buffer_binding_size)
    {
        return Err(PlanError("storage/index range exceeds device limits"));
    }
    usize::try_from(elements).map_err(|_| PlanError("host length overflow"))
}

/// Validated two-pass geometry, including padded rows. Uniforms remain the
/// existing 32/16-byte ABI. This validates shape, not buffer contents or ownership.
#[derive(Debug, Clone, Copy)]
pub struct Plan {
    fused: FusedUniforms,
    reduce: ReduceUniforms,
    geometry: Geometry,
    storage_len: usize,
    partial_len: usize,
}

impl Plan {
    pub fn new(
        batch: u32,
        cols: u32,
        stride: u32,
        geometry: Geometry,
        add_residual: bool,
        limits: &wgpu::Limits,
    ) -> Result<Self, PlanError> {
        geometry.validate(limits)?;
        if batch == 0 || cols == 0 || stride < cols {
            return Err(PlanError("empty shape or short row stride"));
        }
        let (x, y) = geometry.tiles(batch, cols);
        let cap = limits.max_compute_workgroups_per_dimension;
        if x > cap || y > cap || geometry.reduce_dispatch(cols).0 > cap {
            return Err(PlanError("dispatch exceeds device limits"));
        }
        let data_len = storage_len(u64::from(batch) * u64::from(stride), limits)?;
        let partial_len = storage_len(
            u64::from(x) * u64::from(y) * u64::from(geometry.wg_cols),
            limits,
        )?;
        Ok(Self {
            fused: FusedUniforms::new(batch, cols, stride, x, y, add_residual),
            reduce: ReduceUniforms::new(cols, x, y),
            geometry,
            storage_len: data_len,
            partial_len,
        })
    }

    pub fn fused_uniforms(&self) -> FusedUniforms {
        self.fused
    }
    pub fn reduce_uniforms(&self) -> ReduceUniforms {
        self.reduce
    }
    pub fn storage_len(&self) -> usize {
        self.storage_len
    }
    pub fn partial_len(&self) -> usize {
        self.partial_len
    }
    pub fn fused_grid(&self) -> [u32; 3] {
        [self.fused.num_wg_x, self.fused.num_wg_y, 1]
    }
    pub fn reduce_grid(&self) -> [u32; 3] {
        [self.geometry.reduce_dispatch(self.fused.cols).0, 1, 1]
    }
}

/// Shared layout factory; the final binding is always a uniform.
pub(crate) fn layout(device: &Device, readonly: &[bool], label: &str) -> BindGroupLayout {
    let entries: Vec<_> = (0..=readonly.len())
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if i == readonly.len() {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage {
                        read_only: readonly[i],
                    }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

pub fn fused_bind_layout(device: &Device) -> BindGroupLayout {
    layout(
        device,
        &[true, true, false, false, false],
        "st.gelu.fused.layout",
    )
}

pub fn reduce_bind_layout(device: &Device) -> BindGroupLayout {
    layout(device, &[true, false], "st.gelu.reduce.layout")
}

pub(crate) fn bind(
    device: &Device,
    layout: &BindGroupLayout,
    buffers: &[&Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, buffer)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.gelu.bind"),
        layout,
        entries: &entries,
    })
}

/// Buffers and uniform contents must match the plan and belong to this device.
pub fn fused_bind(
    device: &Device,
    layout: &BindGroupLayout,
    buffers: [&Buffer; 6],
) -> wgpu::BindGroup {
    bind(device, layout, &buffers)
}

pub fn reduce_bind(
    device: &Device,
    layout: &BindGroupLayout,
    buffers: [&Buffer; 3],
) -> wgpu::BindGroup {
    bind(device, layout, &buffers)
}

fn specialize(
    source: &str,
    file: &str,
    overrides: &[(&str, u32)],
) -> Result<String, ShaderLoadError> {
    let source = apply_overrides(&std::sync::Arc::from(source), file, overrides)?;
    // WGPU 0.20 cannot size workgroup arrays with override expressions. Values
    // have already been chosen on the host; fold them into ordinary constants.
    Ok(source.replace("override ", "const ").replace(
        "// GELU_DERIVATIVE",
        crate::shader_sources::GELU_DERIVATIVE_WGSL,
    ))
}

fn fused_specialize(source: &str, geometry: Geometry) -> Result<String, ShaderLoadError> {
    let tile = geometry
        .wg_rows
        .checked_mul(geometry.wg_cols)
        .ok_or(ShaderLoadError::InvalidSpecialization("GELU tile overflow"))?;
    specialize(
        source,
        "fused_gelu_back.wgsl",
        &[
            ("WG_ROWS", geometry.wg_rows),
            ("WG_COLS", geometry.wg_cols),
            ("WG_TILE", tile),
        ],
    )
}

pub fn fused_source(geometry: Geometry) -> Result<String, ShaderLoadError> {
    fused_specialize(include_str!("../shaders/fused_gelu_back.wgsl"), geometry)
}

pub fn reduce_source(geometry: Geometry) -> Result<String, ShaderLoadError> {
    specialize(
        include_str!("../shaders/reduce_db.wgsl"),
        "reduce_db.wgsl",
        &[
            ("WG_COLS", geometry.wg_cols),
            ("REDUCE_WG", geometry.reduce_wg),
        ],
    )
}

pub(super) fn build(
    device: &Device,
    geometry: Geometry,
    fused: &str,
    reduce: &str,
) -> Result<Pipelines, ShaderLoadError> {
    geometry
        .validate(&device.limits())
        .map_err(|e| ShaderLoadError::InvalidSpecialization(e.0))?;
    let fused_bind_layout = fused_bind_layout(device);
    let reduce_bind_layout = reduce_bind_layout(device);
    let pipeline = |binding, source, entry, label| {
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[binding],
            push_constant_ranges: &[],
        });
        create_inline_pipeline(device, label, source, entry, &layout)
    };
    let fused = pipeline(
        &fused_bind_layout,
        fused_specialize(fused, geometry)?,
        "main",
        "st.gelu.fused",
    )?;
    let reduce = pipeline(
        &reduce_bind_layout,
        specialize(
            reduce,
            "reduce_db.wgsl",
            &[
                ("WG_COLS", geometry.wg_cols),
                ("REDUCE_WG", geometry.reduce_wg),
            ],
        )?,
        "reduce",
        "st.gelu.reduce",
    )?;
    Ok(Pipelines {
        fused_bind_layout,
        reduce_bind_layout,
        fused,
        reduce,
        geometry,
    })
}

impl Pipelines {
    /// Compile canonical shaders without any filesystem access, including WASM.
    pub fn from_embedded(device: &Device, geometry: Geometry) -> Result<Self, ShaderLoadError> {
        build(
            device,
            geometry,
            include_str!("../shaders/fused_gelu_back.wgsl"),
            include_str!("../shaders/reduce_db.wgsl"),
        )
    }

    /// Encode both stages without submission or observation. Bindings must use
    /// this pipeline's geometry and the supplied plan's uniforms and buffers.
    pub fn encode_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        fused: &wgpu::BindGroup,
        reduce: &wgpu::BindGroup,
        plan: &Plan,
    ) -> Result<(), PlanError> {
        if self.geometry.wg_rows != plan.geometry.wg_rows
            || self.geometry.wg_cols != plan.geometry.wg_cols
            || self.geometry.reduce_wg != plan.geometry.reduce_wg
        {
            return Err(PlanError("pipeline and plan geometries differ"));
        }
        for (pipeline, binding, grid) in [
            (&self.fused, fused, plan.fused_grid()),
            (&self.reduce, reduce, plan.reduce_grid()),
        ] {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("st.gelu.pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, binding, &[]);
            pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fused_plan_guards_shapes_and_specializations() {
        let limits = wgpu::Limits::downlevel_defaults();
        let g = Geometry::default();
        let p = Plan::new(33, 65, 72, g, true, &limits).unwrap();
        assert_eq!(p.storage_len(), 33 * 72);
        assert_eq!(p.partial_len(), 5 * 3 * 16);
        assert_eq!(p.fused_grid(), [5, 3, 1]);
        assert_eq!(p.reduce_grid(), [1, 1, 1]);
        for (r, c, s) in [
            (0, 4, 4),
            (4, 0, 0),
            (4, 8, 7),
            (u32::MAX, 4, 4),
            (4, u32::MAX, u32::MAX),
        ] {
            assert!(Plan::new(r, c, s, g, false, &limits).is_err());
        }
        assert!(Plan::new(1, 1, 1, Geometry { wg_rows: 0, ..g }, false, &limits).is_err());
        assert!(Plan::new(
            1,
            1,
            1,
            Geometry {
                wg_rows: u32::MAX,
                ..g
            },
            false,
            &limits
        )
        .is_err());
        for source in [fused_source(g).unwrap(), reduce_source(g).unwrap()] {
            assert!(!source.contains("override "));
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
        assert!(fused_source(g)
            .unwrap()
            .contains(crate::shader_sources::GELU_DERIVATIVE_WGSL));
    }
}

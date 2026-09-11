//! Reusable shape/layout-specialized chains on the existing tensor device.
//! No pending output is exposed: run submits before returning an immutable tensor.

use super::*;
use st_kernel_contracts::pointwise::{PointwiseChain, PointwiseError, PointwiseExecution};
use std::fmt::Write;

mod inputs;
pub mod vjp;
pub use inputs::PointwiseInputs;

#[derive(Debug)]
pub struct PointwisePlan {
    chain: PointwiseChain,
    layouts: Vec<NdLayout>,
    device: TensorDevice,
    binding_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    metadata: wgpu::Buffer,
    grid: [u32; 3],
}

fn fused_source(chain: &PointwiseChain) -> String {
    generated_source(chain, false)
}

fn generated_source(chain: &PointwiseChain, vjp: bool) -> String {
    let count = chain.input_count();
    let mut code = String::new();
    for i in 0..count {
        writeln!(
            code,
            "@group(0) @binding({i}) var<storage, read> input{i}: array<f32>;"
        )
        .unwrap();
    }
    for (binding, declaration) in [
        "var<storage, read_write> out: array<f32>;",
        "var<storage, read> params: array<u32>;",
        "var<storage, read> inherited: array<u32>;",
        "var<storage, read_write> flags: array<atomic<u32>>;",
    ]
    .iter()
    .enumerate()
    {
        writeln!(
            code,
            "@group(0) @binding({}) {declaration}",
            count + binding
        )
        .unwrap();
    }
    code.push_str(include_str!("../shaders/checked_elementwise.wgsl"));
    if vjp {
        writeln!(
            code,
            "@group(0) @binding({}) var<storage, read> cotangent: array<f32>;",
            count + 4
        )
        .unwrap();
        code.push_str(include_str!("../shaders/gelu_derivative.wgsl"));
    }
    // params: length, rank, grid-x, group-count, shape, (offset, strides)*inputs.
    code.push_str(
        r#"
fn address(index: u32, slot: u32) -> u32 {
    let rank = params[1];
    let base = 4u + rank + slot * (rank + 1u);
    var result = params[base];
    var logical = index;
    for (var axis = rank; axis > 0u; axis--) {
        let d = axis - 1u;
        result += (logical % params[4u + d]) * params[base + 1u + d];
        logical /= params[4u + d];
    }
    return result;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let group = wid.y * params[2] + wid.x;
    if (group >= params[3]) { return; }
    let i = group * 256u + lane;
    if (i == 0u) {
        var bits = 0u;
        for (var j = 0u; j < arrayLength(&inherited); j++) { bits |= inherited[j]; }
        if (bits != 0u) { atomicOr(&flags[0], INVALID_TENSOR_FLAG); }
    }
    if (i >= params[0]) { return; }
"#,
    );
    for i in 0..count {
        writeln!(code, "    let value{i} = input{i}[address(i, {i}u)];").unwrap();
    }
    code.push_str("    var current = value0;\n");
    for (i, step) in chain.steps().iter().enumerate() {
        if vjp {
            writeln!(code, "    let before{i} = current;").unwrap();
        }
        let rhs = step.rhs.map_or("0.0".to_string(), |i| format!("value{i}"));
        writeln!(
            code,
            "    current = checked_apply({}u, current, {rhs});",
            step.op as u32
        )
        .unwrap();
    }
    if vjp {
        code.push_str("    var delta = cotangent[i]; check(delta);\n");
        for slot in 0..count {
            writeln!(code, "    var grad{slot} = 0.0;").unwrap();
        }
        for (j, step) in chain.steps().iter().enumerate().rev() {
            match step.op {
                ElementwiseOp::Identity => {}
                ElementwiseOp::Add => {
                    let slot = step.rhs.unwrap();
                    writeln!(
                        code,
                        "    grad{slot} = checked_apply(OP_ADD, grad{slot}, delta);"
                    )
                    .unwrap();
                }
                ElementwiseOp::Multiply => {
                    let slot = step.rhs.unwrap();
                    writeln!(
                        code,
                        "    let contribution{j} = checked_apply(OP_MULTIPLY, delta, before{j});"
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "    grad{slot} = checked_apply(OP_ADD, grad{slot}, contribution{j});"
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "    delta = checked_apply(OP_MULTIPLY, delta, value{slot});"
                    )
                    .unwrap();
                }
                ElementwiseOp::Relu => {
                    writeln!(code,"    delta = checked_apply(OP_MULTIPLY, delta, select(0.0,1.0,before{j}>0.0));").unwrap();
                }
                ElementwiseOp::Gelu => {
                    writeln!(
                        code,
                        "    delta = checked_apply(OP_MULTIPLY, delta, gelu_prime(before{j}));"
                    )
                    .unwrap();
                }
            }
        }
        code.push_str("    grad0 = checked_apply(OP_ADD, grad0, delta);\n");
        for slot in 0..count {
            writeln!(code, "    out[{slot}u * params[0] + i] = grad{slot};").unwrap();
        }
        code.push_str("}\n");
    } else {
        code.push_str("    out[i] = current;\n}\n");
    }
    substitute_ops(code)
}

impl PointwisePlan {
    /// Prepare once; later runs may supply new immutable inputs with these exact
    /// layouts on the same device/queue. The plan does not retain input values.
    pub fn new(
        device: TensorDevice,
        chain: PointwiseChain,
        layouts: Vec<NdLayout>,
    ) -> Result<Self, TensorError> {
        chain.validate_layouts(&layouts)?;
        let gpu = device.runtime().context().device();
        let limits = gpu.limits();
        let bindings = chain.input_count() as u32 + 4;
        if bindings > limits.max_storage_buffers_per_shader_stage
            || bindings > limits.max_bindings_per_bind_group
        {
            return Err(TensorError::Limit("pointwise input bindings"));
        }
        let output = NdLayout::contiguous(layouts[0].shape())?;
        validate_view(&output, output.len(), &limits)?;
        let grid = grid(output.len(), &limits)?;
        let mut metadata = vec![output.len() as u32, output.rank() as u32, grid[0], grid[2]];
        metadata.extend(output.shape().iter().map(|&n| n as u32));
        for layout in &layouts {
            validate_view(layout, layout.required_storage_len()?, &limits)?;
            let layout = layout.broadcast_to(output.shape())?;
            metadata.push(layout.offset() as u32);
            metadata.extend(layout.strides().iter().map(|&n| n as u32));
        }
        storage_limit(metadata.len(), &limits)?;
        let metadata = runtime::upload_slice(
            gpu,
            "pointwise.metadata",
            &metadata,
            wgpu::BufferUsages::STORAGE,
        )?;
        let entries: Vec<_> = (0..bindings)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding != bindings - 4 && binding != bindings - 1,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let binding_layout = gpu.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pointwise.layout"),
            entries: &entries,
        });
        let pipeline_layout = gpu.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pointwise.pipeline_layout"),
            bind_group_layouts: &[&binding_layout],
            push_constant_ranges: &[],
        });
        let module = gpu.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pointwise.shader"),
            source: wgpu::ShaderSource::Wgsl(fused_source(&chain).into()),
        });
        let pipeline = gpu.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pointwise.fused"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
        });
        Ok(Self {
            chain,
            layouts,
            device,
            binding_layout,
            pipeline,
            metadata,
            grid,
        })
    }

    pub fn run(
        &self,
        inputs: &[&ResidentTensor],
        execution: PointwiseExecution,
    ) -> Result<ResidentTensor, TensorError> {
        if inputs.len() != self.layouts.len() {
            return Err(PointwiseError::Operands.into());
        }
        for (input, layout) in inputs.iter().zip(&self.layouts) {
            input.require_context(self.device.runtime().context())?;
            if input.layout() != layout {
                return Err(PointwiseError::LayoutMismatch.into());
            }
        }
        if execution == PointwiseExecution::Sequential {
            let mut current = inputs[0].clone();
            for step in self.chain.steps() {
                current = current.apply(step.op, step.rhs.map(|i| inputs[i]))?;
            }
            return Ok(current);
        }
        let context = self.device.runtime().context();
        let gpu = context.device();
        let mut encoder = gpu.create_command_encoder(&Default::default());
        let output = if execution == PointwiseExecution::Batched {
            let mut current = inputs[0].clone();
            for step in self.chain.steps() {
                let rhs = step.rhs.map_or(&current, |i| inputs[i]);
                let rhs_layout = rhs.layout.broadcast_to(self.layouts[0].shape())?;
                current = self.device.encode(
                    &mut encoder,
                    step.op,
                    Operand {
                        values: current.values(),
                        flags: current.flags(),
                        layout: current.layout(),
                    },
                    Operand {
                        values: rhs.values(),
                        flags: rhs.flags(),
                        layout: &rhs_layout,
                    },
                    self.layouts[0].shape(),
                )?;
            }
            current
        } else {
            let layout = NdLayout::contiguous(self.layouts[0].shape())?;
            let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
            let values =
                runtime::empty_buffer::<f32>(gpu, "pointwise.output", layout.len().max(1), usage)?;
            let flags = runtime::empty_buffer::<u32>(gpu, "pointwise.flags", 1, usage)?;
            let inherited = runtime::empty_buffer::<u32>(
                gpu,
                "pointwise.inherited",
                inputs.len(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )?;
            for (i, input) in inputs.iter().enumerate() {
                encoder.copy_buffer_to_buffer(input.flags(), 0, &inherited, i as u64 * 4, 4);
            }
            self.encode_into(
                &mut encoder,
                &inputs.iter().map(|t| t.values()).collect::<Vec<_>>(),
                &values,
                &inherited,
                &flags,
            );
            ResidentTensor {
                storage: Shared::new(Storage {
                    values,
                    flags: Shared::new(flags),
                }),
                layout,
                device: self.device.clone(),
            }
        };
        context.queue().submit(Some(encoder.finish()));
        Ok(output)
    }

    /// Internal graph workspace only: the caller owns same-device, exact-layout
    /// buffers and flag initialization. No mutable buffer escapes the public API.
    pub(crate) fn encode_into(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        inputs: &[&wgpu::Buffer],
        output: &wgpu::Buffer,
        inherited: &wgpu::Buffer,
        flags: &wgpu::Buffer,
    ) {
        let binding = self.bind_into(inputs, output, inherited, flags);
        self.encode_bound(encoder, &binding);
    }

    /// Bind a graph's private stable buffers once, not on every training step.
    pub(crate) fn bind_into(
        &self,
        inputs: &[&wgpu::Buffer],
        output: &wgpu::Buffer,
        inherited: &wgpu::Buffer,
        flags: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        assert_eq!(inputs.len(), self.layouts.len());
        let buffers: Vec<_> = inputs
            .iter()
            .copied()
            .chain([output, &self.metadata, inherited, flags])
            .collect();
        vjp::bind(
            self.device.runtime().context().device(),
            &self.binding_layout,
            &buffers,
        )
    }

    pub(crate) fn encode_bound(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        binding: &wgpu::BindGroup,
    ) {
        vjp::encode_bound(
            encoder,
            &self.pipeline,
            binding,
            [self.grid[0], self.grid[1]],
        );
    }

    pub(crate) fn encode_in_pass<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'a>,
        binding: &'a wgpu::BindGroup,
    ) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, binding, &[]);
        pass.dispatch_workgroups(self.grid[0], self.grid[1], 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use st_kernel_contracts::pointwise::PointwiseStep;
    #[test]
    fn generated_fusion_validates_at_program_bounds() {
        for count in [1, 3, 16] {
            let steps = (0..256)
                .map(|i| PointwiseStep {
                    op: ElementwiseOp::Add,
                    rhs: Some(i % count),
                })
                .collect();
            let source = fused_source(&PointwiseChain::new(count, steps).unwrap());
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}

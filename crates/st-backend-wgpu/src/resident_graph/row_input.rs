//! Lazily compiled row-strided first-linear input. No new tensor family or
//! mutable uniform writes: old boundary bindings retain their exact parameters.
use super::*;
use st_kernel_contracts::layout::RowMajorRows;

pub(super) struct RowInputCache {
    tile: MatmulTile,
    kind: MatmulKernel,
    accumulation: MatmulAccumulation,
    kernel: Option<DenseKernel>,
    uniform: Option<(RowMajorRows, runtime::Shared<wgpu::Buffer>)>,
    #[cfg(test)]
    pub uniform_builds: usize,
}

impl RowInputCache {
    pub fn new(tile: MatmulTile, kind: MatmulKernel, accumulation: MatmulAccumulation) -> Self {
        Self {
            tile,
            kind,
            accumulation,
            kernel: None,
            uniform: None,
            #[cfg(test)]
            uniform_builds: 0,
        }
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        template: &DenseDispatch,
        rows: RowMajorRows,
    ) -> Result<(), DenseError> {
        if self.kernel.is_none() {
            self.kernel = Some(DenseKernel::new_row_input(
                device,
                self.tile,
                self.kind,
                self.accumulation,
            )?);
        }
        if self
            .uniform
            .as_ref()
            .is_none_or(|(previous, _)| *previous != rows)
        {
            let uniform = template.row_input_uniform(device, rows)?;
            self.uniform = Some((rows, uniform));
            #[cfg(test)]
            {
                self.uniform_builds += 1;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        &self,
        device: &wgpu::Device,
        template: &DenseDispatch,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        weight: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        unused: &wgpu::Buffer,
        validation: &wgpu::Buffer,
    ) -> DenseDispatch {
        self.kernel.as_ref().unwrap().rebind_with_uniform(
            device,
            template,
            input,
            output,
            weight,
            bias,
            unused,
            validation,
            &self.uniform.as_ref().unwrap().1,
        )
    }

    pub fn encode_in_pass<'a>(
        &'a self,
        pass: &mut wgpu::ComputePass<'a>,
        binding: &'a DenseDispatch,
    ) {
        self.kernel.as_ref().unwrap().encode_in_pass(pass, binding);
    }
}

impl ResidentGraph {
    pub(super) fn prepare_direct_input(
        &mut self,
        input: &ResidentTensor,
        encoder: &mut wgpu::CommandEncoder,
        allow_rows: bool,
    ) -> Result<(ResidentTensor, bool), GraphInferenceError> {
        let layout = input.layout();
        let device = self.device.runtime().context().device();
        if allow_rows
            && (!layout.is_contiguous() || layout.offset() != 0)
            && device.limits().max_uniform_buffer_binding_size
                >= crate::resident_dense::ROW_INPUT_UNIFORM_BYTES
        {
            if let (Node::Linear(template), Some(rows)) = (&self.nodes[0], layout.row_major_rows())
            {
                self.row_input.prepare(device, template, rows)?;
                return Ok((input.clone(), true));
            }
        }
        Ok((input.contiguous_into(encoder)?, false))
    }

    pub(super) fn bind_direct_input(
        &self,
        input: &ResidentTensor,
        output: &wgpu::Buffer,
        rows: bool,
    ) -> BoundaryBinding {
        if rows {
            let (Node::Linear(template), GraphStage::Linear { weight, bias, .. }) =
                (&self.nodes[0], &self.definition.stages()[0])
            else {
                unreachable!("row input requires a first linear stage");
            };
            BoundaryBinding::RowLinear(self.row_input.bind(
                self.device.runtime().context().device(),
                template,
                input.values(),
                output,
                &self.parameters[*weight],
                &self.parameters[*bias],
                &self.unused,
                &self.validation,
            ))
        } else {
            self.bind_boundary(0, input.values(), output)
        }
    }
}

#[cfg(test)]
mod tests;

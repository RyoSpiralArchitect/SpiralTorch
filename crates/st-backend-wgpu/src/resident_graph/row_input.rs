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

#[cfg(test)]
mod tests;

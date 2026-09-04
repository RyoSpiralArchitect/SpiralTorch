use super::*;

/// Exact integer row gather. Unlike legacy embedding utilities, IDs stay u32 on GPU.
pub fn gather_rows(
    input: &[f32],
    indices: &[usize],
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    indexing(input, indices, rows, cols, 1.0, false)
}

/// Deterministic grouped scatter; O((rows + tokens) * cols), no float atomics.
pub fn scatter_add_rows(
    input: &[f32],
    indices: &[usize],
    rows: usize,
    cols: usize,
    scale: f32,
) -> Result<Vec<f32>, String> {
    indexing(input, indices, rows, cols, scale, true)
}

fn indexing(
    input: &[f32],
    indices: &[usize],
    rows: usize,
    cols: usize,
    scale: f32,
    scatter: bool,
) -> Result<Vec<f32>, String> {
    let invalid = || "row indexing dimensions exceed supported range".to_owned();
    let as_u32 = |value| u32::try_from(value).map_err(|_| invalid());
    let volume = |a: usize, b: usize| a.checked_mul(b).ok_or_else(invalid);
    let tokens = indices.len();
    let expected = volume(if scatter { tokens } else { rows }, cols)?;
    let output_elements = volume(if scatter { rows } else { tokens }, cols)?;
    let output_u32 = as_u32(output_elements)?;
    as_u32(expected)?;
    let rows_u32 = as_u32(rows)?;
    let cols_u32 = as_u32(cols)?;
    let tokens_u32 = as_u32(tokens)?;
    if input.len() != expected || indices.iter().any(|&id| id >= rows) || !scale.is_finite() {
        return Err(
            "row indexing requires matching dimensions, in-range IDs and finite scale".into(),
        );
    }
    if tokens == 0 || rows == 0 || cols == 0 {
        return Err("row indexing GPU dispatch requires non-empty operands".into());
    }
    if scatter && input.iter().any(|value| !value.is_finite()) {
        return Err("scatter_add_rows requires finite input".into());
    }
    let transport_len = if scatter {
        rows.checked_add(1)
            .and_then(|n| n.checked_add(tokens))
            .ok_or_else(invalid)?
    } else {
        tokens
    };
    as_u32(transport_len)?;
    let ctx = dense_context()?;
    let device = ctx.device();
    let queue = ctx.queue();
    let limits = device.limits();
    let groups = output_u32.div_ceil(TENSOR_UTIL_WORKGROUP);
    if groups > limits.max_compute_workgroups_per_dimension {
        return Err("row indexing dispatch exceeds device workgroup limit".into());
    }
    for len in [transport_len, input.len(), output_elements] {
        let bytes = len.checked_mul(4).ok_or_else(invalid)? as u64;
        if bytes > limits.max_buffer_size
            || bytes > u64::from(limits.max_storage_buffer_binding_size)
        {
            return Err("row indexing buffer exceeds device storage limit".into());
        }
    }
    let transport: Vec<u32> = if scatter {
        let (offsets, positions) =
            crate::indexing::grouped_rows(indices, rows).map_err(|e| e.to_string())?;
        offsets
            .into_iter()
            .chain(positions)
            .map(as_u32)
            .collect::<Result<_, _>>()?
    } else {
        indices
            .iter()
            .copied()
            .map(as_u32)
            .collect::<Result<_, _>>()?
    };
    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.row_indexing.indices_u32"),
        contents: bytemuck::cast_slice(&transport),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let input_buf = upload_lhs(device, "st.tensor.row_indexing.input", input);
    let output_buf = allocate_output(device, "st.tensor.row_indexing.output", output_elements);
    let params = TensorUtilParams {
        rows: tokens_u32,
        cols: cols_u32,
        values: output_u32,
        flags: rows_u32,
        scalar: scale,
        saturation: 0.0,
        porosity: 0.0,
        _pad2: 0.0,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("st.tensor.row_indexing.params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let pipeline = ctx.tensor_util_pipeline(if scatter {
        TensorUtilKernel::ScatterAddRows
    } else {
        TensorUtilKernel::GatherRows
    })?;
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("st.tensor.row_indexing.bind_group"),
        layout: &ctx.tensor_util_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: index_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: input_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buf.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("st.tensor.row_indexing.encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("st.tensor.row_indexing.pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    let result = readback_f32(device, queue, &output_buf, output_elements)?;
    if result.iter().any(|value| !value.is_finite()) {
        return Err("row indexing produced nonfinite output".into());
    }
    Ok(result)
}

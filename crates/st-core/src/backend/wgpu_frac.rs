// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.


//! WGPU fractional ops wrapper: GL 1D conv and spectral power multiply (fractional Laplacian kernel).
//! Feature-gated with `wgpu`. Callers must provide `Device`/`Queue` and pack axis-last contiguous.

#[cfg(feature="wgpu")]
pub mod wgpu_frac {
    use ndarray::{ArrayD, IxDyn, Axis};
    use std::time::Duration;
    use wgpu::util::DeviceExt;

    use super::wgpu_rt;

    fn flatten_rows_cols(x:&ArrayD<f32>, axis:usize) -> (Vec<f32>, u32, u32, usize) {
        // Permute so that axis is last, then view as [rows, cols]
        let mut perm: Vec<usize> = (0..x.ndim()).collect();
        perm.retain(|&d| d!=axis);
        perm.push(axis);
        let xp = x.view().permuted_axes(perm.clone());
        let cols = xp.shape()[xp.ndim()-1] as u32;
        let rows = (xp.len() as u32) / cols;
        let data: Vec<f32> = xp.iter().cloned().collect();
        (data, rows, cols, axis)
    }

    pub fn fracdiff_gl_wgpu(device:&wgpu::Device, queue:&wgpu::Queue,
        x:&ArrayD<f32>, coeff:&[f32], alpha_scale:f32, axis:usize, pad_zero:bool
    ) -> ArrayD<f32> {
        let (data, rows, cols, _) = flatten_rows_cols(x, axis);
        let n = (rows as usize)*(cols as usize);

        // Buffers
        let buf_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("fracdiff.X"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let buf_c = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("fracdiff.C"),
            contents: bytemuck::cast_slice(coeff),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut y = vec![0f32; n];
        let buf_y = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("fracdiff.Y"),
            contents: bytemuck::cast_slice(&y),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params { cols:u32, rows:u32, klen:u32, alpha_scale:f32, pad_mode:u32, _pad:[u32;3] }
        let params = Params{ cols, rows, klen: coeff.len() as u32, alpha_scale, pad_mode: if pad_zero{0}else{1}, _pad:[0;3]};
        let buf_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("fracdiff.Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("wgpu_fracdiff_gl.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgpu_shaders/wgpu_fracdiff_gl.wgsl").into()),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("fracdiff.bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry{ binding:0, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer{ read_only:true, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry{ binding:1, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer{ read_only:true, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry{ binding:2, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer{ read_only:false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry{ binding:3, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::UniformBuffer{ min_binding_size: None }, count: None },
            ]
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("fracdiff.pipeline_layout"), bind_group_layouts: &[&bind_layout], push_constant_ranges: &[]
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("fracdiff.pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("fracdiff.bind"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: buf_x.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: buf_c.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: buf_y.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:3, resource: buf_p.as_entire_binding() },
            ]
        });

        // Dispatch: one thread per (row,col) element is overkill; we loop columns in shader.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("fracdiff.encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("fracdiff.pass") });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind, &[]);
            // grid over rows; columns are looped inside
            let wg_rows = ((rows + 127) / 128) as u32;
            cpass.dispatch_workgroups(wg_rows, 1, 1);
        }

        // Readback
        let buf_read = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("fracdiff.read"),
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buf_y, 0, &buf_read, 0, (n * std::mem::size_of::<f32>()) as u64);
        let cmd = encoder.finish();
        let cmd_bufs = [cmd];
        if let Err(e) = wgpu_rt::st_submit_with_timeout(device, queue, &cmd_bufs, Duration::from_secs(30)) {
            panic!("wgpu fracdiff submit failed: {e}");
        }
        let bytes = wgpu_rt::st_map_read_with_timeout(
            device,
            &buf_read,
            0..(n * std::mem::size_of::<f32>()) as u64,
            Duration::from_secs(30),
        )
        .unwrap_or_else(|e| panic!("wgpu fracdiff readback failed: {e}"));
        let out: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();

        // Reshape back: [rows, cols] → original permuted axes
        let y = ArrayD::from_shape_vec(IxDyn(&[rows as usize, cols as usize]), out).unwrap();
        // We leave it in [rows, cols] (axis-last) for simplicity in this wrapper.
        y.into_dyn()
    }

    #[repr(C)] #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct SpecParams { cols:u32, rows:u32, power:f32, _pad:u32 }

    pub fn specmul_frac_laplace_wgpu(device:&wgpu::Device, queue:&wgpu::Queue,
        x_fft_reim:&[f32], rows:u32, cols:u32, power:f32
    ) -> Vec<f32> {
        // x_fft_reim: interleaved (re,im) length rows*cols*2 as f32
        let n_cplx = (rows as usize) * (cols as usize);
        // Buffers
        let buf_x = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("specmul.X"),
            contents: bytemuck::cast_slice(x_fft_reim),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut y = vec![0f32; n_cplx*2];
        let buf_y = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("specmul.Y"),
            contents: bytemuck::cast_slice(&y),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });
        let params = SpecParams{ cols, rows, power, _pad:0 };
        let buf_p = device.create_buffer_init(&wgpu::util::BufferInitDescriptor{
            label: Some("specmul.Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        // Pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor{
            label: Some("wgpu_frac_specmul.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("wgpu_shaders/wgpu_frac_specmul.wgsl").into()),
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("specmul.bind_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry{ binding:0, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer{ read_only:true, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry{ binding:1, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer{ read_only:false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry{ binding:2, visibility:wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::UniformBuffer{ min_binding_size: None }, count: None },
            ]
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label: Some("specmul.pipeline_layout"), bind_group_layouts: &[&bind_layout], push_constant_ranges: &[]
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor{
            label: Some("specmul.pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader, entry_point: "main",
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label: Some("specmul.bind"),
            layout: &bind_layout,
            entries: &[
                wgpu::BindGroupEntry{ binding:0, resource: buf_x.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:1, resource: buf_y.as_entire_binding() },
                wgpu::BindGroupEntry{ binding:2, resource: buf_p.as_entire_binding() },
            ]
        });
        // Dispatch over rows×cols grid
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor{ label: Some("specmul.encoder") });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor{ label: Some("specmul.pass") });
            cpass.set_pipeline(&pipeline); cpass.set_bind_group(0, &bind, &[]);
            let wg_x = ((rows + 127)/128) as u32;
            let wg_y = ((cols + 127)/128) as u32; // we index y in shader's gid.y so use 1D groups but 2D dispatch
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        // Readback
        let buf_read = device.create_buffer(&wgpu::BufferDescriptor{
            label: Some("specmul.read"),
            size: (n_cplx*2*std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&buf_y, 0, &buf_read, 0, (n_cplx*2*std::mem::size_of::<f32>()) as u64);
        let cmd = encoder.finish();
        let cmd_bufs = [cmd];
        if let Err(e) = wgpu_rt::st_submit_with_timeout(device, queue, &cmd_bufs, Duration::from_secs(30)) {
            panic!("wgpu specmul submit failed: {e}");
        }
        let bytes = wgpu_rt::st_map_read_with_timeout(
            device,
            &buf_read,
            0..(n_cplx * 2 * std::mem::size_of::<f32>()) as u64,
            Duration::from_secs(30),
        )
        .unwrap_or_else(|e| panic!("wgpu specmul readback failed: {e}"));
        bytemuck::cast_slice(&bytes).to_vec()
    }
}

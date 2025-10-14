// crates/st-tensor/src/util.rs
#![cfg(feature = "wgpu_frac")]
use wgpu::*;

pub fn readback_f32(device: &Device, queue: &Queue, src: &Buffer, len: usize) -> Vec<f32> {
    let size_bytes = (len * std::mem::size_of::<f32>()) as u64;
    let rb = device.create_buffer(&BufferDescriptor{
        label: Some("readback"),
        size: size_bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&CommandEncoderDescriptor{ label: Some("readback-enc") });
    enc.copy_buffer_to_buffer(src, 0, &rb, 0, size_bytes);
    queue.submit(Some(enc.finish()));

    let slice = rb.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(Maintain::Wait);
    receiver.recv().expect("map_async callback dropped").expect("buffer map failed");

    let data = slice.get_mapped_range();
    let mut out = vec![0.0f32; len];
    out.copy_from_slice(bytemuck::cast_slice(&data));
    drop(data);
    rb.unmap();
    out
}

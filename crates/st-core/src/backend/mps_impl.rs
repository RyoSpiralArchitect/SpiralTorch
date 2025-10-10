
use crate::device::Device;
use crate::error::{Result, device as dev_err};
use crate::backend::{Backend, BackendArrayF32};

use ndarray::ArrayD;
use once_cell::sync::OnceCell;
use std::{ptr, ffi::c_void};

const MSL_SRC: &str = include_str!("mps_kernels.metal");

pub struct MpsBackend;
impl MpsBackend { pub fn new() -> Self { MpsBackend } }

struct Ctx { device: metal::Device, queue: metal::CommandQueue }
static CTX: OnceCell<Ctx> = OnceCell::new();
fn ctx() -> &'static Ctx {
    CTX.get_or_init(|| {
        let device = metal::Device::system_default().expect("No MTLDevice");
        let queue = device.new_command_queue();
        Ctx { device, queue }
    })
}

// ===== Pool (env-tunable) =====
fn pool_limits() -> (u64, usize) {
    static LIM: OnceCell<(u64, usize)> = OnceCell::new();
    *LIM.get_or_init(|| {
        let mb  = std::env::var("SPIRALTORCH_MPS_POOL_MAX_MB").ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(256);
        let per = std::env::var("SPIRALTORCH_MPS_POOL_MAX_PER_CLASS").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(32);
        (mb * 1024 * 1024, per)
    })
}
fn round_pow2(x: u64) -> u64 { let mut n=256u64; while n<x { n<<=1; } n }

struct BufferPool {
    bytes_total: u64,
    classes: std::collections::BTreeMap<u64, std::collections::VecDeque<metal::Buffer>>,
}
static POOL: OnceCell<std::sync::Mutex<BufferPool>> = OnceCell::new();
fn pool() -> &'static std::sync::Mutex<BufferPool> {
    POOL.get_or_init(|| std::sync::Mutex::new(BufferPool { bytes_total: 0, classes: std::collections::BTreeMap::new() }))
}
fn temp_buffer(bytes_req: u64) -> metal::Buffer {
    let dev = &ctx().device;
    let class = round_pow2(bytes_req);
    let mut p = pool().lock().unwrap();
    if let Some(q) = p.classes.get_mut(&class) {
        if let Some(buf) = q.pop_front() { return buf; }
    }
    drop(p);
    let buf = dev.new_buffer(class, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
    let (max_bytes, _max_per) = pool_limits();
    let mut p2 = pool().lock().unwrap();
    p2.bytes_total += class;
    while p2.bytes_total > max_bytes {
        let mut evicted=false;
        let keys: Vec<u64> = p2.classes.keys().cloned().collect();
        for k in keys.into_iter().rev() {
            let q = p2.classes.get_mut(&k).unwrap();
            if let Some(_b) = q.pop_back() {
                p2.bytes_total -= k;
                evicted=true;
                break;
            }
        }
        if !evicted { break; }
    }
    buf
}

// ===== Pipelines =====
struct Pipes {
    lib: metal::Library,
    p_red_nd_wg: metal::ComputePipelineState,
    p_red_nd_part: metal::ComputePipelineState,
    p_red_nd_final: metal::ComputePipelineState,
}
static PIPES: OnceCell<Pipes> = OnceCell::new();
fn pipes() -> &'static Pipes {
    PIPES.get_or_init(|| {
        let dev = &ctx().device;
        let opts = metal::CompileOptions::new();
        let lib = dev.new_library_with_source(MSL_SRC, &opts).expect("compile metal");
        macro_rules! f { ($n:literal) => { lib.get_function($n, None).unwrap() } }
        macro_rules! p { ($f:expr) => { dev.new_compute_pipeline_state_with_function(&$f).unwrap() } }
        let p_red_nd_wg = p!(f!("reduce_nd_wg_sum"));
        let p_red_nd_part = p!(f!("reduce_nd_wg_sum_partials"));
        let p_red_nd_final = p!(f!("reduce_nd_wg_sum_finalize"));
        Pipes { lib, p_red_nd_wg, p_red_nd_part, p_red_nd_final }
    })
}

// ===== Backend trait subset =====
impl Backend for MpsBackend {
    fn name(&self) -> &'static str { "mps" }
    fn device(&self) -> Device { Device::Mps }

    fn from_host_f32(&self, host: &ArrayD<f32>) -> Result<BackendArrayF32> {
        let len = host.len();
        let bytes = (len * std::mem::size_of::<f32>()) as u64;
        let dev = &ctx().device;
        let buf = dev.new_buffer(bytes, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        unsafe {
            let ptr_dst = buf.contents() as *mut c_void;
            let src: *const c_void = host.as_ptr() as *const c_void;
            std::ptr::copy_nonoverlapping(src, ptr_dst, bytes as usize);
        }
        Ok(BackendArrayF32::Mps { rows: len, cols: 1, buffer: buf })
    }

    fn to_host_f32(&self, arr: &BackendArrayF32) -> Result<ArrayD<f32>> {
        match arr {
            BackendArrayF32::Mps { rows, cols, buffer } => {
                let len = rows * cols;
                let mut vec = vec![0f32; len];
                unsafe { std::ptr::copy_nonoverlapping(buffer.contents() as *const f32, vec.as_mut_ptr(), len); }
                Ok(ndarray::Array1::from_vec(vec).into_dyn())
            }
        }
    }
}

// ===== Public structs for reduce
#[repr(C)]
pub struct NdWGInfo {
    pub n_rows: u32,
    pub n_cols: u32,
    pub kdims: u32,
    pub rdims: u32,
    pub kshape: [u32; 6],
    pub rshape: [u32; 6],
    pub kstride: [i32; 6],
    pub rstride: [i32; 6],
}

// ===== Helpers
fn grid_1d(n: usize, pso: &metal::ComputePipelineState) -> (metal::MTLSize, metal::MTLSize) {
    let w = pso.thread_execution_width().max(32);
    let tg = metal::MTLSize::new(w as u64, 1, 1);
    let grid = metal::MTLSize::new(n as u64, 1, 1);
    (grid, tg)
}

impl MpsBackend {
    pub fn add(&self, a: &BackendArrayF32, b: &BackendArrayF32) -> Result<BackendArrayF32> {
        let ha = self.to_host_f32(a)?;
        let hb = self.to_host_f32(b)?;
        let y = ha + hb;
        self.from_host_f32(&y)
    }

    // GEMM fwd 2D (MPSMatrix)
    pub fn matmul2d_matrix(&self, a: &BackendArrayF32, b: &BackendArrayF32, m: usize, k: usize, n: usize) -> Result<BackendArrayF32> {
        use objc::runtime::Object;
        let dev = &ctx().device; let q=&ctx().queue;
        let bytes_c = (m*n*std::mem::size_of::<f32>()) as u64;
        let cbuf = temp_buffer(bytes_c);
        unsafe {
            let dtype: u64 = 1;
            let desc_cls = objc::class!(MPSMatrixDescriptor);
            let mtx_cls = objc::class!(MPSMatrix);
            let mm_cls = objc::class!(MPSMatrixMultiplication);
            let rb_a = (k * std::mem::size_of::<f32>()) as u64;
            let rb_b = (n * std::mem::size_of::<f32>()) as u64;
            let rb_c = (n * std::mem::size_of::<f32>()) as u64;
            let desc_a: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: m as u64 columns: k as u64 rowBytes: rb_a dataType: dtype];
            let desc_b: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: k as u64 columns: n as u64 rowBytes: rb_b dataType: dtype];
            let desc_c: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: m as u64 columns: n as u64 rowBytes: rb_c dataType: dtype];
            let (abuf, bbuf) = match (a, b) {
                (BackendArrayF32::Mps{buffer: ab, ..}, BackendArrayF32::Mps{buffer: bb, ..}) => (ab.as_ptr(), bb.as_ptr()),
            };
            let cptr = cbuf.as_ptr();
            let mut m_a: *mut Object = objc::msg_send![mtx_cls, alloc];
            m_a = objc::msg_send![m_a, initWithBuffer: abuf offset: 0u64 descriptor: desc_a];
            let mut m_b: *mut Object = objc::msg_send![mtx_cls, alloc];
            m_b = objc::msg_send![m_b, initWithBuffer: bbuf offset: 0u64 descriptor: desc_b];
            let mut m_c: *mut Object = objc::msg_send![mtx_cls, alloc];
            m_c = objc::msg_send![m_c, initWithBuffer: cptr offset: 0u64 descriptor: desc_c];
            let mut mm: *mut Object = objc::msg_send![mm_cls, alloc];
            mm = objc::msg_send![mm, initWithDevice: dev.as_ptr()
                                         resultRows: m as u64 resultColumns: n as u64 interiorColumns: k as u64
                                         alpha: 1.0f32 beta: 0.0f32];
            let cb = q.new_command_buffer();
            let () = objc::msg_send![mm, encodeToCommandBuffer: cb.as_ptr() leftMatrix: m_a rightMatrix: m_b resultMatrix: m_c];
            cb.commit(); cb.wait_until_completed();
        }
        Ok(BackendArrayF32::Mps { rows: m, cols: n, buffer: cbuf })
    }

    // Batched GEMM fwd
    pub fn matmul2d_matrix_batched(&self, a: &BackendArrayF32, b: &BackendArrayF32, bsz: usize, m: usize, k: usize, n: usize) -> Result<BackendArrayF32> {
        use objc::runtime::Object;
        let dev = &ctx().device; let q=&ctx().queue;
        unsafe {
            let dtype: u64 = 1;
            let desc_cls = objc::class!(MPSMatrixDescriptor);
            let mtx_cls = objc::class!(MPSMatrix);
            let mm_cls = objc::class!(MPSMatrixMultiplication);

            let rb_a = (k * std::mem::size_of::<f32>()) as u64;
            let rb_b = (n * std::mem::size_of::<f32>()) as u64;
            let rb_c = (n * std::mem::size_of::<f32>()) as u64;

            let desc_a: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: m as u64 columns: k as u64 rowBytes: rb_a dataType: dtype];
            let desc_b: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: k as u64 columns: n as u64 rowBytes: rb_b dataType: dtype];
            let desc_c: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: m as u64 columns: n as u64 rowBytes: rb_c dataType: dtype];

            let (abuf, bbuf) = match (a, b) {
                (BackendArrayF32::Mps{buffer: ab, ..}, BackendArrayF32::Mps{buffer: bb, ..}) => (ab.as_ptr(), bb.as_ptr()),
            };
            let stride_a = (m*k*std::mem::size_of::<f32>()) as u64;
            let stride_b = (k*n*std::mem::size_of::<f32>()) as u64;
            let stride_c = (m*n*std::mem::size_of::<f32>()) as u64;
            let cbuf = temp_buffer(stride_c * bsz as u64);

            let cb = q.new_command_buffer();
            for i in 0..bsz {
                let mut m_a: *mut Object = objc::msg_send![mtx_cls, alloc];
                m_a = objc::msg_send![m_a, initWithBuffer: abuf offset: (i as u64)*stride_a descriptor: desc_a];
                let mut m_b: *mut Object = objc::msg_send![mtx_cls, alloc];
                m_b = objc::msg_send![m_b, initWithBuffer: bbuf offset: (i as u64)*stride_b descriptor: desc_b];
                let mut m_c: *mut Object = objc::msg_send![mtx_cls, alloc];
                m_c = objc::msg_send![m_c, initWithBuffer: cbuf.as_ptr() offset: (i as u64)*stride_c descriptor: desc_c];

                let mut mm: *mut Object = objc::msg_send![mm_cls, alloc];
                mm = objc::msg_send![mm, initWithDevice: dev.as_ptr()
                                              resultRows: m as u64 resultColumns: n as u64 interiorColumns: k as u64
                                              alpha: 1.0f32 beta: 0.0f32];
                let () = objc::msg_send![mm, encodeToCommandBuffer: cb.as_ptr() leftMatrix: m_a rightMatrix: m_b resultMatrix: m_c];
            }
            cb.commit(); cb.wait_until_completed();

            Ok(BackendArrayF32::Mps { rows: bsz*m, cols: n, buffer: cbuf })
        }
    }

    // ND reduce: 1-pass
    pub fn reduce_nd_wg_sum(&self, x: &BackendArrayF32, info: &NdWGInfo) -> Result<BackendArrayF32> {
        use std::mem::size_of;
        let dev = &ctx().device; let q=&ctx().queue;
        let lib = crate::backend::mps_impl::pipes();
        let out = temp_buffer((info.n_rows as usize * std::mem::size_of::<f32>()) as u64);
        let ib = dev.new_buffer(size_of::<NdWGInfo>() as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        unsafe { std::ptr::copy_nonoverlapping(info as *const NdWGInfo as *const u8, ib.contents() as *mut u8, size_of::<NdWGInfo>()); }
        let grid = metal::MTLSize::new(info.n_rows as u64 * 256, 1, 1);
        let tg   = metal::MTLSize::new(256, 1, 1);
        let cb = q.new_command_buffer(); let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&lib.p_red_nd_wg);
        match x { BackendArrayF32::Mps { buffer, .. } => enc.set_buffer(0, Some(buffer), 0) }
        enc.set_buffer(1, Some(&out), 0);
        enc.set_buffer(2, Some(&ib), 0);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cb.commit(); cb.wait_until_completed();
        Ok(BackendArrayF32::Mps { rows: info.n_rows as usize, cols: 1, buffer: out })
    }

    // ND reduce: choose 1 or 2 pass
    pub fn reduce_nd_sum_auto(&self, x: &BackendArrayF32, info: &NdWGInfo) -> Result<BackendArrayF32> {
        let n_cols = info.n_cols as usize;
        if n_cols <= 16384 { return self.reduce_nd_wg_sum(x, info); }
        self.reduce_nd_wg_sum_2pass(x, info, ((n_cols + 16383)/16384).max(2).min(1024) as u32)
    }

    pub fn reduce_nd_wg_sum_2pass(&self, x: &BackendArrayF32, info: &NdWGInfo, groups: u32) -> Result<BackendArrayF32> {
        use std::mem::size_of;
        let dev = &ctx().device; let q=&ctx().queue;
        let lib = crate::backend::mps_impl::pipes();
        let cols_per = ((info.n_cols + groups - 1) / groups).max(1);
        let partials = temp_buffer((info.n_rows as usize * groups as usize * std::mem::size_of::<f32>()) as u64);
        #[repr(C)] struct Part { base: NdWGInfo, groups: u32, cols_per: u32 }
        let part = Part { base: *info, groups, cols_per };
        let pib = dev.new_buffer(size_of::<Part>() as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        unsafe { std::ptr::copy_nonoverlapping(&part as *const Part as *const u8, pib.contents() as *mut u8, size_of::<Part>()); }

        let grid1 = metal::MTLSize::new(info.n_rows as u64 * groups as u64 * 256, 1, 1);
        let tg = metal::MTLSize::new(256, 1, 1);
        let cb1 = q.new_command_buffer(); let e1 = cb1.new_compute_command_encoder();
        e1.set_compute_pipeline_state(&lib.p_red_nd_part);
        match x { BackendArrayF32::Mps { buffer, .. } => e1.set_buffer(0, Some(buffer), 0) }
        e1.set_buffer(1, Some(&partials), 0);
        e1.set_buffer(2, Some(&pib), 0);
        e1.dispatch_threads(grid1, tg);
        e1.end_encoding(); cb1.commit(); cb1.wait_until_completed();

        let out = temp_buffer((info.n_rows as usize * std::mem::size_of::<f32>()) as u64);
        let rb = dev.new_buffer(size_of::<u32>() as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let gb = dev.new_buffer(size_of::<u32>() as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        unsafe { *(rb.contents() as *mut u32) = info.n_rows; *(gb.contents() as *mut u32) = groups; }
        let grid2 = metal::MTLSize::new(info.n_rows as u64 * 256, 1, 1);
        let cb2 = q.new_command_buffer(); let e2 = cb2.new_compute_command_encoder();
        e2.set_compute_pipeline_state(&lib.p_red_nd_final);
        e2.set_buffer(0, Some(&partials), 0);
        e2.set_buffer(1, Some(&out), 0);
        e2.set_buffer(2, Some(&rb), 0);
        e2.set_buffer(3, Some(&gb), 0);
        e2.dispatch_threads(grid2, tg);
        e2.end_encoding(); cb2.commit(); cb2.wait_until_completed();

        Ok(BackendArrayF32::Mps { rows: info.n_rows as usize, cols: 1, buffer: out })
    }

    pub fn transpose2d(&self, x: &BackendArrayF32, rows: usize, cols: usize) -> Result<BackendArrayF32> {
        let dev = &ctx().device; let q=&ctx().queue;
        let lib = crate::backend::mps_impl::pipes();
        let out = temp_buffer((rows*cols*std::mem::size_of::<f32>()) as u64);
        let rb = dev.new_buffer(std::mem::size_of::<u32>() as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        let cb = dev.new_buffer(std::mem::size_of::<u32>() as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        unsafe { *(rb.contents() as *mut u32) = rows as u32; *(cb.contents() as *mut u32) = cols as u32; }
        let (grid,tg) = crate::backend::mps_impl::grid_1d(rows*cols, &lib.p_red_nd_wg);
        let cmd = q.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&lib.p_red_nd_wg); // NB: to keep this example minimal we reuse pso; replace with dedicated transpose pso if needed
        match x { BackendArrayF32::Mps{buffer, ..} => enc.set_buffer(0, Some(buffer), 0) }
        enc.set_buffer(1, Some(&out), 0); enc.set_buffer(2, Some(&rb), 0); enc.set_buffer(3, Some(&cb), 0);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding(); cmd.commit(); cmd.wait_until_completed();
        Ok(BackendArrayF32::Mps { rows: cols, cols: rows, buffer: out })
    }
}

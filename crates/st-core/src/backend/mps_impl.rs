
use ndarray::ArrayD;
use once_cell::sync::OnceCell;
use crate::{device::Device, error::{Result, device as dev_err}};

pub struct MpsBackend;
impl MpsBackend { pub fn new() -> Self { MpsBackend } }

pub enum BackendArrayF32 {
    Mps { rows: usize, cols: usize, buffer: metal::Buffer },
    #[allow(dead_code)] HostStub,
}

pub trait Backend {
    fn name(&self)->&'static str;
    fn device(&self)->Device;
    fn from_host_f32(&self, host:&ArrayD<f32>) -> Result<BackendArrayF32>;
    fn to_host_f32(&self, arr:&BackendArrayF32) -> Result<ArrayD<f32>>;
}

struct Ctx { device: metal::Device, queue: metal::CommandQueue }
static CTX: OnceCell<Ctx> = OnceCell::new();
fn ctx()->&'static Ctx{
    CTX.get_or_init(|| {
        let device = metal::Device::system_default().expect("No MTLDevice");
        let queue = device.new_command_queue();
        Ctx{ device, queue }
    })
}

impl Backend for MpsBackend {
    fn name(&self)->&'static str { "mps" }
    fn device(&self)->Device { Device::Mps }
    fn from_host_f32(&self, host:&ArrayD<f32>) -> Result<BackendArrayF32> {
        let len = host.len(); let bytes = (len*4) as u64;
        let buf = ctx().device.new_buffer(bytes, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
        unsafe { std::ptr::copy_nonoverlapping(host.as_ptr(), buf.contents() as *mut f32, len); }
        Ok(BackendArrayF32::Mps{ rows: len, cols:1, buffer: buf })
    }
    fn to_host_f32(&self, arr:&BackendArrayF32) -> Result<ArrayD<f32>> {
        match arr {
            BackendArrayF32::Mps{ rows, cols, buffer } => {
                let len = rows*cols; let mut v=vec![0f32; len];
                unsafe { std::ptr::copy_nonoverlapping(buffer.contents() as *const f32, v.as_mut_ptr(), len); }
                Ok(ndarray::Array1::from_vec(v).into_dyn())
            }
            _ => Err(dev_err("to_host: non-mps"))
        }
    }
}

impl MpsBackend {
    /// Beta-accumulating batched backward (broadcast-friendly).
    pub fn matmul2d_batch_bwd(
        &self,
        go: &BackendArrayF32,  // [bsz, m, n]
        a:  &BackendArrayF32,  // [ba (1|bsz), m, k]
        b:  &BackendArrayF32,  // [bb (1|bsz), k, n]
        bsz: usize, m: usize, k: usize, n: usize,
        ba: usize, bb: usize,
    ) -> Result<(BackendArrayF32, BackendArrayF32)> {
        use objc::runtime::Object;
        let dev = &ctx().device; let q = &ctx().queue;
        unsafe {
            let desc_cls = objc::class!(MPSMatrixDescriptor);
            let mtx_cls  = objc::class!(MPSMatrix);
            let tr_cls   = objc::class!(MPSMatrixTranspose);
            let mm_cls   = objc::class!(MPSMatrixMultiplication);
            let dtype:u64 = 1;
            let rb_go = (n*4) as u64;
            let rb_b  = (n*4) as u64;
            let rb_bt = (k*4) as u64;
            let rb_a  = (k*4) as u64;
            let rb_at = (m*4) as u64;
            let rb_da = (k*4) as u64;
            let rb_db = (n*4) as u64;
            let d_go: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: m as u64 columns: n as u64 rowBytes: rb_go dataType: dtype];
            let d_b : *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: k as u64 columns: n as u64 rowBytes: rb_b  dataType: dtype];
            let d_bt: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: n as u64 columns: k as u64 rowBytes: rb_bt dataType: dtype];
            let d_a : *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: m as u64 columns: k as u64 rowBytes: rb_a  dataType: dtype];
            let d_at: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: k as u64 columns: m as u64 rowBytes: rb_at dataType: dtype];
            let d_da: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: m as u64 columns: k as u64 rowBytes: rb_da dataType: dtype];
            let d_db: *mut Object = objc::msg_send![desc_cls, matrixDescriptorWithRows: k as u64 columns: n as u64 rowBytes: rb_db dataType: dtype];

            let (gobuf, abuf, bbuf) = match (go,a,b) {
                (BackendArrayF32::Mps{ buffer:g, .. },
                 BackendArrayF32::Mps{ buffer:a, .. },
                 BackendArrayF32::Mps{ buffer:b, .. }) => (g.as_ptr(), a.as_ptr(), b.as_ptr()),
                _ => return Err(dev_err("matmul2d_batch_bwd: non-MPS input"))
            };

            // Output buffers + temporaries (simple pool elided)
            let da_buf = ctx().device.new_buffer((m*k*4) as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
            let db_buf = ctx().device.new_buffer((k*n*4) as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
            std::ptr::write_bytes(da_buf.contents(), 0, (m*k*4) as usize);
            std::ptr::write_bytes(db_buf.contents(), 0, (k*n*4) as usize);
            let bt_buf = ctx().device.new_buffer((n*k*4) as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);
            let at_buf = ctx().device.new_buffer((k*m*4) as u64, metal::MTLResourceOptions::CPUCacheModeDefaultCache);

            let mut tr: *mut Object = objc::msg_send![tr_cls, alloc];
            tr = objc::msg_send![tr, initWithDevice: dev.as_ptr()];

            let mut mm_da: *mut Object = objc::msg_send![mm_cls, alloc];
            mm_da = objc::msg_send![mm_da, initWithDevice: dev.as_ptr()
                                resultRows: m as u64 resultColumns: k as u64 interiorColumns: n as u64
                                alpha: 1.0f32 beta: 1.0f32];
            let mut mm_db: *mut Object = objc::msg_send![mm_cls, alloc];
            mm_db = objc::msg_send![mm_db, initWithDevice: dev.as_ptr()
                                resultRows: k as u64 resultColumns: n as u64 interiorColumns: m as u64
                                alpha: 1.0f32 beta: 1.0f32];

            let cb = q.new_command_buffer();

            // dA accumulate if ba==1
            if ba == 1 {
                let mut c_da: *mut Object = objc::msg_send![mtx_cls, alloc];
                c_da = objc::msg_send![c_da, initWithBuffer: da_buf.as_ptr() offset: 0u64 descriptor: d_da];
                for i in 0..(bsz as u64) {
                    let mut g_i: *mut Object = objc::msg_send![mtx_cls, alloc];
                    g_i = objc::msg_send![g_i, initWithBuffer: gobuf offset: i*(m*n*4) as u64 descriptor: d_go];
                    let mut b_i: *mut Object = objc::msg_send![mtx_cls, alloc];
                    let off_b = if bb==1 {0} else { i*(k*n*4) as u64 };
                    b_i = objc::msg_send![b_i, initWithBuffer: bbuf offset: off_b descriptor: d_b];
                    let mut bt_m: *mut Object = objc::msg_send![mtx_cls, alloc];
                    bt_m = objc::msg_send![bt_m, initWithBuffer: bt_buf.as_ptr() offset: 0u64 descriptor: d_bt];
                    let () = objc::msg_send![tr, encodeToCommandBuffer: cb.as_ptr() sourceMatrix: b_i resultMatrix: bt_m];
                    let () = objc::msg_send![mm_da, encodeToCommandBuffer: cb.as_ptr() leftMatrix: g_i rightMatrix: bt_m resultMatrix: c_da];
                }
            }

            // dB accumulate if bb==1
            if bb == 1 {
                let mut c_db: *mut Object = objc::msg_send![mtx_cls, alloc];
                c_db = objc::msg_send![c_db, initWithBuffer: db_buf.as_ptr() offset: 0u64 descriptor: d_db];
                for i in 0..(bsz as u64) {
                    let mut a_i: *mut Object = objc::msg_send![mtx_cls, alloc];
                    let off_a = if ba==1 {0} else { i*(m*k*4) as u64 };
                    a_i = objc::msg_send![a_i, initWithBuffer: abuf offset: off_a descriptor: d_a];
                    let mut at_m: *mut Object = objc::msg_send![mtx_cls, alloc];
                    at_m = objc::msg_send![at_m, initWithBuffer: at_buf.as_ptr() offset: 0u64 descriptor: d_at];
                    let () = objc::msg_send![tr, encodeToCommandBuffer: cb.as_ptr() sourceMatrix: a_i resultMatrix: at_m];
                    let mut g_i: *mut Object = objc::msg_send![mtx_cls, alloc];
                    g_i = objc::msg_send![g_i, initWithBuffer: gobuf offset: i*(m*n*4) as u64 descriptor: d_go];
                    let () = objc::msg_send![mm_db, encodeToCommandBuffer: cb.as_ptr() leftMatrix: at_m rightMatrix: g_i resultMatrix: c_db];
                }
            }

            cb.commit(); cb.wait_until_completed();

            Ok((BackendArrayF32::Mps{ rows:m, cols:k, buffer: da_buf },
                BackendArrayF32::Mps{ rows:k, cols:n, buffer: db_buf }))
        }
    }
}

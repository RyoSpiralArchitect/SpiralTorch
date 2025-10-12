#[cfg(feature="cuda")]
use cudarc::nvrtc::compile_ptx_with_opts;
#[cfg(feature="cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

use crate::error::{Result, device as dev_err};

#[cfg(feature="cuda")]
fn nvrtc_arch() -> String {
    std::env::var("ST_NVRTC_ARCH").unwrap_or_else(|_| "--gpu-architecture=compute_80".to_string())
}

#[cfg(feature="cuda")]
const CUDA_TOPK_SRC_TPL: &str = r#"
extern "C" __global__
void topk_pass1_rowmajor_2d_multi_half(const half* __restrict__ Xh, int rows, int cols,
                                       int ctas_per_row, float* __restrict__ CANDV, int* __restrict__ CANDI) {
    const int rid = blockIdx.x / ctas_per_row;
    const int b   = blockIdx.x % ctas_per_row;
    if (rid >= rows) return;
    const int tid = threadIdx.x; // 128
    const int T = blockDim.x;
    const int KLANE = %(KLANE)s; // 32/64/128

    extern __shared__ unsigned char smem[];
    float* svals = (float*)smem;
    int*   sidxs = (int*)(svals + 128*%(KLANE)s + 128); // padding

    float lv[KLANE];
    int   li[KLANE];
    #pragma unroll
    for (int i=0;i<KLANE;++i){ lv[i] = -INFINITY; li[i] = -1; }

    const int base = rid * cols;
    int seg_cols = (cols + ctas_per_row - 1) / ctas_per_row;
    int c0 = b * seg_cols;
    int c1 = min(cols, c0 + seg_cols);

    int c = c0 + tid*4;
    half2 cur0 = __float2half2_rn(-INFINITY), cur1 = __float2half2_rn(-INFINITY);
    if (c+3 < c1) {
        cur0 = *((const half2*)&Xh[2*(base + c)]);
        cur1 = *((const half2*)&Xh[2*(base + c + 2)]);
    }
    c += T*4;
    while (true) {
        half2 nxt0 = __float2half2_rn(-INFINITY), nxt1 = __float2half2_rn(-INFINITY);
        if (c+3 < c1) {
            nxt0 = *((const half2*)&Xh[2*(base + c)]);
            nxt1 = *((const half2*)&Xh[2*(base + c + 2)]);
        }
        float vv[4] = {
            __half2float(__low2half(cur0)),
            __half2float(__high2half(cur0)),
            __half2float(__low2half(cur1)),
            __half2float(__high2half(cur1))
        };
        int   cc[4] = {c-T*4, c-T*4+1, c-T*4+2, c-T*4+3};
        #pragma unroll
        for (int j=0;j<4;++j){
            float v = vv[j];
            int idx = cc[j];
            if (idx >= c0 && idx < c1) {
                int minp = 0; float minv = lv[0];
                #pragma unroll
                for (int p=1;p<KLANE;++p){ if (lv[p] < minv){ minv = lv[p]; minp = p; } }
                if (v > minv){ lv[minp]=v; li[minp]=idx; }
            }
        }
        if (c+3 >= c1) break;
        cur0 = nxt0; cur1 = nxt1; c += T*4;
    }
    for (int cc2=c0+tid; cc2<c1; cc2+=T){
        float v = __half2float(Xh[base+cc2]);
        int minp = 0; float minv = lv[0];
        #pragma unroll
        for (int p=1;p<KLANE;++p){ if (lv[p] < minv){ minv = lv[p]; minp = p; } }
        if (v > minv){ lv[minp]=v; li[minp]=cc2; }
    }
    const int offset = tid*KLANE + tid;
    #pragma unroll
    for (int i=0;i<KLANE;++i){ svals[offset+i] = lv[i]; sidxs[offset+i] = li[i]; }
    __syncthreads();

    int base_cand = blockIdx.x * (128*KLANE);
    #pragma unroll
    for (int i=0;i<KLANE;++i){
        CANDV[base_cand + tid*KLANE + i] = svals[offset+i];
        CANDI[base_cand + tid*KLANE + i] = sidxs[offset+i];
    }
}

extern "C" __global__
void topk_pass1_rowmajor_2d_multi(const float* __restrict__ X, int rows, int cols,
                                  int ctas_per_row, float* __restrict__ CANDV, int* __restrict__ CANDI) {
    const int rid = blockIdx.x / ctas_per_row; const int b = blockIdx.x % ctas_per_row;
    if (rid >= rows) return;
    const int tid = threadIdx.x; const int T = blockDim.x; const int KLANE = %(KLANE)s;

    extern __shared__ unsigned char smem[];
    float* svals = (float*)smem; int* sidxs = (int*)(svals + 128*%(KLANE)s + 128);

    float lv[KLANE]; int li[KLANE];
    #pragma unroll for (int i=0;i<KLANE;++i){ lv[i] = -INFINITY; li[i] = -1; }

    const int base = rid * cols;
    int seg_cols = (cols + ctas_per_row - 1) / ctas_per_row;
    int c0 = b * seg_cols; int c1 = min(cols, c0 + seg_cols);

    int c = c0 + tid*4;
    float4 cur = (c+3<c1) ? *((const float4*)&X[base+c]) : make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);
    c += T*4;
    while (true) {
        float4 nxt = (c+3<c1) ? *((const float4*)&X[base+c]) : make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);
        float vv[4] = {cur.x,cur.y,cur.z,cur.w};
        int   cc[4] = {c-T*4, c-T*4+1, c-T*4+2, c-T*4+3};
        #pragma unroll for (int j=0;j<4;++j){
            float v = vv[j]; int idx = cc[j];
            if (idx>=c0 && idx<c1){
                int minp=0; float minv=lv[0];
                #pragma unroll for (int p=1;p<KLANE;++p){ if (lv[p]<minv){ minv=lv[p]; minp=p; } }
                if (v>minv){ lv[minp]=v; li[minp]=idx; }
            }
        }
        if (c+3>=c1) break; cur=nxt; c+=T*4;
    }
    for (int cc2=c0+tid; cc2<c1; cc2+=T){
        float v = X[base+cc2];
        int minp=0; float minv=lv[0];
        #pragma unroll for (int p=1;p<KLANE;++p){ if (lv[p]<minv){ minv=lv[p]; minp=p; } }
        if (v>minv){ lv[minp]=v; li[minp]=cc2; }
    }

    const int offset = tid*KLANE + tid;
    #pragma unroll for (int i=0;i<KLANE;++i){ svals[offset+i]=lv[i]; sidxs[offset+i]=li[i]; }
    __syncthreads();
    int base_cand = blockIdx.x * (128*KLANE);
    #pragma unroll for (int i=0;i<KLANE;++i){
        CANDV[base_cand + tid*KLANE + i] = svals[offset+i];
        CANDI[base_cand + tid*KLANE + i] = sidxs[offset+i];
    }
}

extern "C" __global__
void topk_pass2_merge_rowmajor_2d_multi(const float* __restrict__ CANDV, const int* __restrict__ CANDI,
                                        int rows, int k, int ctas_per_row, int cand_per_block,
                                        float* __restrict__ OUTV, int* __restrict__ OUTI) {
    const int rid = blockIdx.x;
    if (rid >= rows) return;
    const int tid = threadIdx.x; // 128
    const int total_cand = ctas_per_row * cand_per_block;
    extern __shared__ unsigned char smem[];
    float* svals = (float*)smem;
    int*   sidxs = (int*)(svals + total_cand + 128); // padding

    for (int b=0; b<ctas_per_row; ++b) {
        int base_cand = (rid*ctas_per_row + b) * cand_per_block;
        for (int t=tid; t<cand_per_block; t+=128) {
            svals[b*cand_per_block + t] = CANDV[base_cand + t];
            sidxs[b*cand_per_block + t] = CANDI[base_cand + t];
        }
    }
    __syncthreads();
    for (int size=2; size<=total_cand; size<<=1){
        for (int stride=size>>1; stride>0; stride>>=1){
            for (int i=tid; i<total_cand; i+=128){
                int j = i ^ stride;
                if (j > i){
                    bool up = ((i & size) == 0);
                    float vi = svals[i]; float vj = svals[j];
                    if ((up && vi < vj) || (!up && vi > vj)){
                        float ti = vi; int ii = sidxs[i];
                        float tj = vj; int ij = sidxs[j];
                        svals[i] = tj; sidxs[i] = ij;
                        svals[j] = ti; sidxs[j] = ii;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (tid < k){
        OUTV[rid*k + tid] = svals[tid];
        OUTI[rid*k + tid] = sidxs[tid];
    }
}
"#;

#[cfg(feature="cuda")]
fn compile_variants(klane: i32, use_half: bool) -> Result<String> {
    let mut src = CUDA_TOPK_SRC_TPL.replace("%(KLANE)s", &klane.to_string());
    let mut opts = vec!["--std=c++14".to_string(), nvrtc_arch()];
    if use_half { opts.push("-DUSE_HALF=1".to_string()); src = format!("#include <cuda_fp16.h>\n{}", src); }
    let ptx = compile_ptx_with_opts(&src, &opts.iter().map(|s| s.as_str()).collect::<Vec<_>>())
        .map_err(|e| dev_err(&format!("nvrtc: {e}")))?;
    Ok(ptx)
}

#[cfg(feature="cuda")]
pub fn topk_lastdim_cuda_2d(x:&[f32], rows:usize, cols:usize, k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    if k==0 || k>cols { return Err(dev_err("topk: invalid k")); }
    let dev = CudaDevice::new(0).map_err(|e| dev_err(&format!("cuda device: {e}")))?;

    // Reuse WGPU heuristics (or SpiralK) if present
    let (u2, _wg, kl, ch) = crate::backend::wgpu_heuristics::choose(rows as u32, cols as u32, k as u32, false)
        .unwrap_or((cols as u32>32768 || k as u32>128, 256, if k>32 {32} else if k>16 {16} else {8}, if cols>16384 {8192} else {0}));
    let klane = if kl >= 64 { 64 } else if kl >= 32 { 32 } else { 16 }; // map to supported
    let ctas_per_row = if ch>0 { ((cols as u32 + ch - 1) / ch).min(8) } else { 1 };

    let use_half = std::env::var("ST_CUDA_HALF").ok().as_deref()==Some("1");
    let ptx = compile_variants(klane as i32, use_half)?;
    dev.load_ptx(ptx, "st_topk", &[
        "topk_pass1_rowmajor_2d_multi_half",
        "topk_pass1_rowmajor_2d_multi",
        "topk_pass2_merge_rowmajor_2d_multi"
    ]).map_err(|e| dev_err(&format!("load ptx: {e}")))?;
    let pass1h = dev.get_func("st_topk", "topk_pass1_rowmajor_2d_multi_half").ok();
    let pass1f = dev.get_func("st_topk", "topk_pass1_rowmajor_2d_multi").map_err(|e| dev_err(&format!("get func: {e}")))?;
    let pass2  = dev.get_func("st_topk", "topk_pass2_merge_rowmajor_2d_multi").map_err(|e| dev_err(&format!("get func: {e}")))?;

    // Buffers
    let mut outv_d = dev.alloc_zeros::<f32>(rows*k).map_err(|e| dev_err(&format!("alloc outv: {e}")))?;
    let mut outi_d = dev.alloc_zeros::<i32>(rows*k).map_err(|e| dev_err(&format!("alloc outi: {e}")))?;
    let cand_per_block = 128 * klane as usize;
    let total_cand = (ctas_per_row as usize) * cand_per_block;
    let mut candv_d = dev.alloc_zeros::<f32>(rows*total_cand).map_err(|e| dev_err(&format!("alloc candv: {e}")))?;
    let mut candi_d = dev.alloc_zeros::<i32>(rows*total_cand).map_err(|e| dev_err(&format!("alloc candi: {e}")))?;

    // Upload X: either f32 or half (repacked)
    let cfg1 = LaunchConfig { grid_dim: ((rows as u32)*(ctas_per_row as u32), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: (cand_per_block+128)*8u32 };
    let shared_bytes = (total_cand + 128) * 4 * 2;
    let cfg2 = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: shared_bytes as u32 };

    if use_half {
        // pack to half
        let mut xh = vec![0u16; rows*cols];
        for i in 0..(rows*cols) { xh[i] = half::f16::from_f32(x[i]).to_bits(); }
        let xh_d = dev.htod_copy_from(xh.as_slice()).map_err(|e| dev_err(&format!("htod half: {e}")))?;
        if let Some(kh) = pass1h {
            unsafe { kh.launch(cfg1, (&xh_d, rows as i32, cols as i32, ctas_per_row as i32, &mut candv_d, &mut candi_d)) }
                .map_err(|e| dev_err(&format!("launch pass1 half: {e}")))?;
        } else {
            return Err(dev_err("half kernel not available"));
        }
    } else {
        let x_d = dev.htod_copy(x).map_err(|e| dev_err(&format!("htod: {e}")))?;
        unsafe { pass1f.launch(cfg1, (&x_d, rows as i32, cols as i32, ctas_per_row as i32, &mut candv_d, &mut candi_d)) }
            .map_err(|e| dev_err(&format!("launch pass1: {e}")))?;
    }
    unsafe { pass2.launch(cfg2, (&candv_d, &candi_d, rows as i32, k as i32, ctas_per_row as i32, cand_per_block as i32, &mut outv_d, &mut outi_d)) }
        .map_err(|e| dev_err(&format!("launch pass2: {e}")))?;

    let outv = dev.dtoh_sync_copy(&outv_d).map_err(|e| dev_err(&format!("dtoh: {e}")))?;
    let outi = dev.dtoh_sync_copy(&outi_d).map_err(|e| dev_err(&format!("dtoh: {e}")))?;
    Ok((outv, outi))
}

#[cfg(not(feature="cuda"))]
pub fn topk_lastdim_cuda_2d(_x:&[f32], _rows:usize, _cols:usize, _k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    Err(dev_err("cuda feature not enabled"))
}

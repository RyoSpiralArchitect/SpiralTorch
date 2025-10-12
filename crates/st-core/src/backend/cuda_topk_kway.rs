#[cfg(feature="cuda")]
use cudarc::nvrtc::compile_ptx_with_opts;
#[cfg(feature="cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

use crate::error::{Result, device as dev_err};

#[cfg(feature="cuda")]
const CUDA_TOPK_SRC_TPL: &str = r#"
extern "C" __global__
void topk_pass1_rowmajor_2d_multi(const float* __restrict__ X, int rows, int cols,
                                  int ctas_per_row,
                                  float* __restrict__ CANDV, int* __restrict__ CANDI) {
    const int rid = blockIdx.x / ctas_per_row;   // row id
    const int b   = blockIdx.x % ctas_per_row;   // block-of-row
    if (rid >= rows) return;
    const int tid = threadIdx.x;                 // 128
    const int T = blockDim.x;
    const int KLANE = %(KLANE)s;                 // 32/64/128

    __shared__ float svals[128*%(KLANE)s + 128]; // +padding to reduce bank conflicts
    __shared__ int   sidxs[128*%(KLANE)s + 128];

    float lv[KLANE];
    int   li[KLANE];
    #pragma unroll
    for (int i=0;i<KLANE;++i){ lv[i] = -INFINITY; li[i] = -1; }

    const int base = rid * cols;
    // assign segment per-CTA
    int seg_cols = (cols + ctas_per_row - 1) / ctas_per_row;
    int c0 = b * seg_cols;
    int c1 = min(cols, c0 + seg_cols);

    // Double-buffered ILP: float4 prefetch
    int c = c0 + tid*4;
    float4 cur, nxt;
    if (c+3 < c1) cur = *((const float4*)&X[base + c]); else cur = make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);
    c += T*4;
    while (true) {
        if (c+3 < c1) nxt = *((const float4*)&X[base + c]); else nxt = make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);
        float vv[4] = {cur.x, cur.y, cur.z, cur.w};
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
        cur = nxt; c += T*4;
    }
    // Tail
    for (int cc2 = c0 + tid; cc2 < c1; cc2 += T) {
        float v = X[base + cc2];
        int minp = 0; float minv = lv[0];
        #pragma unroll
        for (int p=1;p<KLANE;++p){ if (lv[p] < minv){ minv = lv[p]; minp = p; } }
        if (v > minv){ lv[minp]=v; li[minp]=cc2; }
    }

    const int offset = tid*KLANE + tid;
    #pragma unroll
    for (int i=0;i<KLANE;++i){ svals[offset+i] = lv[i]; sidxs[offset+i] = li[i]; }
    __syncthreads();

    // write to global candidates: [rows][ctas_per_row][128*KLANE]
    int base_cand = (rid*ctas_per_row + b) * (128*KLANE);
    #pragma unroll
    for (int i=0;i<KLANE;++i){
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
    const int tid = threadIdx.x;                 // 128
    const int KLANE = %(KLANE)s;                 // 32/64/128

    const int total_cand = ctas_per_row * cand_per_block;
    extern __shared__ unsigned char smem[];
    float* svals = (float*)smem;
    int*   sidxs = (int*)(svals + total_cand + 128); // +padding

    // Load all candidates for this row (multi-CTA)
    for (int b=0; b<ctas_per_row; ++b) {
        int base_cand = (rid*ctas_per_row + b) * cand_per_block;
        for (int t=tid; t<cand_per_block; t+=128) {
            svals[b*cand_per_block + t] = CANDV[base_cand + t];
            sidxs[b*cand_per_block + t] = CANDI[base_cand + t];
        }
    }
    __syncthreads();

    // Bitonic on total_cand
    for (int size=2; size<=total_cand; size<<=1){
        for (int stride=size>>1; stride>0; stride>>=1){
            for (int i=tid; i<total_cand; i+=128){
                int j = i ^ stride;
                if (j > i){
                    bool up = ((i & size) == 0);
                    float vi = svals[i];
                    float vj = svals[j];
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
fn compile_variants(klane: i32) -> Result<String> {
    let src = CUDA_TOPK_SRC_TPL.replace("%(KLANE)s", &klane.to_string());
    let ptx = compile_ptx_with_opts(&src, &["--std=c++14"]).map_err(|e| dev_err(&format!("nvrtc: {e}")))?;
    Ok(ptx)
}

#[cfg(feature="cuda")]
pub fn topk_lastdim_cuda_2d(x:&[f32], rows:usize, cols:usize, k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    if k==0 || k>cols { return Err(dev_err("topk: invalid k")); }
    let dev = CudaDevice::new(0).map_err(|e| dev_err(&format!("cuda device: {e}")))?;

    // Choose KLANE and ctas_per_row
    let klane = if k > 8192 { 128 } else if k > 256 { 64 } else { 32 };
    let ctas_per_row = if cols > 131072 { 8 } else if cols > 65536 { 4 } else if cols > 32768 { 2 } else { 1 };

    let ptx = compile_variants(klane)?;
    dev.load_ptx(ptx, "st_topk", &["topk_pass1_rowmajor_2d_multi", "topk_pass2_merge_rowmajor_2d_multi"])
        .map_err(|e| dev_err(&format!("load ptx: {e}")))?;
    let pass1 = dev.get_func("st_topk", "topk_pass1_rowmajor_2d_multi")
        .map_err(|e| dev_err(&format!("get func: {e}")))?;
    let pass2 = dev.get_func("st_topk", "topk_pass2_merge_rowmajor_2d_multi")
        .map_err(|e| dev_err(&format!("get func: {e}")))?;

    // Device buffers
    let x_d = dev.htod_copy(x).map_err(|e| dev_err(&format!("htod: {e}")))?;
    let mut outv_d = dev.alloc_zeros::<f32>(rows*k).map_err(|e| dev_err(&format!("alloc outv: {e}")))?;
    let mut outi_d = dev.alloc_zeros::<i32>(rows*k).map_err(|e| dev_err(&format!("alloc outi: {e}")))?;
    let cand_per_block = 128 * klane as usize;
    let total_cand = ctas_per_row * cand_per_block;
    let mut candv_d = dev.alloc_zeros::<f32>(rows*total_cand).map_err(|e| dev_err(&format!("alloc candv: {e}")))?;
    let mut candi_d = dev.alloc_zeros::<i32>(rows*total_cand).map_err(|e| dev_err(&format!("alloc candi: {e}")))?;

    // Launch
    let cfg1 = LaunchConfig { grid_dim: ((rows*ctas_per_row) as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
    let shared_bytes = (total_cand + 128) * 4 * 2; // float + int + pad
    let cfg2 = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: shared_bytes as u32 };

    unsafe {
        pass1.launch(cfg1, (&x_d, rows as i32, cols as i32, ctas_per_row as i32, &mut candv_d, &mut candi_d))
            .map_err(|e| dev_err(&format!("launch pass1: {e}")))?;
        pass2.launch(cfg2, (&candv_d, &candi_d, rows as i32, k as i32, ctas_per_row as i32, cand_per_block as i32, &mut outv_d, &mut outi_d))
            .map_err(|e| dev_err(&format!("launch pass2: {e}")))?;
    }

    let outv = dev.dtoh_sync_copy(&outv_d).map_err(|e| dev_err(&format!("dtoh: {e}")))?;
    let outi = dev.dtoh_sync_copy(&outi_d).map_err(|e| dev_err(&format!("dtoh: {e}")))?;
    Ok((outv, outi))
}

#[cfg(not(feature="cuda"))]
pub fn topk_lastdim_cuda_2d(_x:&[f32], _rows:usize, _cols:usize, _k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    Err(dev_err("cuda feature not enabled"))
}

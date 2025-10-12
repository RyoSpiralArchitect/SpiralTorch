#[cfg(feature="cuda")]
use cudarc::nvrtc::compile_ptx_with_opts;
#[cfg(feature="cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

use crate::error::{Result, device as dev_err};

#[cfg(feature="cuda")]
const CUDA_TOPK_SRC_TPL: &str = r#"
extern "C" __global__
void topk_pass1_rowmajor_2d(const float* __restrict__ X, int rows, int cols,
                            float* __restrict__ CANDV, int* __restrict__ CANDI) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;       // 128
    const int T = blockDim.x;
    const int KLANE = %(KLANE)s;       // 32/64/128
    __shared__ float svals[128*%(KLANE)s + 128]; // +padding
    __shared__ int   sidxs[128*%(KLANE)s + 128];

    float lv[KLANE];
    int   li[KLANE];
    #pragma unroll
    for (int i=0;i<KLANE;++i){ lv[i] = -INFINITY; li[i] = -1; }

    const int base = row * cols;

    // Double-buffered ILP: prefetch next float4 while consuming current
    int c = tid * 4;
    float4 cur, nxt;
    if (c+3 < cols) cur = *((const float4*)&X[base + c]); else cur = make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);
    c += T*4;
    while (true) {
        if (c+3 < cols) nxt = *((const float4*)&X[base + c]); else nxt = make_float4(-INFINITY,-INFINITY,-INFINITY,-INFINITY);
        float vv[4] = {cur.x, cur.y, cur.z, cur.w};
        int   cc[4] = {c-T*4, c-T*4+1, c-T*4+2, c-T*4+3};
        #pragma unroll
        for (int j=0;j<4;++j){
            float v = vv[j];
            int idx = cc[j];
            if (idx >= 0 && idx < cols) {
                int minp = 0; float minv = lv[0];
                #pragma unroll
                for (int p=1;p<KLANE;++p){ if (lv[p] < minv){ minv = lv[p]; minp = p; } }
                if (v > minv){ lv[minp]=v; li[minp]=idx; }
            }
        }
        if (c+3 >= cols) break;
        cur = nxt; c += T*4;
    }
    // Tail (scalar)
    int tail = (tid*4) + ((cols - (tid*4)) % (T*4));
    for (int cc2 = tail; cc2 < cols; cc2 += T) {
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

    // Write candidates to global
    const int base_cand = row * (128*KLANE);
    #pragma unroll
    for (int i=0;i<KLANE;++i){
        CANDV[base_cand + tid*KLANE + i] = svals[offset+i];
        CANDI[base_cand + tid*KLANE + i] = sidxs[offset+i];
    }
}

extern "C" __global__
void topk_pass2_merge_rowmajor_2d(const float* __restrict__ CANDV, const int* __restrict__ CANDI,
                                  int rows, int k, int cand_cols,
                                  float* __restrict__ OUTV, int* __restrict__ OUTI) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;       // 128
    const int KLANE = %(KLANE)s;       // 32/64/128

    __shared__ float svals[128*%(KLANE)s + 128];
    __shared__ int   sidxs[128*%(KLANE)s + 128];

    // Load candidates from global
    const int base = row * cand_cols;
    const int offset = tid*KLANE + tid;
    #pragma unroll
    for (int i=0;i<KLANE;++i){
        svals[offset+i] = CANDV[base + tid*KLANE + i];
        sidxs[offset+i] = CANDI[base + tid*KLANE + i];
    }
    __syncthreads();

    // Bitonic on 128*KLANE entries (descending)
    for (int size=2; size<=128*KLANE; size<<=1){
        for (int stride=size>>1; stride>0; stride>>=1){
            int i = tid;
            while (i < 128*KLANE){
                int j = i ^ stride;
                if (j > i){
                    bool up = ((i & size) == 0);
                    float vi = svals[i + (i/ KLANE)];
                    float vj = svals[j + (j/ KLANE)];
                    if ((up && vi < vj) || (!up && vi > vj)){
                        float ti = vi; int ii = sidxs[i + (i/ KLANE)];
                        float tj = vj; int ij = sidxs[j + (j/ KLANE)];
                        svals[i + (i/ KLANE)] = tj; svals[j + (j/ KLANE)] = ti;
                        sidxs[i + (i/ KLANE)] = ij; sidxs[j + (j/ KLANE)] = ii;
                    }
                }
                i += 128;
            }
            __syncthreads();
        }
    }
    if (tid < k){
        OUTV[row*k + tid] = svals[tid + (tid/ KLANE)];
        OUTI[row*k + tid] = sidxs[tid + (tid/ KLANE)];
    }
}
"#;

#[cfg(feature="cuda")]
fn compile_variants(klane: i32) -> Result<(String, String)> {
    let src = CUDA_TOPK_SRC_TPL.replace("%(KLANE)s", &klane.to_string());
    let ptx = compile_ptx_with_opts(&src, &["--std=c++14"]).map_err(|e| dev_err(&format!("nvrtc: {e}")))?;
    // For simplicity, both kernels are in the same PTX
    Ok((ptx.clone(), ptx))
}

#[cfg(feature="cuda")]
pub fn topk_lastdim_cuda_2d(x:&[f32], rows:usize, cols:usize, k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    if k==0 || k>cols { return Err(dev_err("topk: invalid k")); }
    let dev = CudaDevice::new(0).map_err(|e| dev_err(&format!("cuda device: {e}")))?;

    // Choose KLANE variant
    let klane = if k > 8192 { 128 } else if k > 256 { 64 } else { 32 };
    let (ptx1, _ptx2) = compile_variants(klane)?;
    dev.load_ptx(ptx1, "st_topk", &["topk_pass1_rowmajor_2d", "topk_pass2_merge_rowmajor_2d"]).map_err(|e| dev_err(&format!("load ptx: {e}")))?;
    let pass1 = dev.get_func("st_topk", "topk_pass1_rowmajor_2d").map_err(|e| dev_err(&format!("get func: {e}")))?;
    let pass2 = dev.get_func("st_topk", "topk_pass2_merge_rowmajor_2d").map_err(|e| dev_err(&format!("get func: {e}")))?;

    // Device buffers
    let x_d = dev.htod_copy(x).map_err(|e| dev_err(&format!("htod: {e}")))?;
    let mut outv_d = dev.alloc_zeros::<f32>(rows*k).map_err(|e| dev_err(&format!("alloc outv: {e}")))?;
    let mut outi_d = dev.alloc_zeros::<i32>(rows*k).map_err(|e| dev_err(&format!("alloc outi: {e}")))?;
    let cand_cols = 128 * klane as usize;
    let mut candv_d = dev.alloc_zeros::<f32>(rows*cand_cols).map_err(|e| dev_err(&format!("alloc candv: {e}")))?;
    let mut candi_d = dev.alloc_zeros::<i32>(rows*cand_cols).map_err(|e| dev_err(&format!("alloc candi: {e}")))?;

    let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };

    unsafe {
        // Pass1
        pass1.launch(cfg, (&x_d, rows as i32, cols as i32, &mut candv_d, &mut candi_d))
            .map_err(|e| dev_err(&format!("launch pass1: {e}")))?;
        // Pass2
        pass2.launch(cfg, (&candv_d, &candi_d, rows as i32, k as i32, cand_cols as i32, &mut outv_d, &mut outi_d))
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

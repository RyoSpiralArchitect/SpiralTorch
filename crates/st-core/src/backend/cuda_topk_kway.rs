#[cfg(feature="cuda")]
use cudarc::nvrtc::compile_ptx;
#[cfg(feature="cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

use crate::error::{Result, device as dev_err};

#[cfg(feature="cuda")]
const CUDA_TOPK_SRC: &str = r#"
extern "C" __global__
void topk_kway_rowmajor_2d(const float* __restrict__ X, int rows, int cols, int k,
                           float* __restrict__ OUTV, int* __restrict__ OUTI) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int T = blockDim.x; // 128
    const int KLANE = 32;     // per-thread candidates
    __shared__ float svals[128*32 + 128]; // padding to avoid bank conflicts
    __shared__ int   sidxs[128*32 + 128];

    // local candidates
    float lv[KLANE];
    int   li[KLANE];
    #pragma unroll
    for (int i=0;i<KLANE;++i){ lv[i] = -INFINITY; li[i] = -1; }

    const int base = row * cols;
    // vectorized scans using float4 when aligned
    int c = tid * 4;
    for (; c+3 < cols; c += T*4) {
        float4 v4 = *((const float4*)&X[base + c]);
        float vv[4] = {v4.x, v4.y, v4.z, v4.w};
        int   cc[4] = {c, c+1, c+2, c+3};
        #pragma unroll
        for (int j=0;j<4;++j){
            float v = vv[j];
            // min-replace into local top-KLANE
            int minp = 0; float minv = lv[0];
            #pragma unroll
            for (int p=1;p<KLANE;++p){ if (lv[p] < minv){ minv = lv[p]; minp = p; } }
            if (v > minv){ lv[minp]=v; li[minp]=cc[j]; }
        }
    }
    // tail
    for (; c < cols; c += T) {
        float v = X[base + c];
        int minp = 0; float minv = lv[0];
        #pragma unroll
        for (int p=1;p<KLANE;++p){ if (lv[p] < minv){ minv = lv[p]; minp = p; } }
        if (v > minv){ lv[minp]=v; li[minp]=c; }
    }

    // write into shared
    const int offset = tid*KLANE + tid; // +tid padding
    #pragma unroll
    for (int i=0;i<KLANE;++i){ svals[offset+i] = lv[i]; sidxs[offset+i] = li[i]; }
    __syncthreads();

    // bitonic sort over candidates (descending)
    const int total = 128*KLANE + 128; // with padding
    // sort only up to 128*KLANE region (ignoring padding at compare stage)
    for (int size=2; size<=128*KLANE; size<<=1){
        for (int stride=size>>1; stride>0; stride>>=1){
            int i = tid;
            while (i < 128*KLANE){
                int j = i ^ stride;
                if (j > i){
                    bool up = ((i & size) == 0);
                    float vi = svals[i + (i/ KLANE)]; // compensate padding index
                    float vj = svals[j + (j/ KLANE)];
                    if ((up && vi < vj) || (!up && vi > vj)){
                        float ti = vi; int ii = sidxs[i + (i/ KLANE)];
                        float tj = vj; int ij = sidxs[j + (j/ KLANE)];
                        svals[i + (i/ KLANE)] = tj; svals[j + (j/ KLANE)] = ti;
                        sidxs[i + (i/ KLANE)] = ij; sidxs[j + (j/ KLANE)] = ii;
                    }
                }
                i += T;
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
pub fn topk_lastdim_cuda_2d(x:&[f32], rows:usize, cols:usize, k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    if k==0 || k>cols { return Err(dev_err("topk: invalid k")); }
    let dev = CudaDevice::new(0).map_err(|e| dev_err(&format!("cuda device: {e}")))?;
    let ptx = compile_ptx(CUDA_TOPK_SRC).map_err(|e| dev_err(&format!("nvrtc: {e}")))?;
    dev.load_ptx(ptx, "st_topk", &["topk_kway_rowmajor_2d"]).map_err(|e| dev_err(&format!("load ptx: {e}")))?;
    let func = dev.get_func("st_topk", "topk_kway_rowmajor_2d").map_err(|e| dev_err(&format!("get func: {e}")))?;

    let n = rows*cols;
    let x_d = dev.htod_copy(x).map_err(|e| dev_err(&format!("htod: {e}")))?;
    let mut outv_d = dev.alloc_zeros::<f32>(rows*k).map_err(|e| dev_err(&format!("alloc: {e}")))?;
    let mut outi_d = dev.alloc_zeros::<i32>(rows*k).map_err(|e| dev_err(&format!("alloc: {e}")))?;

    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(cfg, (&x_d, rows as i32, cols as i32, k as i32, &mut outv_d, &mut outi_d))
            .map_err(|e| dev_err(&format!("launch: {e}")))?;
    }

    let outv = dev.dtoh_sync_copy(&outv_d).map_err(|e| dev_err(&format!("dtoh: {e}")))?;
    let outi = dev.dtoh_sync_copy(&outi_d).map_err(|e| dev_err(&format!("dtoh: {e}")))?;
    Ok((outv, outi))
}

#[cfg(not(feature="cuda"))]
pub fn topk_lastdim_cuda_2d(_x:&[f32], _rows:usize, _cols:usize, _k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    Err(dev_err("cuda feature not enabled"))
}

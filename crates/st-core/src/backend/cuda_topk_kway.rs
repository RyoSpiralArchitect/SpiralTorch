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
const CUDA_TOPK_SRC: &str = r#"
extern "C" __global__
void topk_pass1_rowmajor_2d_multi(const float* __restrict__ X, int rows, int cols,
                                  int ctas_per_row, float* __restrict__ CANDV, int* __restrict__ CANDI, int KLANE) {
    const int rid = blockIdx.x / ctas_per_row; const int b = blockIdx.x % ctas_per_row;
    if (rid >= rows) return;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* svals = smem;
    int*   sidxs = (int*)(svals + 128*128 + 128);

    float lv[128]; int li[128];
    for (int i=0;i<KLANE;++i){ lv[i] = -INFINITY; li[i] = -1; }

    const int base = rid * cols;
    int seg_cols = (cols + ctas_per_row - 1) / ctas_per_row;
    int c0 = b * seg_cols; int c1 = min(cols, c0 + seg_cols);

    int c = c0 + tid;
    while (c < c1) {
        float v = X[base + c];
        int minp=0; float minv=lv[0];
        for (int p=1;p<KLANE;++p){ if (lv[p]<minv){ minv=lv[p]; minp=p; } }
        if (v>minv){ lv[minp]=v; li[minp]=c; }
        c += blockDim.x;
    }
    const int offset = tid*KLANE + tid;
    for (int i=0;i<KLANE;++i){ svals[offset+i]=lv[i]; sidxs[offset+i]=li[i]; }
    __syncthreads();

    int base_cand = blockIdx.x * (128*KLANE);
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
    const int tid = threadIdx.x;
    const int total_cand = ctas_per_row * cand_per_block;
    extern __shared__ unsigned char smem[];
    float* svals = (float*)smem;
    int*   sidxs = (int*)(svals + total_cand + 128);

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
pub fn topk_lastdim_cuda_2d(x:&[f32], rows:usize, cols:usize, k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    if k==0 || k>cols { return Err(dev_err("topk: invalid k")); }
    let dev = CudaDevice::new(0).map_err(|e| dev_err(&format!("cuda device: {e}")))?;

    // Unified heuristics (SpiralK → table → fallback)
    let (use_2ce, _wg, kl, ch) = if let Some(ch) = crate::backend::wgpu_heuristics::choose(rows, cols, k, false) {
        (ch.use_2ce, ch.wg, ch.kl, ch.ch)
    } else { (cols>32768 || k>128, 256, if k>=32 {32} else if k>=16 {16} else {8}, if cols>16384 {8192} else {0}) };
    let klane = kl.max(8).min(128) as i32;
    let ctas_per_row = if ch>0 { ((cols as u32 + ch - 1) / ch).min(8) } else { 1 } as i32;

    let opts = vec!["--std=c++14".to_string(), nvrtc_arch()];
    let ptx = compile_ptx_with_opts(CUDA_TOPK_SRC, &opts.iter().map(|s| s.as_str()).collect::<Vec<_>>())
        .map_err(|e| dev_err(&format!("nvrtc: {e}")))?;
    dev.load_ptx(ptx, "st_topk", &[
        "topk_pass1_rowmajor_2d_multi",
        "topk_pass2_merge_rowmajor_2d_multi"
    ]).map_err(|e| dev_err(&format!("load ptx: {e}")))?;
    let pass1 = dev.get_func("st_topk", "topk_pass1_rowmajor_2d_multi").map_err(|e| dev_err(&format!("get func: {e}")))?;
    let pass2 = dev.get_func("st_topk", "topk_pass2_merge_rowmajor_2d_multi").map_err(|e| dev_err(&format!("get func: {e}")))?;

    let outv_len = rows*k; let outi_len=outv_len;
    let mut outv_d = dev.alloc_zeros::<f32>(outv_len).map_err(|e| dev_err(&format!("alloc outv: {e}")))?;
    let mut outi_d = dev.alloc_zeros::<i32>(outi_len).map_err(|e| dev_err(&format!("alloc outi: {e}")))?;
    let cand_per_block = 128 * (klane as usize);
    let total_cand = (ctas_per_row as usize) * cand_per_block;
    let mut candv_d = dev.alloc_zeros::<f32>(rows*total_cand).map_err(|e| dev_err(&format!("alloc candv: {e}")))?;
    let mut candi_d = dev.alloc_zeros::<i32>(rows*total_cand).map_err(|e| dev_err(&format!("alloc candi: {e}")))?;

    // upload X
    let x_d = dev.htod_copy(x).map_err(|e| dev_err(&format!("htod: {e}")))?;

    let cfg1 = LaunchConfig { grid_dim: ((rows as u32)*(ctas_per_row as u32), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: (cand_per_block+128)*8u32 };
    let shared_bytes = (total_cand + 128) * 4 * 2;
    let cfg2 = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: shared_bytes as u32 };

    unsafe { pass1.launch(cfg1, (&x_d, rows as i32, cols as i32, ctas_per_row, &mut candv_d, &mut candi_d, klane)) }
        .map_err(|e| dev_err(&format!("launch pass1: {e}")))?;
    unsafe { pass2.launch(cfg2, (&candv_d, &candi_d, rows as i32, k as i32, ctas_per_row, cand_per_block as i32, &mut outv_d, &mut outi_d)) }
        .map_err(|e| dev_err(&format!("launch pass2: {e}")))?;

    let outv = dev.dtoh_sync_copy(&outv_d).map_err(|e| dev_err(&format!("dtoh: {e}")))?;
    let outi = dev.dtoh_sync_copy(&outi_d).map_err(|e| dev_err(&format!("dtoh: {e}")))?;
    Ok((outv, outi))
}

#[cfg(not(feature="cuda"))]
pub fn topk_lastdim_cuda_2d(_x:&[f32], _rows:usize, _cols:usize, _k:usize) -> Result<(Vec<f32>, Vec<i32>)> {
    Err(dev_err("cuda feature not enabled"))
}

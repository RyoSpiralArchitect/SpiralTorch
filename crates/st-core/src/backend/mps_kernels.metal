
#include <metal_stdlib>
using namespace metal;

constant uint MAX_D = 6;

// ---- ReLU backward (strided) minimal (optional use) ----
struct NdInfoA {
    uint ndim;
    uint n;
    uint shape[6];
    int  stride_x[6];
    int  stride_go[6];
};
inline int off_idx_a(uint idx, thread const NdInfoA& info, thread const int *stride) {
    int off = 0;
    for (int d=int(info.ndim)-1; d>=0; --d) {
        uint s = info.shape[d];
        uint i_d = idx % s; idx /= s;
        off += int(i_d) * stride[d];
    }
    return off;
}
kernel void relu_backward_strided(const device float* x [[ buffer(0) ]],
                                  const device float* go [[ buffer(1) ]],
                                  device float*       gx [[ buffer(2) ]],
                                  constant NdInfoA&   info [[ buffer(3) ]],
                                  uint gid [[ thread_position_in_grid ]]) {
    if (gid >= info.n) return;
    int ix  = off_idx_a(gid, info, info.stride_x);
    int igo = off_idx_a(gid, info, info.stride_go);
    float xv = x[ix];
    float gov = go[igo];
    gx[gid] = xv > 0.0f ? gov : 0.0f;
}

// ---- Transpose utilities ----
kernel void transpose_2d(const device float* x [[ buffer(0) ]],
                         device float*       y [[ buffer(1) ]],
                         constant uint&      rows [[ buffer(2) ]],
                         constant uint&      cols [[ buffer(3) ]],
                         uint gid [[ thread_position_in_grid ]]) {
    uint n = rows * cols;
    if (gid >= n) return;
    uint r = gid / cols;
    uint c = gid % cols;
    y[c * rows + r] = x[r * cols + c];
}
kernel void transpose_2d_batched(const device float* x [[ buffer(0) ]],
                                 device float*       y [[ buffer(1) ]],
                                 constant uint&      rows [[ buffer(2) ]],
                                 constant uint&      cols [[ buffer(3) ]],
                                 constant uint&      batches [[ buffer(4) ]],
                                 constant uint&      stride_x [[ buffer(5) ]],
                                 constant uint&      stride_y [[ buffer(6) ]],
                                 uint gid [[ thread_position_in_grid ]]) {
    uint per = rows * cols;
    uint total = per * batches;
    if (gid >= total) return;
    uint b = gid / per;
    uint l = gid % per;
    uint r = l / cols;
    uint c = l % cols;
    uint offx = b * stride_x + r * cols + c;
    uint offy = b * stride_y + c * rows + r;
    y[offy] = x[offx];
}

// ---- Softmax backward helpers ----
kernel void rowwise_dot_gy_wg(const device float* go [[ buffer(0) ]],
                              const device float* y  [[ buffer(1) ]],
                              device float*       dot[[ buffer(2) ]],
                              constant uint&      rows [[ buffer(3) ]],
                              constant uint&      cols [[ buffer(4) ]],
                              uint tid [[ thread_index_in_threadgroup ]],
                              uint3 tg [[ threadgroup_position_in_grid ]]) {
    if (tg.x >= rows) return;
    threadgroup float sdata[256];
    float loc = 0.0f;
    uint r = tg.x;
    for (uint c = tid; c < cols; c += 256) loc += go[r*cols + c] * y[r*cols + c];
    sdata[tid] = loc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) dot[r] = sdata[0];
}
kernel void softmax_bw_from_dot(const device float* go [[ buffer(0) ]],
                                const device float* y  [[ buffer(1) ]],
                                const device float* dot[[ buffer(2) ]],
                                device float*       gx [[ buffer(3) ]],
                                constant uint&      rows [[ buffer(4) ]],
                                constant uint&      cols [[ buffer(5) ]],
                                uint gid [[ thread_position_in_grid ]]) {
    uint n = rows * cols;
    if (gid >= n) return;
    uint r = gid / cols;
    float d = dot[r];
    gx[gid] = (go[gid] - d) * y[gid];
}

// ---- ND reduce (1-pass and 2-pass) ----
struct NdWGInfo {
    uint n_rows;
    uint n_cols;
    uint kdims;
    uint rdims;
    uint kshape[6];
    uint rshape[6];
    int  kstride[6];
    int  rstride[6];
};
inline int base_offset_row(uint ridx, thread const NdWGInfo& inf) {
    int off = 0;
    for (int d=int(inf.kdims)-1; d>=0; --d) {
        uint s = inf.kshape[d];
        uint i = ridx % s; ridx /= s;
        off += int(i) * inf.kstride[d];
    }
    return off;
}
kernel void reduce_nd_wg_sum(const device float* x [[ buffer(0) ]],
                             device float*       out [[ buffer(1) ]],
                             constant NdWGInfo&  inf [[ buffer(2) ]],
                             uint tid [[ thread_index_in_threadgroup ]],
                             uint3 tg  [[ threadgroup_position_in_grid ]]) {
    if (tg.x >= inf.n_rows) return;
    threadgroup float sdata[256];
    float acc = 0.0f;
    int base = base_offset_row(tg.x, inf);
    for (uint c = tid; c < inf.n_cols; c += 256) {
        uint rem = c;
        int off = base;
        for (int d=int(inf.rdims)-1; d>=0; --d) {
            uint sz = inf.rshape[d];
            uint i = rem % sz; rem /= sz;
            off += int(i) * inf.rstride[d];
        }
        acc += x[off];
    }
    sdata[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[tg.x] = sdata[0];
}

struct NdWGPartInfo {
    NdWGInfo base;
    uint groups;
    uint cols_per;
};
kernel void reduce_nd_wg_sum_partials(const device float* x [[ buffer(0) ]],
                                      device float*       partials [[ buffer(1) ]],
                                      constant NdWGPartInfo&  inf [[ buffer(2) ]],
                                      uint tid [[ thread_index_in_threadgroup ]],
                                      uint3 tg  [[ threadgroup_position_in_grid ]]) {
    uint row = tg.x / inf.groups;
    uint g   = tg.x % inf.groups;
    if (row >= inf.base.n_rows) return;
    threadgroup float sdata[256];
    float acc = 0.0f;
    int base = base_offset_row(row, inf.base);
    uint start = g * inf.cols_per;
    uint end   = min((g+1) * inf.cols_per, inf.base.n_cols);
    for (uint c = start + tid; c < end; c += 256) {
        uint rem = c;
        int off = base;
        for (int d=int(inf.base.rdims)-1; d>=0; --d) {
            uint sz = inf.base.rshape[d];
            uint i = rem % sz; rem /= sz;
            off += int(i) * inf.base.rstride[d];
        }
        acc += x[off];
    }
    sdata[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        partials[row * inf.groups + g] = sdata[0];
    }
}
kernel void reduce_nd_wg_sum_finalize(const device float* partials [[ buffer(0) ]],
                                      device float*       out      [[ buffer(1) ]],
                                      constant uint&      n_rows   [[ buffer(2) ]],
                                      constant uint&      groups   [[ buffer(3) ]],
                                      uint tid [[ thread_index_in_threadgroup ]],
                                      uint3 tg  [[ threadgroup_position_in_grid ]]) {
    if (tg.x >= n_rows) return;
    threadgroup float sdata[256];
    float acc = 0.0f;
    for (uint g = tid; g < groups; g += 256) {
        acc += partials[tg.x * groups + g];
    }
    sdata[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid+s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[tg.x] = sdata[0];
}

#include <metal_stdlib>
using namespace metal;
struct RC { uint nd; uint n; };
struct RB { uint c_base; uint x_base; uint y_base; };
kernel void where_nd_strided_u8(
    device const uchar* C           [[ buffer(0) ]],
    device const float* X           [[ buffer(1) ]],
    device const float* Y           [[ buffer(2) ]],
    device float*       O           [[ buffer(3) ]],
    device const uint* OUT_SHAPE    [[ buffer(4) ]],
    device const uint* OUT_STRIDES  [[ buffer(5) ]],
    device const uint* C_SHAPE      [[ buffer(6) ]],
    device const uint* C_STRIDES    [[ buffer(7) ]],
    device const uint* X_SHAPE      [[ buffer(8) ]],
    device const uint* X_STRIDES    [[ buffer(9) ]],
    device const uint* Y_SHAPE      [[ buffer(10) ]],
    device const uint* Y_STRIDES    [[ buffer(11) ]],
    constant RC& rc                 [[ buffer(12) ]],
    constant RB& rb                 [[ buffer(13) ]],
    uint gid [[thread_position_in_grid]]
){
    if (gid >= rc.n) return;
    uint rem = gid;
    uint oc = rb.c_base, ox = rb.x_base, oy = rb.y_base;
    for (uint d=0u; d<rc.nd; ++d) {
        uint s  = OUT_STRIDES[d];
        uint ho = OUT_SHAPE[d];
        uint cd = (rem / s) % ho;
        rem = rem % s;
        if (C_SHAPE[d] != 1u) { oc += cd * C_STRIDES[d]; }
        if (X_SHAPE[d] != 1u) { ox += cd * X_STRIDES[d]; }
        if (Y_SHAPE[d] != 1u) { oy += cd * Y_STRIDES[d]; }
    }
    uchar cb = C[oc];
    float vx = X[ox];
    float vy = Y[oy];
    O[gid] = (cb != 0) ? vx : vy;
}

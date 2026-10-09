ROUNDED_ADD
WIDE_ARITHMETIC
struct Params { rows:u32, cols:u32, input_offset:u32, groups_x:u32,
    groups:u32, padding0:u32, padding1:u32, padding2:u32 };
@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@group(0) @binding(2) var<storage,read_write> flags:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> p:Params;
fn checked(v:f32)->f32 {
    if ((bitcast<u32>(v)&0x7f800000u)==0x7f800000u) {
        atomicOr(&flags[0],INVALID_TENSOR_FLAG); return 0.0;
    }
    return v;
}
fn invocation(w:vec3<u32>,lane:u32)->u32 {
    let g=w.y*p.groups_x+w.x;
    if(g>=p.groups) {return 0xffffffffu;}
    return g*256u+lane;
}
fn root_weight(v:f32,maximum:f32)->f32 {
    // Halve first: subtraction cannot overflow for opposite finite extremes.
    return exp(0.5*v-0.5*maximum);
}
@compute @workgroup_size(256)
fn forward(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let row=invocation(w,lane);if(row>=p.rows) {return;}
    var maximum=checked(input[p.input_offset+row*p.cols]);
    for(var i=1u;i<p.cols;i++) {maximum=max(maximum,checked(input[p.input_offset+row*p.cols+i]));}
    var square=parts(0.0);
    for(var i=0u;i<p.cols;i++) {
        let r=parts(root_weight(checked(input[p.input_offset+row*p.cols+i]),maximum));
        square=wide_add(square,wide_mul(r,r));
    }
    let norm=wide_sqrt(square);
    for(var i=0u;i<p.cols;i++) {
        let r=parts(root_weight(checked(input[p.input_offset+row*p.cols+i]),maximum));
        output[row*p.cols+i]=checked(wide_float(wide_div(r,norm)));
    }
}

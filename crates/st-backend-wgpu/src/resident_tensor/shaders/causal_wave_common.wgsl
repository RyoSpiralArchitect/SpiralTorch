struct Params {
    batch:u32, steps:u32, cols:u32, pairs:u32,
    values:u32, state_values:u32, groups_x:u32, groups:u32,
    radius:f32, drive_offset:u32, decay_offset:u32, phase_offset:u32,
    initial_offset:u32, cotangent_offset:u32, terminal_offset:u32, padding:u32,
};
ROUNDED_ADD
fn checked(v:f32)->f32 {
    if ((bitcast<u32>(v)&0x7f800000u)==0x7f800000u) {
        atomicOr(&flags[0],INVALID_TENSOR_FLAG); return 0.0;
    }
    return v;
}
fn sum(a:f32,b:f32)->f32 { return checked(bitcast<f32>(rounded_add_bits(bitcast<u32>(a),bitcast<u32>(b)))); }
fn difference(a:f32,b:f32)->f32 { return sum(a,-b); }
fn product(a:f32,b:f32)->f32 { return checked(a*b); }
fn invocation(wid:vec3<u32>, lane:u32)->u32 {
    let group=wid.y*params.groups_x+wid.x;
    if (group>=params.groups) { return 0xffffffffu; }
    return group*256u+lane;
}

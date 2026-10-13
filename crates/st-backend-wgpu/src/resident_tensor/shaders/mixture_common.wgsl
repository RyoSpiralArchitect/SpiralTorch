struct Params {
    batch:u32, heads:u32, positions:u32, components:u32,
    groups_x:u32, groups:u32, pad0:u32, pad1:u32,
};
ROUNDED_ADD
fn checked(v:f32)->f32 {
    if ((bitcast<u32>(v)&0x7f800000u)==0x7f800000u) {
        atomicOr(&flags[0],INVALID_TENSOR_FLAG); return 0.0;
    }
    return v;
}
fn sum(a:f32,b:f32)->f32 { return checked(rounded_add(a,b)); }
fn product(a:f32,b:f32)->f32 { return checked(a*b); }
fn invocation(wid:vec3<u32>,lane:u32)->u32 {
    let group=wid.y*params.groups_x+wid.x;
    if (group>=params.groups) { return 0xffffffffu; }
    return group*256u+lane;
}

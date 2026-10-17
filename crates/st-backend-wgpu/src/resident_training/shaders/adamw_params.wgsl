struct Params {
    len: u32, groups_x: u32, count: u32, index: u32,
    rate: f32, decay: f32, beta1: f32, beta2: f32,
    epsilon: f32, padding0: u32, padding1: u32, padding2: u32,
};
fn index(wid: vec3<u32>, lane: u32) -> u32 {
    return (wid.y * p.groups_x + wid.x) * 256u + lane;
}

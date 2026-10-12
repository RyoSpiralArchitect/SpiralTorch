@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> raw_logits:array<f32>;
@group(0) @binding(2) var<storage,read_write> mass:array<f32>;
@group(0) @binding(3) var<storage,read_write> output:array<f32>;
@group(0) @binding(4) var<storage,read_write> flags:array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params:Params;
COMMON
@compute @workgroup_size(256)
fn weights(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let h=invocation(wid,lane); if (h>=params.heads) { return; }
    let begin=h*params.components;
    var maximum=checked(raw_logits[begin]);
    for(var k=1u;k<params.components;k++) { maximum=max(maximum,checked(raw_logits[begin+k])); }
    var total=0.0;
    for(var k=0u;k<params.components;k++) {
        let d=raw_logits[begin+k]-maximum;
        let w=checked(exp(d)); mass[begin+k]=w; total=sum(total,w);
    }
    for(var k=0u;k<params.components;k++) { mass[begin+k]=checked(mass[begin+k]/total); }
}
@compute @workgroup_size(256)
fn forward(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let i=invocation(wid,lane); if(i>=params.batch*params.heads*params.positions) { return; }
    let h=(i/params.positions)%params.heads;
    var result=0.0;
    for(var k=0u;k<params.components;k++) {
        result=sum(result,product(checked(input[i*params.components+k]),mass[h*params.components+k]));
    }
    output[i]=result;
}

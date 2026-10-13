@group(0) @binding(0) var<storage,read> input:array<f32>;
@group(0) @binding(1) var<storage,read> mass:array<f32>;
@group(0) @binding(2) var<storage,read> seed:array<f32>;
@group(0) @binding(3) var<storage,read_write> input_grad:array<f32>;
@group(0) @binding(4) var<storage,read_write> logits_grad:array<f32>;
@group(0) @binding(5) var<storage,read_write> flags:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> params:Params;
COMMON
EXACT_ROUTER
@compute @workgroup_size(256)
fn components(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let i=invocation(wid,lane);
    if(i>=params.batch*params.heads*params.positions*params.components) { return; }
    let k=i%params.components;
    let row=i/params.components;
    let h=(row/params.positions)%params.heads;
    input_grad[i]=product(checked(seed[row]),mass[h*params.components+k]);
}
@compute @workgroup_size(256)
fn logits(@builtin(workgroup_id) wid:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let h=invocation(wid,lane); if(h>=params.heads) { return; }
    for(var k=0u;k<params.components;k++) {
        var total=RouterSum();
        for(var b=0u;b<params.batch;b++) {
            for(var p=0u;p<params.positions;p++) {
                let row=(b*params.heads+h)*params.positions+p;
                let x=checked(input[row*params.components+k]);
                let s=checked(seed[row]);let wk=mass[h*params.components+k];
                for(var j=0u;j<params.components;j++) {
                    if(j==k) {continue;}
                    let wj=mass[h*params.components+j];
                    let y=checked(input[row*params.components+j]);
                    router_product(&total,vec4<f32>(s,wk,wj,x),false);
                    router_product(&total,vec4<f32>(s,wk,wj,y),true);
                }
            }
        }
        logits_grad[h*params.components+k]=checked(router_float(&total));
    }
}

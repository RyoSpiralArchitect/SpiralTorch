COMMON
WIDE_ARITHMETIC
@group(0) @binding(0) var<storage,read> drive:array<f32>;
@group(0) @binding(1) var<storage,read> tape:array<f32>;
@group(0) @binding(2) var<storage,read> initial:array<f32>;
@group(0) @binding(3) var<storage,read> cotangent:array<f32>;
@group(0) @binding(4) var<storage,read> terminal:array<f32>;
// Do not narrow chart components before their rotation contraction in BPTT.
@group(0) @binding(5) var<storage,read_write> state_gradient:array<Wide>;
// Drive VJP, initial-state VJP, per-batch parameter partials, final parameter VJPs.
@group(0) @binding(6) var<storage,read_write> gradient:array<f32>;
@group(0) @binding(7) var<storage,read_write> flags:array<atomic<u32>>;
@group(0) @binding(8) var<uniform> params:Params;

@compute @workgroup_size(256)
fn project_vjp(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let row=invocation(wid,lane);if (row>=params.batch*params.steps) { return; }
    let base=row*params.cols;var maximum=0.0;var pivot=0u;
    for (var c=0u;c<params.cols;c++) { if (abs(tape[base+c])>maximum) { maximum=abs(tape[base+c]);pivot=c; } }
    let sp=parts(tape[base+pivot]);
    let gp=parts(checked(cotangent[params.cotangent_offset+base+pivot]));
    var q=parts(1.0);var dot=parts(0.0);
    for (var c=0u;c<params.cols;c++) {
        let s=parts(tape[base+c]);let g=parts(checked(cotangent[params.cotangent_offset+base+c]));
        var residual=g;
        if (sp.hi!=0.0) { residual=wide_div(wide_difference_of_products(g,sp,gp,s),sp); }
        state_gradient[base+c]=residual;
        q=wide_add(q,wide_mul(s,s));dot=wide_add(dot,wide_mul(s,residual));
    }
    let inverse=wide_div(parts(1.0),q);
    let scale=wide_div(parts(params.radius),wide_sqrt(q));
    let projection=wide_mul(dot,inverse);
    let radial_scale=wide_mul(scale,wide_mul(wide_div(gp,sp),inverse));
    for (var c=0u;c<params.cols;c++) {
        let s=parts(tape[base+c]);let residual=state_gradient[base+c];
        state_gradient[base+c]=wide_add(
            wide_mul(scale,wide_sub(residual,wide_mul(s,projection))),
            wide_mul(radial_scale,s));
    }
}

@compute @workgroup_size(256)
fn pullback(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let id=invocation(wid,lane);if (id>=params.batch*params.pairs) { return; }
    let b=id/params.pairs;let p=id%params.pairs;let coeff=params.values+5u*p;
    let rho=tape[coeff];let cs=tape[coeff+1u];let sn=tape[coeff+2u];
    let dr=parts(tape[coeff+3u]);let dt=parts(tape[coeff+4u]);
    let r=parts(rho);let one_minus=parts(1.0-rho);let cosine=parts(cs);let sine=parts(sn);
    var gx=parts(checked(terminal[params.terminal_offset+b*params.cols+2u*p]));
    var gy=parts(checked(terminal[params.terminal_offset+b*params.cols+2u*p+1u]));
    var gd=parts(0.0);var gp=parts(0.0);var t=params.steps;
    loop {
        if (t==0u) { break; }t--;
        let i=(b*params.steps+t)*params.cols+2u*p;
        gx=wide_add(gx,state_gradient[i]);gy=wide_add(gy,state_gradient[i+1u]);
        gradient[i]=checked(wide_float(wide_mul(one_minus,gx)));
        gradient[i+1u]=checked(wide_float(wide_mul(one_minus,gy)));
        var x=0.0;var y=0.0;
        if (t==0u) { x=initial[params.initial_offset+b*params.cols+2u*p];y=initial[params.initial_offset+b*params.cols+2u*p+1u]; }
        else { x=tape[i-params.cols];y=tape[i-params.cols+1u]; }
        let rx=parts(difference(product(cs,x),product(sn,y)));
        let ry=parts(sum(product(sn,x),product(cs,y)));
        let dx=wide_mul(gx,wide_sub(rx,parts(drive[params.drive_offset+i])));
        let dy=wide_mul(gy,wide_sub(ry,parts(drive[params.drive_offset+i+1u])));
        gd=wide_add(gd,wide_mul(wide_add(dx,dy),dr));
        gp=wide_add(gp,wide_mul(wide_mul(r,wide_difference_of_products(gy,rx,gx,ry)),dt));
        let nx=wide_mul(r,wide_add(wide_mul(cosine,gx),wide_mul(sine,gy)));
        let ny=wide_mul(r,wide_difference_of_products(cosine,gy,sine,gx));
        gx=nx;gy=ny;
    }
    let start=params.values+b*params.cols+2u*p;
    gradient[start]=checked(wide_float(gx));gradient[start+1u]=checked(wide_float(gy));
    let partial=params.values+params.state_values+2u*id;
    gradient[partial]=checked(wide_float(gd));gradient[partial+1u]=checked(wide_float(gp));
}

@compute @workgroup_size(256)
fn reduce_parameters(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let p=invocation(wid,lane);if (p>=params.pairs) { return; }
    var gd=0.0;var gp=0.0;
    for (var b=0u;b<params.batch;b++) {
        let i=params.values+params.state_values+2u*(b*params.pairs+p);
        gd=sum(gd,gradient[i]);gp=sum(gp,gradient[i+1u]);
    }
    let base=params.values+2u*params.state_values;
    gradient[base+p]=gd;gradient[base+params.pairs+p]=gp;
}

COMMON
@group(0) @binding(0) var<storage,read> drive:array<f32>;
@group(0) @binding(1) var<storage,read> decay:array<f32>;
@group(0) @binding(2) var<storage,read> phase:array<f32>;
@group(0) @binding(3) var<storage,read> initial:array<f32>;
// All raw states, then P*(rho,cos,sin,drho,dtheta), then terminal raw state.
@group(0) @binding(4) var<storage,read_write> tape:array<f32>;
@group(0) @binding(5) var<storage,read_write> features:array<f32>;
@group(0) @binding(6) var<storage,read_write> flags:array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params:Params;

@compute @workgroup_size(256)
fn scan(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let id=invocation(wid,lane); if (id>=params.batch*params.pairs) { return; }
    let b=id/params.pairs; let p=id%params.pairs;
    let a=checked(decay[params.decay_offset+p]);
    var sigmoid=0.0;
    if (a>=0.0) { sigmoid=1.0/(1.0+exp(-a)); } else { let e=exp(a); sigmoid=e/(1.0+e); }
    let h=tanh(checked(phase[params.phase_offset+p]));
    let theta=PHASE_LIMIT*h;
    let rho=MAX_DECAY*sigmoid; let cs=cos(theta); let sn=sin(theta);
    if (b==0u) {
        let base=params.values+5u*p;
        tape[base]=rho; tape[base+1u]=cs; tape[base+2u]=sn;
        tape[base+3u]=MAX_DECAY*sigmoid*(1.0-sigmoid);
        tape[base+4u]=PHASE_LIMIT*(1.0-h*h);
    }
    var x=checked(initial[params.initial_offset+b*params.cols+2u*p]);
    var y=checked(initial[params.initial_offset+b*params.cols+2u*p+1u]);
    for (var t=0u;t<params.steps;t++) {
        let i=(b*params.steps+t)*params.cols+2u*p;
        let rx=difference(product(cs,x),product(sn,y));
        let ry=sum(product(sn,x),product(cs,y));
        x=sum(product(rho,rx),product(1.0-rho,checked(drive[params.drive_offset+i])));
        y=sum(product(rho,ry),product(1.0-rho,checked(drive[params.drive_offset+i+1u])));
        tape[i]=x; tape[i+1u]=y;
    }
    let last=params.values+5u*params.pairs+b*params.cols+2u*p;
    tape[last]=x; tape[last+1u]=y;
}

@compute @workgroup_size(256)
fn project(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    let row=invocation(wid,lane); if (row>=params.batch*params.steps) { return; }
    let base=row*params.cols; var scale=1.0;
    for (var c=0u;c<params.cols;c++) { scale=max(scale,abs(tape[base+c])); }
    let inv=1.0/scale; var squared=inv*inv;
    for (var c=0u;c<params.cols;c++) { let v=tape[base+c]/scale; squared=sum(squared,product(v,v)); }
    let root=sqrt(squared);
    for (var c=0u;c<params.cols;c++) { features[base+c]=product(params.radius,(tape[base+c]/scale)/root); }
}

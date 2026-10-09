ROUNDED_ADD
WIDE_ARITHMETIC
struct Params {
    batch:u32, steps:u32, cols:u32, heads:u32,
    coordinates:u32, pairs:u32, scores:u32, groups_x:u32,
    groups:u32, coordinates_offset:u32, gain_offset:u32, seed_offset:u32,
    curvature_magnitude:f32, distance_scale:f32, metric_kind:u32, padding:u32,
};
@group(0) @binding(0) var<storage,read> coordinates:array<f32>;
@group(0) @binding(1) var<storage,read> raw_gain:array<f32>;
// Four extended values per pair: squared distance, base and two radial factors.
@group(0) @binding(2) var<storage,read_write> cache:array<Wide>;
@group(0) @binding(3) var<storage,read> cotangent:array<f32>;
// Float outputs followed, in backward only, by one Wide scale per pair.
// Integer storage preserves the Wide exponent bits without a float roundtrip.
@group(0) @binding(4) var<storage,read_write> output:array<u32>;
@group(0) @binding(5) var<storage,read_write> flags:array<atomic<u32>>;
@group(0) @binding(6) var<uniform> p:Params;
fn checked(x:f32)->f32 {
    if ((bitcast<u32>(x)&0x7f800000u)==0x7f800000u) {
        atomicOr(&flags[0],INVALID_TENSOR_FLAG);return 0.0;
    }
    return x;
}
fn invocation(w:vec3<u32>,lane:u32)->u32 {
    let group=w.y*p.groups_x+w.x;
    if (group>=p.groups) { return 0xffffffffu; }
    return group*256u+lane;
}
fn coordinate(row:u32,col:u32)->Wide {
    return parts(checked(coordinates[p.coordinates_offset+row*p.cols+col]));
}
fn margin(norm:Wide)->Wide {
    let a=wide_sub(parts(1.0),wide_mul(parts(p.curvature_magnitude),norm));
    if (a.hi<=0.0) { atomicOr(&flags[0],INVALID_TENSOR_FLAG);return parts(1.0); }
    return a;
}
struct Gain { value:Wide, derivative:Wide };
fn gain(head:u32)->Gain {
    let raw=checked(raw_gain[p.gain_offset+head]);
    var e=parts(0.0);
    // Beyond this range even the extended metric/seed products cannot rescue
    // a representable f32 output. Avoid converting an unbounded f32 to i32.
    if (abs(raw)<1400.0) {
        let exponent=-abs(raw)*1.4426950408889634;
        let whole=floor(exponent);
        e=parts(exp2(exponent-whole));e.exponent+=i32(whole);
    }
    var log_one_plus=parts(0.0);
    if (abs(raw)>=5.0) {
        // log(1+e) without erasing tiny e by first rounding 1+e to f32.
        let e2=wide_mul(e,e);
        let e3=wide_mul(e2,e);
        let e4=wide_mul(e3,e);
        log_one_plus=wide_add(wide_sub(e,wide_mul(parts(0.5),e2)),
            wide_sub(wide_mul(parts(1.0/3.0),e3),wide_mul(parts(0.25),e4)));
    } else { log_one_plus=parts(log(1.0+exp(-abs(raw)))); }
    let denominator=wide_add(parts(1.0),e);
    if (raw>=0.0) { return Gain(wide_add(parts(raw),log_one_plus),wide_div(parts(1.0),denominator)); }
    return Gain(log_one_plus,wide_div(e,denominator));
}
@compute @workgroup_size(256)
fn prepare_pairs(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let id=invocation(w,lane);if (id>=p.pairs) { return; }
    let k=id%p.steps;let q=(id/p.steps)%p.steps;let b=id/(p.steps*p.steps);
    if (k>q) { for(var j=0u;j<4u;j++) { cache[4u*id+j]=parts(0.0); }return; }
    var nx=parts(0.0);var ny=parts(0.0);var square=parts(0.0);
    for(var i=0u;i<p.cols;i++) {
        let x=coordinate(b*p.steps+q,i);let y=coordinate(b*p.steps+k,i);
        let delta=wide_sub(x,y);
        nx=wide_add(nx,wide_mul(x,x));ny=wide_add(ny,wide_mul(y,y));
        square=wide_add(square,wide_mul(delta,delta));
    }
    if (p.metric_kind==1u) {
        let scale=parts(p.distance_scale);
        cache[4u*id]=wide_mul(scale,square);
        cache[4u*id+1u]=wide_mul(parts(2.0),scale);
        cache[4u*id+2u]=parts(0.0);
        cache[4u*id+3u]=parts(0.0);
        return;
    }
    let a=margin(nx);let d=margin(ny);let ad=wide_mul(a,d);
    let c=parts(p.curvature_magnitude);
    let v=wide_div(wide_mul(c,square),ad);
    var distance=parts(0.0);var kernel=parts(1.0);
    if (v.hi==0.0 || v.exponent < -9) {
        // Analytic small-v series, not a constant-distance epsilon deadzone.
        let t=wide_float(v);
        let f=1.0+t*(-1.0/3.0+t*(8.0/45.0+t*(-4.0/35.0+t*128.0/1575.0)));
        kernel=parts(1.0+t*(-2.0/3.0+t*(8.0/15.0+t*(-16.0/35.0+t*128.0/315.0))));
        distance=wide_div(wide_mul(parts(4.0*f),square),ad);
    } else {
        let root=wide_sqrt(v);let other=wide_sqrt(wide_add(parts(1.0),v));
        let sum=wide_add(root,other);
        let angle=log(sum.hi)+f32(sum.exponent)*0.6931471805599453;
        distance=wide_div(wide_mul(parts(4.0),wide_mul(parts(angle),parts(angle))),c);
        kernel=wide_div(parts(angle),wide_mul(root,other));
    }
    let eight=wide_mul(parts(8.0),kernel);
    cache[4u*id]=distance;
    cache[4u*id+1u]=wide_div(eight,ad);
    cache[4u*id+2u]=wide_div(wide_mul(eight,v),a);
    cache[4u*id+3u]=wide_div(wide_mul(eight,v),d);
}
@compute @workgroup_size(256)
fn scores(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let id=invocation(w,lane);if(id>=p.scores) { return; }
    let pair=id%(p.steps*p.steps);let h=(id/(p.steps*p.steps))%p.heads;
    let b=id/(p.heads*p.steps*p.steps);let g=gain(h);
    output[id]=bitcast<u32>(checked(wide_float(wide_neg(wide_mul(g.value,cache[4u*(b*p.steps*p.steps+pair)])))));
}
fn pair_seed(b:u32,q:u32,k:u32)->Wide {
    var scale=parts(0.0);
    for(var h=0u;h<p.heads;h++) {
        let seed=parts(checked(cotangent[p.seed_offset+((b*p.heads+h)*p.steps+q)*p.steps+k]));
        scale=wide_sub(scale,wide_mul(gain(h).value,seed));
    }
    return scale;
}
@compute @workgroup_size(256)
fn prepare_pair_seeds(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let id=invocation(w,lane);if(id>=p.pairs) { return; }
    let k=id%p.steps;let q=(id/p.steps)%p.steps;let b=id/(p.steps*p.steps);
    var scale=parts(0.0);
    if(k<=q) { scale=pair_seed(b,q,k); }
    let offset=p.coordinates+p.heads+4u*id;
    output[offset]=bitcast<u32>(scale.hi);
    output[offset+1u]=bitcast<u32>(scale.lo);
    output[offset+2u]=bitcast<u32>(scale.exponent);
    output[offset+3u]=bitcast<u32>(scale.tail);
}
fn pair_scale(pair:u32)->Wide {
    let offset=p.coordinates+p.heads+4u*pair;
    return Wide(bitcast<f32>(output[offset]),bitcast<f32>(output[offset+1u]),
        bitcast<i32>(output[offset+2u]),bitcast<f32>(output[offset+3u]));
}
@compute @workgroup_size(256)
fn coordinates_vjp(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let id=invocation(w,lane);if(id>=p.coordinates) { return; }
    let i=id%p.cols;let t=(id/p.cols)%p.steps;let b=id/(p.steps*p.cols);
    let x=coordinate(b*p.steps+t,i);var gradient=parts(0.0);
    for(var k=0u;k<p.steps;k++) {
        let q=max(t,k);let key=min(t,k);let offset=4u*((b*p.steps+q)*p.steps+key);
        let y=coordinate(b*p.steps+k,i);
        let radial=select(3u,2u,k<=t);
        let derivative=wide_add(wide_mul(cache[offset+1u],wide_sub(x,y)),wide_mul(cache[offset+radial],x));
        gradient=wide_add(gradient,wide_mul(pair_scale(offset/4u),derivative));
    }
    output[id]=bitcast<u32>(checked(wide_float(gradient)));
}
@compute @workgroup_size(256)
fn gain_vjp(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let h=invocation(w,lane);if(h>=p.heads) { return; }
    var gradient=parts(0.0);let dg=gain(h).derivative;
    for(var b=0u;b<p.batch;b++) {
        for(var q=0u;q<p.steps;q++) {
            for(var k=0u;k<p.steps;k++) {
                // Check masked positions as well; masking is not a NaN escape.
                let seed=parts(checked(cotangent[p.seed_offset+((b*p.heads+h)*p.steps+q)*p.steps+k]));
                if(k<=q) { gradient=wide_sub(gradient,wide_mul(seed,wide_mul(dg,cache[4u*((b*p.steps+q)*p.steps+k)]))); }
            }
        }
    }
    output[p.coordinates+h]=bitcast<u32>(checked(wide_float(gradient)));
}

struct Params {
    rows: u32, classes: u32, row_groups_x: u32, label_groups_x: u32,
    reduction: u32, ignore_enabled: u32, ignore: f32, smoothing: f32,
    uniform_mass: f32, target_adjustment: f32, nll_m: f32, nll_e: i32,
    uniform_m: f32, uniform_e: i32, pad0: u32, pad1: u32,
}
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> labels: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradient: array<f32>;
@group(0) @binding(3) var<storage, read_write> row_loss: array<f32>;
@group(0) @binding(4) var<storage, read_write> value: array<f32>;
@group(0) @binding(5) var<storage, read_write> active_count: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> validation: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> p: Params;
var<workgroup> scratch: array<f32, 256>;
var<workgroup> indices: array<u32, 256>;

fn finite(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u; }
fn checked(x: f32) -> f32 {
    if (!finite(x)) { atomicOr(&validation[0], 1u); return 0.0; }
    return x;
}
fn integral_label(x: f32) -> bool {
    return finite(x) && x >= -9223372036854775808.0 && x < 9223372036854775808.0 && floor(x) == x;
}
fn ignored(x: f32) -> bool { return p.ignore_enabled != 0u && x == p.ignore; }
fn class_label(x: f32) -> bool {
    if (!integral_label(x) || x < 0.0 || x >= 4294967296.0) { return false; }
    return u32(x) < p.classes;
}

@compute @workgroup_size(256)
fn count_labels(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let i = (group.y * p.label_groups_x + group.x) * 256u + lane;
    if (i >= p.rows) { return; }
    let label = labels[i];
    if (!integral_label(label)) { atomicOr(&validation[0], 2u); }
    else if (!ignored(label)) {
        if (!class_label(label)) { atomicOr(&validation[0], 2u); }
        else { atomicAdd(&active_count, 1u); }
    }
}

struct Gap { fraction: f32, exponent: i32 }
// Subtract aligned significands instead of overflowing finite, opposite logits.
fn gap_parts(a: f32, b: f32) -> Gap {
    if (a==b) { return Gap(0.0,0); }
    if (max(abs(a),abs(b))<=1.701411733e38) {
        let parts=frexp(a-b);
        return Gap(parts.fract,parts.exp);
    }
    let ap=frexp(a); let bp=frexp(b);
    let exponent=max(ap.exp,bp.exp);
    let aligned=ldexp(ap.fract,max(ap.exp-exponent,-149)) - ldexp(bp.fract,max(bp.exp-exponent,-149));
    let parts=frexp(aligned);
    return Gap(parts.fract,exponent+parts.exp);
}
// Tiny smoothing remains a mantissa/exponent until after multiplication.
fn scaled_gap(a: f32, b: f32, m: f32, e: i32, divisor: f32) -> f32 {
    if (m==0.0 || a==b) { return 0.0; }
    let gap=gap_parts(a,b);
    let base = (gap.fraction * m) / divisor;
    if (base == 0.0) { return 0.0; }
    let parts = frexp(base);
    let exponent = parts.exp + gap.exponent + e;
    if (exponent > 128) { atomicOr(&validation[0], 4u); return 0.0; }
    if (exponent < -149) { return 0.0; }
    let bounded = clamp(exponent, -126, 127);
    return checked(ldexp(parts.fract, bounded) * exp2(f32(exponent - bounded)));
}
fn weight(x: f32, maximum: f32) -> f32 {
    if (x==maximum) { return 1.0; }
    if (max(abs(x),abs(maximum))<=1.701411733e38) {
        let gap=maximum-x;
        if (gap>104.0) { return 0.0; }
        return exp(-gap);
    }
    let gap=gap_parts(maximum,x);
    if (gap.fraction==0.0) { return 1.0; }
    // At gaps >= 128 even the correctly rounded f32 exponential is zero.
    if (gap.exponent>7) { return 0.0; }
    return exp(-ldexp(gap.fraction,gap.exponent));
}
fn log_one_plus(tail: f32) -> f32 {
    let whole = 1.0 + tail;
    if (whole == 1.0) { return tail; }
    return log(whole) * (tail / (whole - 1.0));
}

@compute @workgroup_size(256)
fn classify_rows(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let row = group.y * p.row_groups_x + group.x;
    if (row >= p.rows) { return; }
    let start = row * p.classes;
    var maximum = -3.402823466e38;
    var maximum_index = 0xffffffffu;
    for (var c = lane; c < p.classes; c += 256u) {
        let x = checked(logits[start + c]);
        if (x > maximum || (x == maximum && c < maximum_index)) { maximum=x; maximum_index=c; }
    }
    scratch[lane]=maximum; indices[lane]=maximum_index;
    workgroupBarrier();
    for (var s=128u; s>0u; s/=2u) {
        if (lane<s) {
            if (scratch[lane+s]>scratch[lane] || (scratch[lane+s]==scratch[lane] && indices[lane+s]<indices[lane])) {
                scratch[lane]=scratch[lane+s]; indices[lane]=indices[lane+s];
            }
        }
        workgroupBarrier();
    }
    maximum=scratch[0]; maximum_index=indices[0];
    workgroupBarrier();
    var tail=0.0;
    for (var c=lane; c<p.classes; c+=256u) {
        if (c!=maximum_index) { tail+=weight(checked(logits[start+c]),maximum); }
    }
    scratch[lane]=tail; workgroupBarrier();
    for (var s=128u; s>0u; s/=2u) {
        if (lane<s) { scratch[lane]+=scratch[lane+s]; }
        workgroupBarrier();
    }
    tail=scratch[0];
    workgroupBarrier();
    let label_value=labels[row];
    let valid_sample=class_label(label_value) && !ignored(label_value);
    var label=maximum_index;
    if (valid_sample) { label=u32(label_value); }
    var divisor=1.0;
    if (p.reduction==2u) { divisor=f32(max(atomicLoad(&active_count),1u)); }
    var uniform_loss=0.0;
    for (var c=lane; c<p.classes; c+=256u) {
        let x=checked(logits[start+c]);
        var seed=0.0;
        if (valid_sample) {
            let is_target=c==label;
            if (c==maximum_index) {
                seed=select(1.0-p.uniform_mass,p.target_adjustment,is_target) - tail/(1.0+tail);
            } else {
                seed=weight(x,maximum)/(1.0+tail);
                if (is_target) { seed=(seed-1.0)+p.target_adjustment; }
                else { seed-=p.uniform_mass; }
            }
            uniform_loss+=scaled_gap(maximum,x,p.uniform_m,p.uniform_e,divisor);
        }
        gradient[start+c]=checked(seed/divisor);
    }
    if (lane==0u && valid_sample) {
        uniform_loss+=scaled_gap(maximum,checked(logits[start+label]),p.nll_m,p.nll_e,divisor) + log_one_plus(tail)/divisor;
    }
    scratch[lane]=checked(uniform_loss); workgroupBarrier();
    for (var s=128u; s>0u; s/=2u) {
        if (lane<s) { scratch[lane]=checked(scratch[lane]+scratch[lane+s]); }
        workgroupBarrier();
    }
    if (lane==0u) {
        row_loss[row]=scratch[0];
        if (p.reduction==0u) { value[row]=scratch[0]; }
    }
}

@compute @workgroup_size(256)
fn reduce_loss(@builtin(local_invocation_index) lane: u32) {
    if (p.reduction==0u) { return; }
    if (p.reduction==2u && atomicLoad(&active_count)==0u) { atomicOr(&validation[0],8u); }
    var total=0.0;
    for (var r=lane; r<p.rows; r+=256u) { total=checked(total+row_loss[r]); }
    scratch[lane]=total; workgroupBarrier();
    for (var s=128u; s>0u; s/=2u) {
        if (lane<s) { scratch[lane]=checked(scratch[lane]+scratch[lane+s]); }
        workgroupBarrier();
    }
    if (lane==0u) { value[0]=scratch[0]; }
}

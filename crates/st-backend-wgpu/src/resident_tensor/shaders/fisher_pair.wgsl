fn fisher_logit(row:u32,i:u32)->f32 {
    return bitcast<f32>(coordinates[p.coordinates_offset+5u*(row*p.cols+i)+4u]);
}
fn fisher_relative_logit(q:u32,k:u32,pivot:u32,i:u32)->Wide {
    // Four primitive terms, not two independently rounded logit differences.
    var value=wide_sum_zero();
    value=wide_sum_add(value,parts(fisher_logit(q,i)));
    value=wide_sum_add(value,wide_neg(parts(fisher_logit(q,pivot))));
    value=wide_sum_add(value,wide_neg(parts(fisher_logit(k,i))));
    value=wide_sum_add(value,parts(fisher_logit(k,pivot)));
    return wide_sum_value(value);
}
fn fisher_close_delta(q:u32,k:u32,pivot:u32,i:u32,inverse:Wide,normalization:Wide)->Wide {
    let delta=wide_mul(parts(0.5),fisher_relative_logit(q,k,pivot,i));
    let deviation=fisher_expm1(delta);
    return wide_mul(coordinate(k,i),wide_sub(wide_mul(deviation,inverse),normalization));
}
fn fisher_prepare(q:u32,k:u32,id:u32) {
    var pivot=0u;
    for(var i=1u;i<p.cols;i++) {
        if(fisher_logit(k,i)>fisher_logit(k,pivot)) {pivot=i;}
    }
    var nearby=true;var base_square=parts(0.0);var linear=wide_sum_zero();var nonlinear=wide_sum_zero();
    for(var i=0u;i<p.cols;i++) {
        let delta=fisher_relative_logit(q,k,pivot,i);
        if(wide_sub(delta,parts(0.25)).hi>0.0 || wide_sub(delta,parts(-0.25)).hi<0.0) {
            nearby=false;break;
        }
        let root=coordinate(k,i);let square=wide_mul(root,root);
        base_square=wide_add(base_square,square);
        linear=wide_sum_add(linear,wide_mul(square,delta));
        nonlinear=wide_sum_add(nonlinear,wide_mul(square,fisher_exponential_remainder(delta)));
    }
    var inverse=parts(0.0);var normalization=parts(0.0);
    if(nearby) {
        let change=wide_add(wide_sum_value(linear),wide_sum_value(nonlinear));
        let a=wide_sqrt(wide_add(base_square,change));
        let b=wide_sqrt(base_square);
        inverse=wide_div(parts(1.0),a);
        // A-B = (A^2-B^2)/(A+B), without subtracting rounded norms.
        normalization=wide_div(change,wide_mul(wide_mul(a,b),wide_add(a,b)));
    }
    var square=parts(0.0);
    for(var i=0u;i<p.cols;i++) {
        var delta=wide_sub(coordinate(q,i),coordinate(k,i));
        if(nearby) {delta=fisher_close_delta(q,k,pivot,i,inverse,normalization);}
        square=wide_add(square,wide_mul(delta,delta));
    }
    let t=min(wide_float(square),2.0);let z=0.25*t;
    var distance=parts(0.0);var slope=parts(4.0);
    if(z<0.0001) {
        let ratio=1.0+z*(1.0/3.0+z*(8.0/45.0+z*(4.0/35.0+z*128.0/1575.0)));
        distance=wide_mul(parts(4.0*ratio),square);
        slope=parts(4.0*(1.0+z*(2.0/3.0+z*(8.0/15.0+z*(16.0/35.0+z*128.0/315.0)))));
    } else {
        let root=sqrt(z);let angle=asin(root);
        distance=parts(16.0*angle*angle);
        slope=parts(4.0*angle/(root*sqrt(1.0-z)));
    }
    cache[4u*id]=distance;cache[4u*id+1u]=wide_mul(parts(2.0),slope);
    // Reuse the otherwise unused radial slot as the close-pair flag. No
    // additional per-pair metadata or quadratic-by-category root tape.
    cache[4u*id+2u]=parts(select(0.0,1.0,nearby));cache[4u*id+3u]=parts(0.0);
}

struct FisherPullbacks { query:Wide, key:Wide };
fn fisher_mass_parts(value:f32,maximum:f32)->array<Wide,4> {
    var delta=wide_sum_add(wide_sum_zero(),parts(value));
    delta=wide_sum_add(delta,wide_neg(parts(maximum)));
    let zero=parts(0.0);
    // exp(-700), even with maximal gain/seed and u32-sized reductions,
    // cannot contribute a representable f32 close-chart gradient.
    if(wide_sub(delta.value,parts(-700.0)).hi<0.0) {return array<Wide,4>(zero,zero,zero,zero);}
    let exponent=round(wide_float(delta.value)*1.4426950408889634);
    let ln_two=wide_add(parts(0.6931471824645996),parts(-1.904654323148236e-9));
    let remainder=wide_sub(delta.value,wide_mul(parts(exponent),ln_two));
    var base=parts(1.0);base.exponent+=i32(exponent);
    let body=fisher_expm1(remainder);let tail=fisher_expm1(delta.correction);
    // Keep mass changes separate until after multiplication by delta.
    return array<Wide,4>(base,wide_mul(base,body),wide_mul(base,tail),wide_mul(wide_mul(base,body),tail));
}

// Coordinate-centered covariance of the normalized distributions. The close
// path is O(cols^2) per pair, but retains curvature signals without a larger
// pair-by-category tape. Numerators keep linear/remainder reductions separate.
fn fisher_close_pullbacks(q:u32,k:u32,i:u32)->FisherPullbacks {
    var maximum=fisher_logit(k,0u);
    for(var j=1u;j<p.cols;j++) {maximum=max(maximum,fisher_logit(k,j));}
    var square=parts(0.0);var linear=wide_sum_zero();
    var half=wide_sum_zero();var full=wide_sum_zero();
    for(var j=0u;j<p.cols;j++) {
        let delta=fisher_relative_logit(q,k,i,j);
        let half_delta=fisher_exponential_remainder(wide_mul(parts(0.5),delta));
        let full_delta=fisher_exponential_remainder(delta);
        var masses=fisher_mass_parts(fisher_logit(k,j),maximum);
        for(var part=0u;part<4u;part++) {
            let mass=masses[part];
            square=wide_add(square,mass);
            linear=wide_sum_add(linear,wide_mul(mass,delta));
            half=wide_sum_add(half,wide_mul(mass,half_delta));
            full=wide_sum_add(full,wide_mul(mass,full_delta));
        }
    }
    let linear_value=wide_sum_value(linear);let half_value=wide_sum_value(half);let full_value=wide_sum_value(full);
    let a2=wide_add(square,wide_add(linear_value,full_value));
    let ab=wide_sqrt(wide_mul(a2,square));
    var dx=wide_sum_zero();var dy=wide_sum_zero();
    dx=wide_sum_add(dx,wide_mul(parts(-0.5),linear_value));
    dx=wide_sum_add(dx,half_value);dx=wide_sum_add(dx,wide_neg(full_value));
    dy=wide_sum_add(dy,wide_mul(parts(0.5),linear_value));dy=wide_sum_add(dy,half_value);
    let masses=fisher_mass_parts(fisher_logit(k,i),maximum);
    let mass=wide_add(wide_add(masses[0],masses[1]),wide_add(masses[2],masses[3]));
    return FisherPullbacks(wide_div(wide_mul(mass,wide_sum_value(dx)),wide_mul(a2,ab)),
        wide_div(wide_mul(mass,wide_sum_value(dy)),wide_mul(square,ab)));
}

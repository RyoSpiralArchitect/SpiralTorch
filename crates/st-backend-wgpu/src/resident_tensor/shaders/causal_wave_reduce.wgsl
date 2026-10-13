// An expansion of scaled f32 terms, not a fixed number of compensation levels.
// Nonlinear chart/rotation contractions still have their own finite precision.
struct WaveScalar { value:f32, exponent:i32 };
struct WaveScalarPair { high:WaveScalar, low:WaveScalar };
fn wave_scalar(value:f32,exponent:i32)->WaveScalar {
    let a=parts(value);return WaveScalar(a.hi,a.exponent+exponent);
}
fn wave_scalar_add(a_in:WaveScalar,b_in:WaveScalar)->WaveScalarPair {
    var a=a_in;var b=b_in;
    if(a.value==0.0) {return WaveScalarPair(b,a);}
    if(b.value==0.0) {return WaveScalarPair(a,b);}
    if(a.exponent<b.exponent || (a.exponent==b.exponent && abs(a.value)<abs(b.value))) {
        a=b_in;b=a_in;
    }
    // Unsigned subtraction also handles widely separated signed exponents.
    let gap=bitcast<u32>(a.exponent)-bitcast<u32>(b.exponent);
    if(gap>25u) {return WaveScalarPair(a,b);}
    let small=ldexp(b.value,-i32(gap));
    let high=rounded_add(a.value,small);
    let low=rounded_add(small,-rounded_add(high,-a.value));
    return WaveScalarPair(wave_scalar(high,a.exponent),wave_scalar(low,a.exponent));
}
fn wave_load_scalar(index:u32)->WaveScalar {
    let offset=4u*params.values+2u*index;
    return wave_scalar(bitcast<f32>(state_gradient[offset]),bitcast<i32>(state_gradient[offset+1u]));
}
fn wave_store_scalar(index:u32,value:WaveScalar) {
    let offset=4u*params.values+2u*index;
    state_gradient[offset]=bitcast<u32>(value.value);
    state_gradient[offset+1u]=bitcast<u32>(value.exponent);
}
fn wave_load_adjoint(index:u32)->Wide {
    let offset=4u*index;
    return Wide(bitcast<f32>(state_gradient[offset]),bitcast<f32>(state_gradient[offset+1u]),
        bitcast<i32>(state_gradient[offset+2u]),bitcast<f32>(state_gradient[offset+3u]));
}
fn wave_store_adjoint(index:u32,value:Wide) {
    let offset=4u*index;
    state_gradient[offset]=bitcast<u32>(value.hi);state_gradient[offset+1u]=bitcast<u32>(value.lo);
    state_gradient[offset+2u]=bitcast<u32>(value.exponent);state_gradient[offset+3u]=bitcast<u32>(value.tail);
}
fn wave_store_contribution(parameter:u32,b:u32,t:u32,value:Wide) {
    let stride=3u*params.batch*params.steps+2u;
    let base=parameter*stride+3u*(b*params.steps+t);
    wave_store_scalar(base,WaveScalar(value.hi,value.exponent));
    wave_store_scalar(base+1u,WaveScalar(value.lo,value.exponent));
    wave_store_scalar(base+2u,WaveScalar(value.tail,value.exponent));
}
fn wave_insert(base:u32,count:u32,value:WaveScalar)->u32 {
    var x=value;var kept=0u;
    for(var i=0u;i<count;i++) {
        let pair=wave_scalar_add(x,wave_load_scalar(base+i));
        if(pair.low.value!=0.0) {wave_store_scalar(base+kept,pair.low);kept++;}
        x=pair.high;
    }
    if(x.value!=0.0) {wave_store_scalar(base+kept,x);kept++;}
    return kept;
}
fn wave_expansion_sign(base:u32,count:u32)->i32 {
    if(count==0u) {return 0;}
    return select(-1,1,wave_load_scalar(base+count-1u).value>0.0);
}
fn wave_reduce(parameter:u32)->f32 {
    let terms=3u*params.batch*params.steps;let base=parameter*(terms+2u);
    var count=0u;
    // The compacted prefix never overwrites the next unread input term.
    for(var i=0u;i<terms;i++) {count=wave_insert(base,count,wave_load_scalar(base+i));}
    if(count==0u) {return 0.0;}
    // A last cancellation can leave overlapping retained parts. Collapse
    // exact leading additions first, stopping at the first rounding tail.
    var leading=wave_load_scalar(base+count-1u);var remaining=count-1u;
    loop {
        if(remaining==0u) {break;}remaining--;
        let pair=wave_scalar_add(leading,wave_load_scalar(base+remaining));
        leading=pair.high;
        if(pair.low.value!=0.0) {break;}
    }
    var bits=bitcast<u32>(wide_float(Wide(leading.value,0.0,leading.exponent,0.0)));
    if((bits&0x7fffffffu)==0x7f800000u) {bits=(bits&0x80000000u)|0x7f7fffffu;}
    let candidate=bitcast<f32>(bits);
    count=wave_insert(base,count,wave_scalar(bitcast<f32>(bits^0x80000000u),0));
    let sign=wave_expansion_sign(base,count);
    if(sign==0) {return candidate;}
    var next_bits=0u;
    if((bits&0x7fffffffu)==0u) {next_bits=select(0x80000001u,1u,sign>0);}
    else if((sign>0)==((bits>>31u)==0u)) {next_bits=bits+1u;}
    else {next_bits=bits-1u;}
    var half=WaveScalar(select(-0.5,0.5,sign>0),104);
    if((next_bits&0x7fffffffu)!=0x7f800000u) {
        half=wave_scalar_add(wave_scalar(bitcast<f32>(next_bits),0),wave_scalar(bitcast<f32>(bits^0x80000000u),0)).high;
        half.exponent--;
    }
    half.value=-half.value;
    count=wave_insert(base,count,half);
    let comparison=wave_expansion_sign(base,count);
    if(comparison==sign || (comparison==0 && (bits&1u)!=0u)) {return checked(bitcast<f32>(next_bits));}
    if((bits&0x7fffffffu)==0u && sign<0) {return bitcast<f32>(0x80000000u);}
    return candidate;
}

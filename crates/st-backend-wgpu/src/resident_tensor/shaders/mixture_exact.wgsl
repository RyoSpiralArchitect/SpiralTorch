// Exact mixture-router polynomial, not a generic Wide replacement. Finite f32
// seed * weight_k * weight_j * component has quantum 2^-596 and magnitude
// below 2^256 (weights in [0,1]). Checked u32 input lengths bound the absolute
// sum of both signs below 2^289. Signed limbs cover [-608,320) without overflow.
struct RouterSum { words: array<u32,29> };

fn router_word(s:ptr<function,RouterSum>, index:u32, value:u32, subtract:bool) {
    var i=index; var carry=value;
    loop {
        if(carry==0u || i>=29u) {break;}
        let previous=(*s).words[i];
        if(subtract) {
            (*s).words[i]=previous-carry;
            carry=select(0u,1u,previous<carry);
        } else {
            let next=previous+carry;
            (*s).words[i]=next;
            carry=select(0u,1u,next<previous);
        }
        i++;
    }
}

fn router_product(s:ptr<function,RouterSum>, factors:vec4<f32>, subtract:bool) {
    var negative=subtract; var exponent=608i;
    var digits=array<u32,8>(); digits[0]=1u;
    for(var f=0u;f<4u;f++) {
        let bits=bitcast<u32>(factors[f]);
        negative=negative!=((bits>>31u)!=0u);
        let encoded=(bits>>23u)&255u;
        var mantissa=bits&0x7fffffu;
        if(encoded==0u) {exponent-=149i;}
        else {mantissa|=0x800000u;exponent+=i32(encoded)-150i;}
        if(mantissa==0u) {return;}
        // Four 24-bit significands fit exactly in eight base-4096 digits.
        // Each convolution/carry fits u32; no shader-f64 or FMA assumptions.
        var next=array<u32,8>(); var carry=0u;
        for(var i=0u;i<8u;i++) {
            var value=digits[i]*(mantissa&4095u)+carry;
            if(i!=0u) {value+=digits[i-1u]*(mantissa>>12u);}
            next[i]=value&4095u;carry=value>>12u;
        }
        digits=next;
    }
    for(var i=0u;i<8u;i++) {
        let offset=u32(exponent)+12u*i;let shift=offset%32u;
        router_word(s,offset/32u,digits[i]<<shift,negative);
        if(shift!=0u) {router_word(s,offset/32u+1u,digits[i]>>(32u-shift),negative);}
    }
}

fn router_float(s:ptr<function,RouterSum>)->f32 {
    let sign=(*s).words[28]&0x80000000u;
    var magnitude=(*s).words;
    if(sign!=0u) {
        var carry=1u;
        for(var i=0u;i<29u;i++) {
            let inverted=~magnitude[i];let next=inverted+carry;
            magnitude[i]=next;carry=select(0u,1u,next<inverted);
        }
    }
    var highest=-1i;
    for(var i=0u;i<29u;i++) {
        if(magnitude[i]!=0u) {highest=i32(32u*i+31u-countLeadingZeros(magnitude[i]));}
    }
    if(highest<0i) {return 0.0;}
    var exponent=highest-608i;
    if(exponent>127i) {return bitcast<f32>(sign|0x7f800000u);}
    // Round once to nearest/ties-even, including subnormals and remote sticky
    // bits. Intermediate f64/f32 conversion would lose exact tie information.
    let shift=u32(max(highest-23i,459i));
    var mantissa=magnitude[shift/32u]>>(shift%32u);
    if(shift%32u!=0u && shift/32u+1u<29u) {
        mantissa|=magnitude[shift/32u+1u]<<(32u-shift%32u);
    }
    let guard=shift-1u;
    let guard_set=(magnitude[guard/32u]&(1u<<(guard%32u)))!=0u;
    var sticky=(magnitude[guard/32u]&((1u<<(guard%32u))-1u))!=0u;
    for(var i=0u;i<guard/32u;i++) {sticky=sticky || magnitude[i]!=0u;}
    if(guard_set && (sticky || (mantissa&1u)!=0u)) {mantissa++;}
    if(exponent< -126i) {return bitcast<f32>(sign|mantissa);}
    if(mantissa==0x1000000u) {mantissa>>=1u;exponent++;}
    if(exponent>127i) {return bitcast<f32>(sign|0x7f800000u);}
    return bitcast<f32>(sign|(u32(exponent+127i)<<23u)|(mantissa&0x7fffffu));
}

// Exact signed sum in units of 2^-149. At most u32::MAX finite f32 terms
// occupy fewer than 310 signed bits. Caller owns nonfinite/prefix guards.
fn f32_sum_add(words: ptr<function, array<u32, 10>>, value: f32) {
    let bits = bitcast<u32>(value);
    let exponent = (bits >> 23u) & 255u;
    if (exponent == 255u) { return; }
    let mantissa = (bits & 0x7fffffu) | select(0x800000u, 0u, exponent == 0u);
    let shift = select(exponent - 1u, 0u, exponent == 0u);
    let first = shift / 32u;
    let offset = shift % 32u;
    let low = mantissa << offset;
    var high = 0u;
    if (offset != 0u) { high = mantissa >> (32u - offset); }
    let negative = (bits >> 31u) != 0u;
    var carry = 0u;
    for (var i = first; i < 10u; i = i + 1u) {
        var part = 0u;
        if (i == first) { part = low; }
        else if (i == first + 1u) { part = high; }
        let old = (*words)[i];
        var total = 0u;
        var next = 0u;
        if (negative) {
            let middle = old - part;
            total = middle - carry;
            next = select(0u, 1u, old < part || middle < carry);
        } else {
            let middle = old + part;
            total = middle + carry;
            next = select(0u, 1u, middle < old || total < middle);
        }
        (*words)[i] = total;
        carry = next;
        if (i > first && carry == 0u) { break; }
    }
}

fn f32_sum_round(words: ptr<function, array<u32, 10>>) -> f32 {
    var magnitude = *words;
    let sign = magnitude[9] & 0x80000000u;
    if (sign != 0u) {
        var carry = 1u;
        for (var i = 0u; i < 10u; i = i + 1u) {
            let inverted = ~magnitude[i];
            let total = inverted + carry;
            carry = select(0u, 1u, total < inverted);
            magnitude[i] = total;
        }
    }
    var last = 9u;
    loop {
        if (magnitude[last] != 0u) { break; }
        if (last == 0u) { return 0.0; }
        last = last - 1u;
    }
    var top = last * 32u + 31u - countLeadingZeros(magnitude[last]);
    if (top < 23u) { return bitcast<f32>(sign | magnitude[0]); }
    if (top >= 277u) { return bitcast<f32>(sign | 0x7f800000u); }
    let shift = top - 23u;
    let first = shift / 32u;
    let offset = shift % 32u;
    var mantissa = magnitude[first] >> offset;
    if (offset != 0u) { mantissa = mantissa | (magnitude[first + 1u] << (32u - offset)); }
    if (shift != 0u) {
        let round_word = (shift - 1u) / 32u;
        let round_bit = (shift - 1u) % 32u;
        let round = ((magnitude[round_word] >> round_bit) & 1u) != 0u;
        var sticky = (magnitude[round_word] & ((1u << round_bit) - 1u)) != 0u;
        for (var i = 0u; i < round_word; i = i + 1u) {
            sticky = sticky || magnitude[i] != 0u;
        }
        if (round && (sticky || (mantissa & 1u) != 0u)) { mantissa = mantissa + 1u; }
    }
    if (mantissa == 0x1000000u) { mantissa = mantissa >> 1u; top = top + 1u; }
    let exponent = top - 22u;
    if (exponent >= 255u) { return bitcast<f32>(sign | 0x7f800000u); }
    return bitcast<f32>(sign | (exponent << 23u) | (mantissa & 0x7fffffu));
}

// Integer-defined round-to-nearest-even addition prevents floating-point
// reassociation from erasing compensated-sum residuals. Inputs are f32 bits.
fn shift_right_jam(value: u32, distance: u32) -> u32 {
    if (distance == 0u) { return value; }
    if (distance >= 32u) { return select(0u, 1u, value != 0u); }
    let lost = value & ((1u << distance) - 1u);
    return (value >> distance) | select(0u, 1u, lost != 0u);
}

fn rounded_add(left: f32, right: f32) -> f32 {
    return bitcast<f32>(rounded_add_bits(bitcast<u32>(left), bitcast<u32>(right)));
}

fn rounded_add_bits(left_bits: u32, right_bits: u32) -> u32 {
    let left_magnitude = left_bits & 0x7fffffffu;
    let right_magnitude = right_bits & 0x7fffffffu;
    let a = select(right_bits, left_bits, left_magnitude >= right_magnitude);
    let b = select(left_bits, right_bits, left_magnitude >= right_magnitude);
    let sign_a = a & 0x80000000u;
    let sign_b = b & 0x80000000u;
    let exponent_a = (a >> 23u) & 255u;
    let exponent_b = (b >> 23u) & 255u;
    let fraction_a = a & 0x7fffffu;
    let fraction_b = b & 0x7fffffu;
    if (exponent_a == 255u) {
        if (fraction_a != 0u || (exponent_b == 255u && (fraction_b != 0u || sign_a != sign_b))) {
            return 0x7fc00000u;
        }
        return a;
    }
    var exponent = max(exponent_a, 1u);
    let significand_a = (fraction_a | select(0u, 0x800000u, exponent_a != 0u)) << 3u;
    let significand_b = shift_right_jam(
        (fraction_b | select(0u, 0x800000u, exponent_b != 0u)) << 3u,
        exponent - max(exponent_b, 1u));
    var significand: u32;
    if (sign_a == sign_b) {
        significand = significand_a + significand_b;
    } else {
        significand = significand_a - significand_b;
    }
    if (significand == 0u) { return sign_a & sign_b; }
    if (significand >= 0x8000000u) {
        significand = shift_right_jam(significand, 1u);
        exponent = exponent + 1u;
    } else {
        let shift = min(countLeadingZeros(significand) - 5u, exponent - 1u);
        significand = significand << shift;
        exponent = exponent - shift;
    }
    var rounded = significand >> 3u;
    let remainder = significand & 7u;
    if (remainder > 4u || (remainder == 4u && (rounded & 1u) != 0u)) {
        rounded = rounded + 1u;
    }
    if (rounded >= 0x1000000u) {
        rounded = rounded >> 1u;
        exponent = exponent + 1u;
    }
    if (exponent >= 255u) { return sign_a | 0x7f800000u; }
    let encoded_exponent = select(exponent, 0u, rounded < 0x800000u);
    return sign_a | (encoded_exponent << 23u) | (rounded & 0x7fffffu);
}

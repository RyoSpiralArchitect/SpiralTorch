// Ordinary image statistics stay on hardware division. Extreme exponents use
// integer significands so reciprocal overflow and denormal flushing cannot
// turn a finite f32 quotient into an invalid value (or a spurious zero).
fn checked_divide(x: f32, y: f32) -> f32 {
    let xb = bitcast<u32>(x);
    let yb = bitcast<u32>(y);
    let sign = (xb ^ yb) & 0x80000000u;
    let ax = xb & 0x7fffffffu;
    let ay = yb & 0x7fffffffu;
    if (ay == 0u) { return 0.0; }
    if (ax >= 0x7f800000u || ay >= 0x7f800000u) { return 0.0; }
    if (ax == 0u) { return bitcast<f32>(sign); }
    var ex = i32(ax >> 23u) - 127;
    var ey = i32(ay >> 23u) - 127;
    if (abs(ex) < 100 && abs(ey) < 100 && abs(ex - ey) < 100) { return x / y; }
    var mx = (ax & 0x7fffffu) | 0x800000u;
    var my = (ay & 0x7fffffu) | 0x800000u;
    if (ax < 0x800000u) {
        let shift = countLeadingZeros(ax) - 8u;
        mx = ax << shift;
        ex = -126 - i32(shift);
    }
    if (ay < 0x800000u) {
        let shift = countLeadingZeros(ay) - 8u;
        my = ay << shift;
        ey = -126 - i32(shift);
    }
    var exponent = ex - ey;
    if (mx < my) { mx = mx << 1u; exponent -= 1; }
    if (exponent > 127) { return bitcast<f32>(sign | 0x7f800000u); }
    if (exponent < -150) { return bitcast<f32>(sign); }
    if (exponent == -150) { return bitcast<f32>(sign | select(0u, 1u, mx > my)); }
    let digits = u32(min(23, exponent + 149));
    var quotient = 1u;
    var remainder = mx - my;
    for (var digit = 0u; digit < digits; digit += 1u) {
        quotient = quotient << 1u;
        remainder = remainder << 1u;
        if (remainder >= my) { remainder -= my; quotient += 1u; }
    }
    let twice = remainder << 1u;
    if (twice > my || (twice == my && (quotient & 1u) != 0u)) { quotient += 1u; }
    if (exponent < -126) { return bitcast<f32>(sign | quotient); }
    if (quotient >= 0x1000000u) { quotient >>= 1u; exponent += 1; }
    if (exponent > 127) { return bitcast<f32>(sign | 0x7f800000u); }
    return bitcast<f32>(sign | (u32(exponent + 127) << 23u) | (quotient & 0x7fffffu));
}

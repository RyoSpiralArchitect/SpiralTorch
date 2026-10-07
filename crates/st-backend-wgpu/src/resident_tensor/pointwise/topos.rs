//! Topos specializes the common immutable-input pointwise/VJP executor. Its
//! gate adjoint uses the existing deterministic sum, with no row averaging.
use super::*;

pub(super) fn write_body(
    code: &mut String,
    kernel: ToposResonatorKernel,
    vjp: bool,
    residual_guard: bool,
) {
    for (name, value) in [
        ("coupling", kernel.coupling()),
        ("saturation", kernel.saturation()),
        ("porosity", kernel.porosity()),
    ] {
        writeln!(code, "    let {name} = bitcast<f32>({}u);", value.to_bits()).unwrap();
    }
    writeln!(code, "    let iterations = {}u;", kernel.iterations()).unwrap();
    code.push_str(
        r#"
    let drive = checked_apply(OP_MULTIPLY, value0, value1);
    var state = 0.0;
    var sensitivity = 0.0;
    for (var step = 0u; step < iterations; step++) {
        let raw = checked_apply(OP_ADD, drive, checked_apply(OP_MULTIPLY, coupling, state));
        let magnitude = abs(raw);
        var rewritten = raw;
        var slope = 1.0;
        if (magnitude > saturation) {
            rewritten = sign(raw) * saturation;
            slope = 0.0;
            if (porosity > 1.1920928955078125e-7) {
                let relative = checked_apply(OP_DIVIDE, saturation, magnitude);
                let bleed = checked_apply(OP_DIVIDE, 1.0 - relative, 1.0 + relative);
                let absorb = porosity * 0.25;
                rewritten = sign(raw) * saturation * max(1.0 - absorb * min(bleed, 1.0), 0.0);
                // Equivalent ratio without forming magnitude+saturation, which
                // could overflow for finite operands (the CPU reference uses f64).
                let ratio = checked_apply(OP_DIVIDE, relative, 1.0 + relative);
                slope = -2.0 * absorb * ratio * ratio;
            }
        }
        check(rewritten); check(slope);
        state = rewritten;
        sensitivity = checked_apply(OP_MULTIPLY, slope,
            checked_apply(OP_ADD, 1.0, checked_apply(OP_MULTIPLY, coupling, sensitivity)));
    }
"#,
    );
    if residual_guard {
        code.push_str("    let residual_drive = checked_apply(OP_ADD, drive, checked_apply(OP_MULTIPLY, coupling, state)); check(residual_drive);\n");
    }
    if vjp {
        code.push_str(
            r#"
    let grad_drive = checked_apply(OP_MULTIPLY, cotangent[i], sensitivity);
    out[i] = checked_apply(OP_MULTIPLY, grad_drive, value1);
    out[params[0] + i] = checked_apply(OP_MULTIPLY, grad_drive, value0);
}
"#,
        );
    } else {
        code.push_str("    out[i] = state;\n}\n");
    }
}

#[cfg(test)]
mod tests;

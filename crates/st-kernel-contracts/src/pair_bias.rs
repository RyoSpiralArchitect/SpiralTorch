// SPDX-License-Identifier: AGPL-3.0-or-later
pub(crate) fn softplus_gain(raw: f32) -> (f64, f64) {
    let r = f64::from(raw);
    let e = (-r.abs()).exp();
    (
        r.max(0.) + e.ln_1p(),
        if r >= 0. { 1. / (1. + e) } else { e / (1. + e) },
    )
}

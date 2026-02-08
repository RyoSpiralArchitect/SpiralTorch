// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::device_ops::get_ops;

// Minimal CG solver: A*x via ops::matvec; uses fused ops if installed.
pub fn cg_solve_device(
    a: &[f32],
    m: usize,
    n: usize,
    b: &[f32],
    x: &mut [f32],
    iters: usize,
    tol: f32,
) {
    if m == 0 || n == 0 || b.len() < m || x.len() < n {
        return;
    }

    let ops = get_ops();
    let mut r = vec![0.0f32; m];
    // r = b - A*x
    let mut tmp = vec![0.0f32; m];
    ops.matvec(m, n, a, x.as_ptr(), tmp.as_mut_ptr());
    for i in 0..m {
        r[i] = b[i] - tmp[i];
    }
    let mut p = r.clone();
    let mut rs_old = ops.dot(m, r.as_ptr(), r.as_ptr());
    if rs_old <= tol * tol {
        return;
    }
    const EPS: f32 = 1.0e-30;
    for _ in 0..iters {
        ops.matvec(m, n, a, p.as_ptr(), tmp.as_mut_ptr());
        let denom = ops.dot(m, p.as_ptr(), tmp.as_ptr());
        if !denom.is_finite() || denom.abs() <= EPS {
            break;
        }
        let alpha = rs_old / denom;
        for i in 0..m {
            x[i] += alpha * p[i];
            r[i] -= alpha * tmp[i];
        }
        let rs_new = ops.dot(m, r.as_ptr(), r.as_ptr());
        if !rs_new.is_finite() || rs_new.sqrt() < tol {
            break;
        }
        let beta = rs_new / rs_old.max(EPS);
        for i in 0..m {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

use super::device_ops::get_ops;

// Minimal CG stub: A*x via ops::matvec; uses fused ops if installed.
pub fn cg_solve_device(a:&[f32], m:usize, n:usize, b:&[f32], x:&mut [f32], iters:usize, tol:f32){
    let ops = get_ops();
    let mut r = vec![0f32; m];
    // r = b - A*x
    let mut tmp = vec![0f32; m];
    ops.matvec(m,n,a,x.as_ptr(),tmp.as_mut_ptr());
    for i in 0..m{ r[i] = b[i]-tmp[i]; }
    let mut p = r.clone();
    let mut rs_old = ops.dot(m, r.as_ptr(), r.as_ptr());
    for _ in 0..iters {
        ops.matvec(m,n,a,p.as_ptr(),tmp.as_mut_ptr());
        let alpha = rs_old / ops.dot(m, p.as_ptr(), tmp.as_ptr());
        for i in 0..m { x[i] += alpha * p[i]; r[i] -= alpha * tmp[i]; }
        let rs_new = ops.dot(m, r.as_ptr(), r.as_ptr());
        if rs_new.sqrt() < tol { break; }
        let beta = rs_new / rs_old;
        for i in 0..m { p[i] = r[i] + beta * p[i]; }
        rs_old = rs_new;
    }
}

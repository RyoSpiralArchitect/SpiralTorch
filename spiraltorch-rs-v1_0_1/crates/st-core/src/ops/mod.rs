
use ndarray::{ArrayD, IxDyn, Axis, Zip};
use crate::{Tensor, autograd::{BackwardNode, GradFn}, error::{Result, shape}};
use std::collections::{HashMap, HashSet};

// ---------- helpers ----------
fn bcast_to(x: &ArrayD<f32>, out_shape: &[usize]) -> Result<ArrayD<f32>> {
    x.broadcast(IxDyn(out_shape)).ok_or_else(|| shape("broadcast: incompatible shape"))?
     .to_owned().into_dimensionality::<IxDyn>().map_err(|_| shape("broadcast: into_dimensionality failed"))
}

// ---------- basic op used in tests ----------
pub fn sum(x: &Tensor) -> Result<Tensor> {
    let xi = x.data(); let s = xi.sum();
    let out = Tensor::from_array(ndarray::arr0(s).into_dyn());
    if x.0.borrow().requires_grad {
        struct Node { x: Tensor }
        impl BackwardNode for Node {
            fn name(&self) -> &'static str { "sum" }
            fn parents(&self) -> Vec<Tensor> { vec![self.x.clone()] }
            fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                let go = grads_out[0].as_ref().unwrap().sum();
                let ones = ndarray::ArrayD::<f32>::from_elem(IxDyn(&self.x.shape()), 1.0);
                vec![Some(ones * go)]
            }
        } let gf = GradFn::new(Node { x: x.clone() }); out.attach_grad_fn(gf, 0, 1, true);
    } Ok(out)
}

// ---------- index_reduce (sum/mean/min/max/amin/amax/prod exact grad) ----------
pub fn index_reduce(base: &Tensor, dim: isize, index1d: &Tensor, src: &Tensor, reduce: &str, include_self: bool) -> Result<Tensor> {
    let bv = base.data(); let idx = index1d.data_i32(); let sv = src.data();
    if idx.ndim() != 1 { return Err(shape("index_reduce: index must be 1-D")); }
    if bv.ndim() != sv.ndim() { return Err(shape("index_reduce: src.ndim must equal base.ndim")); }
    let nd = bv.ndim() as isize; let mut d = dim; if d < 0 { d += nd; }
    if d < 0 || (d as usize) >= bv.ndim() { return Err(shape("index_reduce: dim out of range")); }
    let du = d as usize;
    for ax in 0..bv.ndim() {
        let want = if ax == du { idx.len() } else { bv.shape()[ax] };
        if sv.shape()[ax] != want { return Err(shape(format!("index_reduce: src shape mismatch at axis {} (got {}, want {})", ax, sv.shape()[ax], want))); }
    }

    let mut out = bv.clone();
    let rank = bv.ndim();
    let mut perm: Vec<usize> = (0..rank).collect();
    perm.remove(du); perm.insert(0, du);
    let mut outp = out.view_mut().permuted_axes(perm.clone());
    let basep = bv.view().permuted_axes(perm.clone());
    let srcp = sv.view().permuted_axes(perm.clone());

    match reduce {
        "sum" => {
            for j in 0..idx.len() {
                let tgt = idx[j]; if tgt < 0 || (tgt as usize) >= outp.shape()[0] { return Err(shape("index_reduce: index out of range")); }
                let mut dest = outp.index_axis_move(Axis(0), tgt as usize);
                let srcs = srcp.index_axis_move(Axis(0), j);
                Zip::from(&mut dest).and(&srcs).apply(|o, &s| { *o = *o + s; });
            }
        }
        "mean" => {
            let mut count = ndarray::ArrayD::<f32>::zeros(IxDyn(outp.raw_dim().slice()));
            if include_self { count.fill(1.0); }
            let mut sum = outp.to_owned(); if !include_self { sum.fill(0.0); }
            for j in 0..idx.len() {
                let tgt = idx[j]; if tgt < 0 || (tgt as usize) >= outp.shape()[0] { return Err(shape("index_reduce: index out of range")); }
                let mut sumd = sum.index_axis_move(Axis(0), tgt as usize);
                let srcs = srcp.index_axis_move(Axis(0), j);
                Zip::from(&mut sumd).and(&srcs).apply(|o, &s| { *o += s; });
                let mut cntd = count.index_axis_move(Axis(0), tgt as usize);
                cntd += 1.0;
            }
            for i in 0..outp.shape()[0] {
                let s = sum.index_axis_move(Axis(0), i);
                let c = count.index_axis_move(Axis(0), i);
                let mut o = outp.index_axis_move(Axis(0), i);
                Zip::from(&mut o).and(&s).and(&c).apply(|o, &sv, &cv| {
                    if cv > 0.0 { *o = sv / cv; }
                });
            }
        }
        "min" | "max" | "amin" | "amax" => {
            let pick_min = reduce == "min" || reduce == "amin";
            let mut acc = if include_self { outp.to_owned() } else {
                if pick_min { ndarray::ArrayD::<f32>::from_elem(outp.raw_dim(), f32::INFINITY) }
                else { ndarray::ArrayD::<f32>::from_elem(outp.raw_dim(), f32::NEG_INFINITY) }
            };
            let mut touched = ndarray::ArrayD::<bool>::from_elem(outp.raw_dim(), include_self);
            for j in 0..idx.len() {
                let tgt = idx[j]; if tgt < 0 || (tgt as usize) >= outp.shape()[0] { return Err(shape("index_reduce: index out of range")); }
                let mut accd = acc.index_axis_move(Axis(0), tgt as usize);
                let srcs = srcp.index_axis_move(Axis(0), j);
                if pick_min { Zip::from(&mut accd).and(&srcs).apply(|a, &s| { if s < *a { *a = s; } }); }
                else { Zip::from(&mut accd).and(&srcs).apply(|a, &s| { if s > *a { *a = s; } }); }
                let mut tmask = touched.index_axis_move(Axis(0), tgt as usize);
                tmask.fill(true);
            }
            for i in 0..outp.shape()[0] {
                let accd = acc.index_axis_move(Axis(0), i);
                let tmask = touched.index_axis_move(Axis(0), i);
                let mut o = outp.index_axis_move(Axis(0), i);
                Zip::from(&mut o).and(&accd).and(&tmask).apply(|o, &a, &t| { if t { *o = a; } });
            }
        }
        "prod" => {
            // Stable product with zero-aware handling
            let mut prod = if include_self { outp.to_owned() } else { ndarray::ArrayD::<f32>::from_elem(IxDyn(outp.raw_dim().slice()), 1.0) };
            let mut zero_count = ndarray::ArrayD::<i32>::from_elem(IxDyn(outp.raw_dim().slice()), 0);
            if include_self {
                let base_is_zero = basep.map(|&v| (v == 0.0) as i32);
                Zip::from(&mut zero_count).and(&base_is_zero).apply(|z, &b| { *z += b; });
            }
            for j in 0..idx.len() {
                let tgt = idx[j]; if tgt < 0 || (tgt as usize) >= outp.shape()[0] { return Err(shape("index_reduce: index out of range")); }
                let mut pd = prod.index_axis_move(Axis(0), tgt as usize);
                let srcs = srcp.index_axis_move(Axis(0), j);
                Zip::from(&mut pd).and(&srcs).apply(|p, &s| { *p *= s; });
                let mut z = zero_count.index_axis_move(Axis(0), tgt as usize);
                let zadd = srcs.map(|&s| if s == 0.0 { 1i32 } else { 0i32 });
                Zip::from(&mut z).and(&zadd).apply(|a, &b| { *a += b; });
            }
            for i in 0..outp.shape()[0] {
                let z = zero_count.index_axis_move(Axis(0), i);
                let p = prod.index_axis_move(Axis(0), i);
                let mut o = outp.index_axis_move(Axis(0), i);
                Zip::from(&mut o).and(&p).and(&z).apply(|o, &pv, &zv| { *o = if zv >= 1 { 0.0 } else { pv }; });
            }
        }
        _ => return Err(shape("index_reduce: reduce must be 'sum'|'mean'|'min'|'max'|'amin'|'amax'|'prod'"))
    }

    let out_t = Tensor::from_array(out.into_dyn());
    if base.0.borrow().requires_grad || src.0.borrow().requires_grad {
        struct Node { base: Tensor, index1d: Tensor, src: Tensor, dim: usize, reduce: String, include_self: bool }
        impl BackwardNode for Node {
            fn name(&self) -> &'static str { "index_reduce" }
            fn parents(&self) -> Vec<Tensor> { vec![self.base.clone(), self.src.clone()] }
            fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                let go = grads_out[0].as_ref().unwrap().clone();
                let bv = self.base.data();
                let sv = self.src.data();
                let idx = self.index1d.data_i32();
                let rank = bv.ndim();
                let mut perm: Vec<usize> = (0..rank).collect();
                perm.remove(self.dim); perm.insert(0, self.dim);
                let gop = go.view().permuted_axes(perm.clone()).to_owned();
                let basep = bv.view().permuted_axes(perm.clone());
                let srcp = sv.view().permuted_axes(perm.clone());

                let mut gb = gop.clone(); // default passthrough
                let mut gs = ndarray::ArrayD::<f32>::zeros(IxDyn(&sv.shape().to_vec()));

                match self.reduce.as_str() {
                    "sum" => {
                        for j in 0..idx.len() {
                            let tgt = idx[j] as usize;
                            let gol = gop.index_axis_move(Axis(0), tgt);
                            let mut gsl = gs.view_mut().permuted_axes(perm.clone()).index_axis_move(Axis(0), j);
                            gsl.assign(&gol);
                        }
                    }
                    "mean" => {
                        let mut count = ndarray::ArrayD::<f32>::zeros(IxDyn(gop.raw_dim().slice()));
                        if self.include_self { count.fill(1.0); }
                        for j in 0..idx.len() {
                            let tgt = idx[j] as usize;
                            let mut cntd = count.index_axis_move(Axis(0), tgt);
                            cntd += 1.0;
                        }
                        if self.include_self {
                            Zip::from(&mut gb).and(&count).apply(|g, &c| { *g = if c > 0.0 { *g / c } else { *g }; });
                        } else {
                            Zip::from(&mut gb).and(&count).apply(|g, &c| { *g = if c > 0.0 { 0.0 } else { *g }; });
                        }
                        let mut gs_p = gs.view_mut().permuted_axes(perm.clone());
                        for j in 0..idx.len() {
                            let tgt = idx[j] as usize;
                            let mut dst = gs_p.index_axis_move(Axis(0), j);
                            let gol = gop.index_axis_move(Axis(0), tgt);
                            let cnt = count.index_axis_move(Axis(0), tgt);
                            Zip::from(&mut dst).and(&gol).and(&cnt).apply(|o, &g, &c| { *o = if c > 0.0 { g / c } else { 0.0 }; });
                        }
                    }
                    "min" | "max" | "amin" | "amax" => {
                        let pick_min = self.reduce.as_str() == "min" || self.reduce.as_str() == "amin";
                        let mut best = if self.include_self { basep.to_owned() } else {
                            if pick_min { ndarray::ArrayD::<f32>::from_elem(IxDyn(gop.raw_dim().slice()), f32::INFINITY) }
                            else { ndarray::ArrayD::<f32>::from_elem(IxDyn(gop.raw_dim().slice()), f32::NEG_INFINITY) }
                        };
                        let mut touched = ndarray::ArrayD::<bool>::from_elem(IxDyn(gop.raw_dim().slice()), self.include_self);
                        for j in 0..idx.len() {
                            let tgt = idx[j] as usize;
                            let srcj = srcp.index_axis_move(Axis(0), j);
                            let mut bj = best.index_axis_move(Axis(0), tgt);
                            if pick_min { Zip::from(&mut bj).and(&srcj).apply(|b, &s| { if s < *b { *b = s; } }); }
                            else { Zip::from(&mut bj).and(&srcj).apply(|b, &s| { if s > *b { *b = s; } }); }
                            let mut t = touched.index_axis_move(Axis(0), tgt); t.fill(true);
                        }
                        let mut ties = ndarray::ArrayD::<f32>::zeros(IxDyn(gop.raw_dim().slice()));
                        if self.include_self { Zip::from(&mut ties).and(&best).and(&basep).apply(|t, &b, &a| { if a == b { *t += 1.0; } }); }
                        for j in 0..idx.len() {
                            let tgt = idx[j] as usize;
                            let srcj = srcp.index_axis_move(Axis(0), j);
                            let bj = best.index_axis_move(Axis(0), tgt);
                            let mut tj = ties.index_axis_move(Axis(0), tgt);
                            Zip::from(&mut tj).and(&srcj).and(&bj).apply(|t, &s, &b| { if s == b { *t += 1.0; } });
                        }
                        if self.include_self {
                            let mut gbp = gb.view_mut();
                            Zip::from(&mut gbp).and(&best).and(&basep).and(&ties).apply(|g, &b, &a, &t| {
                                if a == b && t > 0.0 { *g = *g / t; } else { *g = 0.0; }
                            });
                        } else {
                            let mut gbp = gb.view_mut();
                            Zip::from(&mut gbp).and(&touched).apply(|g, &t| { if t { *g = 0.0; } });
                        }
                        let mut gs_p = gs.view_mut().permuted_axes(perm.clone());
                        for j in 0..idx.len() {
                            let tgt = idx[j] as usize;
                            let srcj = srcp.index_axis_move(Axis(0), j);
                            let bj = best.index_axis_move(Axis(0), tgt);
                            let tj = ties.index_axis_move(Axis(0), tgt);
                            let gol = gop.index_axis_move(Axis(0), tgt);
                            let mut dst = gs_p.index_axis_move(Axis(0), j);
                            Zip::from(&mut dst).and(&srcj).and(&bj).and(&tj).and(&gol).apply(|o, &s, &b, &t, &g| {
                                if s == b && t > 0.0 { *o = g / t; } else { *o = 0.0; }
                            });
                        }
                    }
                    "prod" => {
                        // exact gradient handling with zero_count & product of non-zero factors
                        // Precompute per-dest: (zero_count_total, base_zero?, prod_src_nonzero)
                        let mut zeros_total = ndarray::ArrayD::<i32>::from_elem(IxDyn(gop.raw_dim().slice()), 0);
                        let mut base_zero = ndarray::ArrayD::<bool>::from_elem(IxDyn(gop.raw_dim().slice()), false);
                        if self.include_self {
                            Zip::from(&mut base_zero).and(&basep).apply(|bz, &a| { *bz = a == 0.0; });
                            let bz_i32 = base_zero.map(|&b| if b { 1i32 } else { 0i32 });
                            Zip::from(&mut zeros_total).and(&bz_i32).apply(|z, &b| { *z += b; });
                        }
                        // prod of non-zero src per dest
                        let mut prod_src_nz = ndarray::ArrayD::<f32>::from_elem(IxDyn(gop.raw_dim().slice()), 1.0);
                        for j in 0..idx.len() {
                            let tgt = idx[j] as usize;
                            let srcj = srcp.index_axis_move(Axis(0), j);
                            let mut p = prod_src_nz.index_axis_move(Axis(0), tgt);
                            let mut z = zeros_total.index_axis_move(Axis(0), tgt);
                            Zip::from(&mut p).and(&srcj).apply(|pp, &s| { if s != 0.0 { *pp *= s; } });
                            let zadd = srcj.map(|&s| if s == 0.0 { 1i32 } else { 0i32 });
                            Zip::from(&mut z).and(&zadd).apply(|a, &b| { *a += b; });
                        }
                        // gb: base grad
                        if self.include_self {
                            let mut gbp = gb.view_mut();
                            Zip::from(&mut gbp).and(&zeros_total).and(&base_zero).and(&prod_src_nz).and(&basep).apply(|g, &z, &bz, &ps, &a| {
                                if z >= 2 { *g = 0.0; }
                                else if z == 1 {
                                    if bz { *g = *g * ps; } else { *g = 0.0; }
                                } else { // z==0
                                    *g = *g * ps;
                                }
                            });
                        } else {
                            let mut touched = ndarray::ArrayD::<bool>::from_elem(IxDyn(gop.raw_dim().slice()), false);
                            for j in 0..idx.len() { let tgt = idx[j] as usize; let mut t = touched.index_axis_move(Axis(0), tgt); t.fill(true); }
                            let mut gbp = gb.view_mut();
                            Zip::from(&mut gbp).and(&touched).apply(|g, &t| { if t { *g = 0.0; } });
                        }
                        // gs: src grad
                        let mut gs_p = gs.view_mut().permuted_axes(perm.clone());
                        for j in 0..idx.len() {
                            let tgt = idx[j] as usize;
                            let srcj = srcp.index_axis_move(Axis(0), j);
                            let bz = base_zero.index_axis_move(Axis(0), tgt);
                            let zt = zeros_total.index_axis_move(Axis(0), tgt);
                            let ps = prod_src_nz.index_axis_move(Axis(0), tgt);
                            let a = basep.index_axis_move(Axis(0), tgt);
                            let gol = gop.index_axis_move(Axis(0), tgt);
                            let mut dst = gs_p.index_axis_move(Axis(0), j);
                            Zip::from(&mut dst).and(&srcj).and(&bz).and(&zt).and(&ps).and(&a).and(&gol).apply(|o, &s, &bzv, &zv, &psv, &av, &g| {
                                if zv >= 2 { *o = 0.0; }
                                else if zv == 1 {
                                    if s == 0.0 && !bzv { // unique zero is this src
                                        let factor = if av == 0.0 { 1.0 } else { av };
                                        *o = g * factor * psv;
                                    } else {
                                        *o = 0.0;
                                    }
                                } else { // zv == 0
                                    if s != 0.0 {
                                        let factor = if self.include_self { av } else { 1.0 };
                                        *o = g * factor * (psv / s);
                                    } else { *o = 0.0; }
                                }
                            });
                        }
                    }
                    _ => unreachable!(),
                }

                vec![Some(crate::tensor::unbroadcast(gb.into_dyn(), &self.base.shape())), Some(gs)]
            }
        }
        let gf = GradFn::new(Node { base: base.clone(), index1d: index1d.clone(), src: src.clone(), dim: du, reduce: reduce.to_string(), include_self });
        out_t.attach_grad_fn(gf, 0, 1, true);
    }
    Ok(out_t)
}

// ---------- segment_* wrappers ----------
pub fn segment_sum(data: &Tensor, indices1d: &Tensor, num_segments: usize) -> Result<Tensor> {
    if indices1d.data_i32().ndim() != 1 { return Err(shape("segment_sum: indices must be 1-D")); }
    let mut out_shape = data.shape(); out_shape[0] = num_segments;
    let base = Tensor::from_array(ndarray::ArrayD::<f32>::zeros(IxDyn(&out_shape)));
    index_reduce(&base, 0, indices1d, data, "sum", false)
}
pub fn segment_mean(data: &Tensor, indices1d: &Tensor, num_segments: usize) -> Result<Tensor> {
    if indices1d.data_i32().ndim() != 1 { return Err(shape("segment_mean: indices must be 1-D")); }
    let mut out_shape = data.shape(); out_shape[0] = num_segments;
    let base = Tensor::from_array(ndarray::ArrayD::<f32>::zeros(IxDyn(&out_shape)));
    index_reduce(&base, 0, indices1d, data, "mean", false)
}
pub fn segment_max(data: &Tensor, indices1d: &Tensor, num_segments: usize) -> Result<Tensor> {
    if indices1d.data_i32().ndim() != 1 { return Err(shape("segment_max: indices must be 1-D")); }
    let mut out_shape = data.shape(); out_shape[0] = num_segments;
    let base = Tensor::from_array(ndarray::ArrayD::<f32>::from_elem(IxDyn(&out_shape), f32::NEG_INFINITY));
    index_reduce(&base, 0, indices1d, data, "max", true)
}
pub fn segment_min(data: &Tensor, indices1d: &Tensor, num_segments: usize) -> Result<Tensor> {
    if indices1d.data_i32().ndim() != 1 { return Err(shape("segment_min: indices must be 1-D")); }
    let mut out_shape = data.shape(); out_shape[0] = num_segments;
    let base = Tensor::from_array(ndarray::ArrayD::<f32>::from_elem(IxDyn(&out_shape), f32::INFINITY));
    index_reduce(&base, 0, indices1d, data, "min", true)
}

// ---------- coalesce & ragged ----------
pub fn coalesce_indices(indices1d: &Tensor) -> Result<(Tensor, Tensor, usize)> {
    let idx = indices1d.data_i32();
    if idx.ndim() != 1 { return Err(shape("coalesce_indices: indices must be 1-D")); }
    let mut uniq: Vec<i32> = Vec::new();
    let mut remap: Vec<i32> = Vec::with_capacity(idx.len());
    for j in 0..idx.len() {
        let v = idx[ndarray::IxDyn(&[j])];
        if let Some(pos) = uniq.iter().position(|&u| u==v) {
            remap.push(pos as i32);
        } else {
            uniq.push(v);
            remap.push((uniq.len()-1) as i32);
        }
    }
    let unique_t = Tensor::from_i32(ndarray::Array1::from_vec(uniq).into_dyn());
    let remap_t = Tensor::from_i32(ndarray::Array1::from_vec(remap).into_dyn());
    Ok((unique_t, remap_t, unique_t.shape()[0]))
}

pub fn ragged_segment_sum(data: &Tensor, row_splits: &Tensor) -> Result<Tensor> {
    let rs = row_splits.data_i32();
    if rs.ndim()!=1 { return Err(shape("ragged_segment_sum: row_splits must be 1-D")); }
    let k = rs.len()-1;
    let n = *rs.last().unwrap() as usize;
    let mut idx = Vec::<i32>::with_capacity(n);
    for i in 0..k {
        let a = rs[ndarray::IxDyn(&[i])] as usize;
        let b = rs[ndarray::IxDyn(&[i+1])] as usize;
        for _ in a..b { idx.push(i as i32); }
    }
    let idx_t = Tensor::from_i32(ndarray::Array1::from_vec(idx).into_dyn());
    segment_sum(data, &idx_t, k)
}
pub fn ragged_segment_mean(data: &Tensor, row_splits: &Tensor) -> Result<Tensor> {
    let rs = row_splits.data_i32();
    if rs.ndim()!=1 { return Err(shape("ragged_segment_mean: row_splits must be 1-D")); }
    let k = rs.len()-1;
    let n = *rs.last().unwrap() as usize;
    let mut idx = Vec::<i32>::with_capacity(n);
    for i in 0..k {
        let a = rs[ndarray::IxDyn(&[i])] as usize;
        let b = rs[ndarray::IxDyn(&[i+1])] as usize;
        for _ in a..b { idx.push(i as i32); }
    }
    let idx_t = Tensor::from_i32(ndarray::Array1::from_vec(idx).into_dyn());
    segment_mean(data, &idx_t, k)
}
pub fn ragged_segment_max(data: &Tensor, row_splits: &Tensor) -> Result<Tensor> {
    let rs = row_splits.data_i32();
    if rs.ndim()!=1 { return Err(shape("ragged_segment_max: row_splits must be 1-D")); }
    let k = rs.len()-1;
    let n = *rs.last().unwrap() as usize;
    let mut idx = Vec::<i32>::with_capacity(n);
    for i in 0..k {
        let a = rs[ndarray::IxDyn(&[i])] as usize;
        let b = rs[ndarray::IxDyn(&[i+1])] as usize;
        for _ in a..b { idx.push(i as i32); }
    }
    let idx_t = Tensor::from_i32(ndarray::Array1::from_vec(idx).into_dyn());
    segment_max(data, &idx_t, k)
}
pub fn ragged_segment_min(data: &Tensor, row_splits: &Tensor) -> Result<Tensor> {
    let rs = row_splits.data_i32();
    if rs.ndim()!=1 { return Err(shape("ragged_segment_min: row_splits must be 1-D")); }
    let k = rs.len()-1;
    let n = *rs.last().unwrap() as usize;
    let mut idx = Vec::<i32>::with_capacity(n);
    for i in 0..k {
        let a = rs[ndarray::IxDyn(&[i])] as usize;
        let b = rs[ndarray::IxDyn(&[i+1])] as usize;
        for _ in a..b { idx.push(i as i32); }
    }
    let idx_t = Tensor::from_i32(ndarray::Array1::from_vec(idx).into_dyn());
    segment_min(data, &idx_t, k)
}

// ---------- einsum helpers ----------
fn parse_einsum(spec: &str) -> Result<(Vec<Vec<char>>, Vec<char>)> {
    let parts: Vec<&str> = spec.split("->").collect();
    let inputs_part = parts[0];
    let inputs: Vec<Vec<char>> = inputs_part.split(',').map(|s| s.trim().chars().collect()).collect();
    let output: Vec<char> = if parts.len() == 2 { parts[1].trim().chars().collect() } else { Vec::new() };
    if inputs.iter().any(|v| v.iter().any(|&c| c == '.')) { return Err(shape("einsum: ellipsis not supported")); }
    Ok((inputs, output))
}

fn dims_from_labels(labels_in: &Vec<Vec<char>>, shapes: &Vec<Vec<usize>>) -> Result<HashMap<char, usize>> {
    let mut dims: HashMap<char, usize> = HashMap::new();
    for (ls, sh) in labels_in.iter().zip(shapes.iter()) {
        if ls.len() != sh.len() { return Err(shape("einsum: operand rank mismatch")); }
        for (lbl, &d) in ls.iter().zip(sh.iter()) {
            if let Some(&old) = dims.get(lbl) {
                if old == d || d == 1 { /* ok */ }
                else if old == 1 { dims.insert(*lbl, d); }
                else { return Err(shape(format!("einsum: incompatible dim for label {} ({} vs {})", lbl, old, d))); }
            } else {
                dims.insert(*lbl, d);
            }
        }
    }
    Ok(dims)
}

fn align_to_labels(arr: ArrayD<f32>, in_labels: &[char], target_labels: &[char], dims: &HashMap<char, usize>) -> ArrayD<f32> {
    let mut out = arr;
    let mut current = in_labels.to_vec();
    for (pos, &lbl) in target_labels.iter().enumerate() {
        if let Some(idx) = current.iter().position(|&c| c==lbl) {
            if idx != pos {
                let mut axes: Vec<usize> = (0..current.len()).collect();
                let mut new_order = axes.clone(); new_order.remove(idx); new_order.insert(pos, idx);
                out = out.view().permuted_axes(new_order.clone()).to_owned();
                let c = current.remove(idx); current.insert(pos, c);
            }
        } else {
            out = out.insert_axis(Axis(pos));
            current.insert(pos, lbl);
        }
    }
    out.broadcast(IxDyn(&target_labels.iter().map(|c| *dims.get(c).unwrap()).collect::<Vec<_>>())).unwrap().to_owned()
}

// ---------- einsum (DP generalization + greedy fallback) ----------
pub fn einsum_opt(spec: &str, tensors: &[Tensor], optimize: bool) -> Result<Tensor> {
    let (labels_in, mut labels_out) = parse_einsum(spec)?;
    if labels_in.len() != tensors.len() { return Err(shape("einsum: number of operands mismatch")); }

    let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape()).collect();
    let dims = dims_from_labels(&labels_in, &shapes)?;

    if labels_out.is_empty() {
        let mut counts: HashMap<char, usize> = HashMap::new();
        for ls in &labels_in { for &c in ls { *counts.entry(c).or_insert(0) += 1; } }
        let mut all: Vec<char> = Vec::new();
        for ls in &labels_in { for &c in ls { if !all.contains(&c) { all.push(c); } } }
        labels_out = all.into_iter().filter(|c| counts.get(c)==Some(&1)).collect();
    }

    if !optimize || tensors.len() <= 1 {
        let mut all: Vec<char> = Vec::new();
        for ls in &labels_in { for &c in ls { if !all.contains(&c) { all.push(c); } } }
        let aligned: Vec<ArrayD<f32>> = tensors.iter().enumerate()
            .map(|(i,t)| align_to_labels(t.data(), &labels_in[i], &all, &dims)).collect();
        let full_shape: Vec<usize> = all.iter().map(|c| *dims.get(c).unwrap()).collect();
        let mut prod = ndarray::ArrayD::<f32>::from_elem(IxDyn(&full_shape), 1.0);
        for a in &aligned { Zip::from(&mut prod).and(a).apply(|p, &v| { *p *= v; }); }
        let mut y = prod;
        for (ax, &lbl) in all.iter().enumerate().rev() { if !labels_out.contains(&lbl) { y = y.sum_axis(Axis(ax)); } }
        let remain: Vec<char> = all.into_iter().filter(|c| labels_out.contains(c)).collect();
        let mut order: Vec<usize> = Vec::new();
        for &c in &labels_out { let pos = remain.iter().position(|&r| r==c).unwrap(); order.push(pos); }
        if !order.is_empty() && (0..order.len()).any(|i| order[i]!=i) { y = y.view().permuted_axes(order).to_owned(); }
        let out_t = Tensor::from_array(y.into_dyn());
        if tensors.iter().any(|t| t.0.borrow().requires_grad) {
            struct Node {
                inputs: Vec<Tensor>,
                labels_in: Vec<Vec<char>>,
                labels_out: Vec<char>,
                dims: HashMap<char, usize>,
            }
            impl BackwardNode for Node {
                fn name(&self) -> &'static str { "einsum" }
                fn parents(&self) -> Vec<Tensor> { self.inputs.clone() }
                fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                    let go = grads_out[0].as_ref().unwrap().clone();
                    let mut all: Vec<char> = Vec::new();
                    for ls in &self.labels_in { for &c in ls { if !all.contains(&c) { all.push(c); } } }
                    let full_shape: Vec<usize> = all.iter().map(|c| *self.dims.get(c).unwrap()).collect();
                    let mut g = go;
                    let mut cur = self.labels_out.clone();
                    for (pos, &lbl) in all.iter().enumerate() {
                        if let Some(idx) = cur.iter().position(|&c| c==lbl) {
                            if idx != pos {
                                let mut axes: Vec<usize> = (0..cur.len()).collect();
                                let mut new_order = axes.clone(); new_order.remove(idx); new_order.insert(pos, idx);
                                g = g.view().permuted_axes(new_order).to_owned();
                                let c = cur.remove(idx); cur.insert(pos, c);
                            }
                        } else {
                            g = g.insert_axis(Axis(pos));
                            cur.insert(pos, lbl);
                        }
                    }
                    g = g.broadcast(IxDyn(&full_shape)).unwrap().to_owned();
                    fn align(arr: ArrayD<f32>, labels: &[char], all: &[char], dims: &HashMap<char, usize>) -> ArrayD<f32> {
                        let mut out = arr;
                        let mut cur = labels.to_vec();
                        for (pos, &lbl) in all.iter().enumerate() {
                            if let Some(idx) = cur.iter().position(|&c| c==lbl) {
                                if idx != pos {
                                    let mut axes: Vec<usize> = (0..cur.len()).collect();
                                    let mut new_order = axes.clone(); new_order.remove(idx); new_order.insert(pos, idx);
                                    out = out.view().permuted_axes(new_order).to_owned();
                                    let c = cur.remove(idx); cur.insert(pos, c);
                                }
                            } else { out = out.insert_axis(Axis(pos)); cur.insert(pos, lbl); }
                        }
                        out.broadcast(IxDyn(&all.iter().map(|c| *dims.get(c).unwrap()).collect::<Vec<_>>())).unwrap().to_owned()
                    }
                    let aligned: Vec<ArrayD<f32>> = self.inputs.iter().enumerate()
                        .map(|(i,t)| align(t.data(), &self.labels_in[i], &all, &self.dims)).collect();
                    let mut grads: Vec<Option<ArrayD<f32>>> = Vec::new();
                    for i in 0..self.inputs.len() {
                        let mut prod_except = ndarray::ArrayD::<f32>::from_elem(IxDyn(&full_shape), 1.0);
                        for j in 0..self.inputs.len() { if i==j { continue; } Zip::from(&mut prod_except).and(&aligned[j]).apply(|p,&v|{*p*=v;}); }
                        let mut gi = &g * &prod_except;
                        let keep: Vec<bool> = all.iter().map(|c| self.labels_in[i].contains(c)).collect();
                        for (ax, &k) in keep.iter().enumerate().rev() { if !k { gi = gi.sum_axis(Axis(ax)); } }
                        let remain: Vec<char> = all.iter().cloned().filter(|c| self.labels_in[i].contains(c)).collect();
                        let mut order: Vec<usize> = Vec::new();
                        for &c in &self.labels_in[i] { let pos = remain.iter().position(|&r| r==c).unwrap(); order.push(pos); }
                        if !order.is_empty() && (0..order.len()).any(|k| order[k]!=k) { gi = gi.view().permuted_axes(order).to_owned(); }
                        grads.push(Some(gi.into_dyn()));
                    }
                    grads
                }
            }
            let gf = GradFn::new(Node { inputs: tensors.to_vec(), labels_in: labels_in.clone(), labels_out: labels_out.clone(), dims });
            out_t.attach_grad_fn(gf, 0, 1, true);
        }
        return Ok(out_t);
    }

    // DP generalization over linear sequence
    let n = tensors.len();
    let mut pos_list: HashMap<char, Vec<usize>> = HashMap::new();
    for (i, ls) in labels_in.iter().enumerate() {
        for &c in ls {
            pos_list.entry(c).or_insert_with(Vec::new).push(i);
        }
    }
    let mut counts_prefix: HashMap<char, Vec<usize>> = HashMap::new();
    for (&lbl, positions) in &pos_list {
        let mut pref = vec![0usize; n+1];
        for &p in positions { pref[p+1] += 1; }
        for i in 0..n { pref[i+1] += pref[i]; }
        counts_prefix.insert(lbl, pref);
    }
    let count_in_range = |lbl: char, i: usize, j: usize| -> usize {
        let pref = counts_prefix.get(&lbl).unwrap();
        pref[j+1] - pref[i]
    };
    let appears_outside = |lbl: char, i: usize, j: usize| -> bool {
        let total = pos_list.get(&lbl).map(|v| v.len()).unwrap_or(0);
        let inside = count_in_range(lbl, i, j);
        total > inside
    };

    let mut cost = vec![vec![0u128; n]; n];
    let mut labels_tbl: Vec<Vec<Vec<char>>> = vec![vec![Vec::new(); n]; n];
    for i in 0..n { labels_tbl[i][i] = labels_in[i].clone(); }

    for len in 2..=n {
        for i in 0..=(n-len) {
            let j = i + len - 1;
            let mut best_cost = u128::MAX / 4;
            let mut best_labels: Vec<char> = Vec::new();
            for k in i..j {
                let left = &labels_tbl[i][k];
                let right = &labels_tbl[k+1][j];
                let mut union: Vec<char> = left.clone();
                for &c in right { if !union.contains(&c) { union.push(c); } }
                let set_l: HashSet<char> = left.iter().cloned().collect();
                let set_r: HashSet<char> = right.iter().cloned().collect();
                let shared: Vec<char> = set_l.intersection(&set_r).cloned().collect();
                let mut reducible: Vec<char> = Vec::new();
                for &c in &shared { if !labels_out.contains(&c) && !appears_outside(c, i, j) { reducible.push(c); } }
                let mut keep: Vec<char> = Vec::new();
                for &c in &union { if !reducible.contains(&c) { keep.push(c); } }
                let dims_u = |ls: &Vec<char>| -> u128 { ls.iter().map(|c| *dims.get(c).unwrap() as u128).product::<u128>() };
                let l_only: Vec<char> = left.iter().filter(|c| !right.contains(c)).cloned().collect();
                let r_only: Vec<char> = right.iter().filter(|c| !left.contains(c)).cloned().collect();
                let s_keep: Vec<char> = shared.iter().filter(|c| !reducible.contains(c)).cloned().collect();
                let k_red: Vec<char> = reducible.clone();
                let m_sz = dims_u(&l_only);
                let n_sz = dims_u(&r_only);
                let b_sz = dims_u(&s_keep);
                let k_sz = if k_red.is_empty() { 1 } else { dims_u(&k_red) };
                let pair_cost = (b_sz) * (m_sz) * (n_sz) * (k_sz);
                let ctot = cost[i][k] + cost[k+1][j] + pair_cost;
                if ctot < best_cost { best_cost = ctot; best_labels = keep; }
            }
            cost[i][j] = best_cost;
            labels_tbl[i][j] = best_labels;
        }
    }

    fn contract_pair(a: ArrayD<f32>, la: &Vec<char>, b: ArrayD<f32>, lb: &Vec<char>,
                     dims: &HashMap<char, usize>, reducible: &Vec<char>) -> (ArrayD<f32>, Vec<char>) {
        let mut union: Vec<char> = la.clone();
        for &c in lb { if !union.contains(&c) { union.push(c); } }
        let aa = align_to_labels(a, la, &union, dims);
        let bb = align_to_labels(b, lb, &union, dims);
        let mut prod = aa * &bb;
        for ax in (0..union.len()).rev() {
            if reducible.contains(&union[ax]) { prod = prod.sum_axis(Axis(ax)); }
        }
        let keep: Vec<char> = union.into_iter().filter(|c| !reducible.contains(c)).collect();
        (prod, keep)
    }

    fn build_tensor(tensors: &[Tensor], labels_in: &Vec<Vec<char>>, labels_out: &Vec<char>,
                    dims: &HashMap<char, usize>, cost: &Vec<Vec<u128>>, labels_tbl: &Vec<Vec<Vec<char>>>, i: usize, j: usize,
                    appears_outside: &dyn Fn(char, usize, usize) -> bool) -> (ArrayD<f32>, Vec<char>) {
        if i == j { return (tensors[i].data(), labels_in[i].clone()); }
        for k in i..j {
            let (al, ll) = build_tensor(tensors, labels_in, labels_out, dims, cost, labels_tbl, i, k, appears_outside);
            let (ar, lr) = build_tensor(tensors, labels_in, labels_out, dims, cost, labels_tbl, k+1, j, appears_outside);
            let set_l: HashSet<char> = ll.iter().cloned().collect();
            let set_r: HashSet<char> = lr.iter().cloned().collect();
            let shared: Vec<char> = set_l.intersection(&set_r).cloned().collect();
            let mut reducible: Vec<char> = Vec::new();
            for &c in &shared { if !labels_out.contains(&c) && !appears_outside(c, i, j) { reducible.push(c); } }
            let (prod, keep) = contract_pair(al, &ll, ar, &lr, dims, &reducible);
            if &keep == &labels_tbl[i][j] { return (prod, keep); }
        }
        panic!("DP reconstruction failed");
    }

    let (mut y, mut y_labels) = build_tensor(tensors, &labels_in, &labels_out, &dims, &cost, &labels_tbl, 0, n-1, &appears_outside);
    let mut order: Vec<usize> = Vec::new();
    for &c in &labels_out { let pos = y_labels.iter().position(|&r| r==c).unwrap(); order.push(pos); }
    if !order.is_empty() && (0..order.len()).any(|i| order[i]!=i) { y = y.view().permuted_axes(order).to_owned(); }
    let out_t = Tensor::from_array(y.into_dyn());
    if tensors.iter().any(|t| t.0.borrow().requires_grad) {
        struct Node {
            inputs: Vec<Tensor>,
            labels_in: Vec<Vec<char>>,
            labels_out: Vec<char>,
            dims: HashMap<char, usize>,
        }
        impl BackwardNode for Node {
            fn name(&self) -> &'static str { "einsum(dp-general)" }
            fn parents(&self) -> Vec<Tensor> { self.inputs.clone() }
            fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                let go = grads_out[0].as_ref().unwrap().clone();
                let mut all: Vec<char> = Vec::new();
                for ls in &self.labels_in { for &c in ls { if !all.contains(&c) { all.push(c); } } }
                let full_shape: Vec<usize> = all.iter().map(|c| *self.dims.get(c).unwrap()).collect();
                let mut g = go;
                let mut cur = self.labels_out.clone();
                for (pos, &lbl) in all.iter().enumerate() {
                    if let Some(idx) = cur.iter().position(|&c| c==lbl) {
                        if idx != pos {
                            let mut axes: Vec<usize> = (0..cur.len()).collect();
                            let mut new_order = axes.clone(); new_order.remove(idx); new_order.insert(pos, idx);
                            g = g.view().permuted_axes(new_order).to_owned();
                            let c = cur.remove(idx); cur.insert(pos, c);
                        }
                    } else {
                        g = g.insert_axis(Axis(pos));
                        cur.insert(pos, lbl);
                    }
                }
                g = g.broadcast(IxDyn(&full_shape)).unwrap().to_owned();
                fn align(arr: ArrayD<f32>, labels: &[char], all: &[char], dims: &HashMap<char, usize>) -> ArrayD<f32> {
                    let mut out = arr;
                    let mut cur = labels.to_vec();
                    for (pos, &lbl) in all.iter().enumerate() {
                        if let Some(idx) = cur.iter().position(|&c| c==lbl) {
                            if idx != pos {
                                let mut axes: Vec<usize> = (0..cur.len()).collect();
                                let mut new_order = axes.clone(); new_order.remove(idx); new_order.insert(pos, idx);
                                out = out.view().permuted_axes(new_order).to_owned();
                                let c = cur.remove(idx); cur.insert(pos, c);
                            }
                        } else { out = out.insert_axis(Axis(pos)); cur.insert(pos, lbl); }
                    }
                    out.broadcast(IxDyn(&all.iter().map(|c| *dims.get(c).unwrap()).collect::<Vec<_>>())).unwrap().to_owned()
                }
                let aligned: Vec<ArrayD<f32>> = self.inputs.iter().enumerate()
                    .map(|(i,t)| align(t.data(), &self.labels_in[i], &all, &self.dims)).collect();
                let mut grads: Vec<Option<ArrayD<f32>>> = Vec::new();
                for i in 0..self.inputs.len() {
                    let mut prod_except = ndarray::ArrayD::<f32>::from_elem(IxDyn(&full_shape), 1.0);
                    for j in 0..self.inputs.len() { if i==j { continue; } Zip::from(&mut prod_except).and(&aligned[j]).apply(|p,&v|{*p*=v;}); }
                    let mut gi = &g * &prod_except;
                    let keep: Vec<bool> = all.iter().map(|c| self.labels_in[i].contains(c)).collect();
                    for (ax, &k) in keep.iter().enumerate().rev() { if !k { gi = gi.sum_axis(Axis(ax)); } }
                    let remain: Vec<char> = all.iter().cloned().filter(|c| self.labels_in[i].contains(c)).collect();
                    let mut order: Vec<usize> = Vec::new();
                    for &c in &self.labels_in[i] { let pos = remain.iter().position(|&r| r==c).unwrap(); order.push(pos); }
                    if !order.is_empty() && (0..order.len()).any(|k| order[k]!=k) { gi = gi.view().permuted_axes(order).to_owned(); }
                    grads.push(Some(gi.into_dyn()));
                }
                grads
            }
        }
        let gf = GradFn::new(Node { inputs: tensors.to_vec(), labels_in: labels_in.clone(), labels_out: labels_out.clone(), dims });
        out_t.attach_grad_fn(gf, 0, 1, true);
    }
    Ok(out_t)
}

pub fn einsum(spec: &str, tensors: &[Tensor]) -> Result<Tensor> { einsum_opt(spec, tensors, false) }

// ---------- logprod (logabs, sign) ----------
pub fn logprod(x: &Tensor, dim: isize, keepdim: bool, eps: f32, nan_policy: &str, inf_policy: &str) -> Result<(Tensor, Tensor)> {
    let xv = x.data();
    let nd = xv.ndim() as isize;
    let mut d = dim; if d < 0 { d += nd; }
    if d < 0 || (d as usize) >= xv.ndim() { return Err(shape("logprod: dim out of range")); }
    let du = d as usize;
    let mut perm: Vec<usize> = (0..xv.ndim()).collect();
    perm.remove(du); perm.insert(0, du);
    let xpf = xv.view().permuted_axes(perm.clone()).to_owned();
    let mut out_shape = x.shape();
    let reduced = out_shape.remove(du);
    let out_ix = IxDyn(&out_shape);

    let mut out_log = ndarray::ArrayD::<f32>::zeros(out_ix.clone());
    let mut out_sign = ndarray::ArrayD::<f32>::from_elem(out_ix.clone(), 1.0);

    let rest = out_ix.clone();
    let rest_len: usize = if rest.ndim()==0 { 1 } else { rest.slice().iter().product() };
    for flat in 0..rest_len {
        let mut idx_rest: Vec<usize> = Vec::with_capacity(rest.ndim());
        let mut tmp = flat;
        for &d in rest.slice().iter().rev() { idx_rest.push(tmp % d); tmp /= d; }
        idx_rest.reverse();

        let mut any_zero = false;
        let mut any_nan = false;
        let mut any_inf = false;
        let mut sgn: i32 = 1;
        let mut sum_log = 0.0f32;
        let kdim = reduced;
        for k in 0..kdim {
            let mut ix = vec![0usize; 1 + idx_rest.len()];
            ix[0] = k;
            for t in 0..idx_rest.len() { ix[1+t] = idx_rest[t]; }
            let v = xpf[ndarray::IxDyn(&ix)];
            if v.is_nan() { any_nan = true; }
            if v == 0.0 { any_zero = true; }
            if !v.is_nan() && v != 0.0 {
                if v.is_infinite() { any_inf = true; } else { sum_log += (v.abs().max(eps)).ln(); }
                if v.is_sign_negative() { sgn *= -1; }
            }
        }
        let out_l = if any_nan && nan_policy == "propagate" { f32::NAN }
                    else if any_zero { f32::NEG_INFINITY }
                    else if any_inf && inf_policy == "propagate" { f32::INFINITY }
                    else { sum_log };
        let out_s = if any_nan && nan_policy == "propagate" { f32::NAN }
                    else if any_zero { 0.0 } else { sgn as f32 };
        out_log[ndarray::IxDyn(&idx_rest)] = out_l;
        out_sign[ndarray::IxDyn(&idx_rest)] = out_s;
    }
    if keepdim {
        out_log = out_log.insert_axis(Axis(du));
        out_sign = out_sign.insert_axis(Axis(du));
    }
    let t_log = Tensor::from_array(out_log);
    let t_sgn = Tensor::from_array(out_sign);

    if x.0.borrow().requires_grad {
        struct Node { x: Tensor, dim: usize, keepdim: bool, eps: f32, nan_policy: String }
        impl BackwardNode for Node {
            fn name(&self) -> &'static str { "logprod" }
            fn parents(&self) -> Vec<Tensor> { vec![self.x.clone()] }
            fn num_outputs(&self) -> usize { 2 }
            fn backward_multi(&self, grads_out: &[Option<ArrayD<f32>>]) -> Vec<Option<ArrayD<f32>>> {
                let go_opt = &grads_out[0];
                if go_opt.is_none() { return vec![None]; }
                let go = go_opt.as_ref().unwrap().clone();
                let xv = self.x.data();
                let du = self.dim;
                let mut perm: Vec<usize> = (0..xv.ndim()).collect();
                perm.remove(du); perm.insert(0, du);
                let xpf = xv.view().permuted_axes(perm.clone()).to_owned();
                let mut out_shape = self.x.shape();
                out_shape.remove(du);
                let out_ix = IxDyn(&out_shape);
                let go_aligned = if self.keepdim {
                    let mut perm_back: Vec<usize> = (0..go.ndim()).collect();
                    perm_back.remove(du); perm_back.insert(0, du);
                    go.view().permuted_axes(perm_back).to_owned()
                } else {
                    go.insert_axis(Axis(0))
                };
                let mut gx = ndarray::ArrayD::<f32>::zeros(IxDyn(&xv.shape().to_vec()));
                let rest_ix = out_ix.clone();
                let rest_len: usize = if rest_ix.ndim()==0 { 1 } else { rest_ix.slice().iter().product() };
                let kdim = xpf.shape()[0];
                for flat in 0..rest_len {
                    let mut idx_rest: Vec<usize> = Vec::with_capacity(rest_ix.ndim());
                    let mut tmp = flat;
                    for &d in rest_ix.slice().iter().rev() { idx_rest.push(tmp % d); tmp /= d; }
                    idx_rest.reverse();
                    let mut any_zero = false;
                    let mut any_nan = false;
                    for k in 0..kdim {
                        let mut ix = vec![0usize; 1 + idx_rest.len()];
                        ix[0] = k;
                        for t in 0..idx_rest.len() { ix[1+t] = idx_rest[t]; }
                        let v = xpf[ndarray::IxDyn(&ix)];
                        if v.is_nan() { any_nan = true; }
                        if v == 0.0 { any_zero = true; }
                    }
                    if any_nan && self.nan_policy == "propagate" { continue; }
                    if any_zero { continue; }
                    let mut go_pos_ix = vec![0usize; 1 + idx_rest.len()];
                    go_pos_ix[0] = 0;
                    for t in 0..idx_rest.len() { go_pos_ix[1+t] = idx_rest[t]; }
                    let gval = go_aligned[ndarray::IxDyn(&go_pos_ix)];
                    for k in 0..kdim {
                        let mut ix = vec![0usize; 1 + idx_rest.len()];
                        ix[0] = k;
                        for t in 0..idx_rest.len() { ix[1+t] = idx_rest[t]; }
                        let v = xpf[ndarray::IxDyn(&ix)];
                        let grad = if v != 0.0 && !v.is_nan() && !v.is_infinite() { gval / v } else { 0.0 };
                        let mut orig_idx = vec![0usize; xv.ndim()];
                        orig_idx[du] = k;
                        let mut src_ax = 0usize;
                        for ax in 0..xv.ndim() {
                            if ax == du { continue; }
                            orig_idx[ax] = idx_rest[src_ax]; src_ax += 1;
                        }
                        let old = gx[ndarray::IxDyn(&orig_idx)];
                        gx[ndarray::IxDyn(&orig_idx)] = old + grad;
                    }
                }
                vec![Some(gx)]
            }
        }
        let gf = GradFn::new(Node { x: x.clone(), dim: du, keepdim, eps, nan_policy: nan_policy.to_string() });
        t_log.attach_grad_fn(gf.clone(), 0, 2, true);
        t_sgn.attach_grad_fn(gf, 1, 2, false);
    }
    Ok((t_log, t_sgn))
}

use std::collections::{HashMap, HashSet};
use ndarray::ArrayD;
use crate::{Tensor, error::Result};
use super::{GradFn, GradBuf};
use crate::device::Device;

#[cfg(all(feature="mps", target_os="macos"))]
use crate::backend::{Backend, MpsBackend, BackendArrayF32};

pub fn run_backward(root: &Tensor, seed: ArrayD<f32>) -> Result<()> {
    if root.grad_fn().is_none() {
        if matches!(root.dtype(), crate::dtype::DType::F32) {
            let g = if let Some(old) = root.grad() { old + &seed } else { seed.clone() };
            root.set_grad(Some(g));
        }
        return Ok(());
    }
    let mut topo: Vec<GradFn> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    fn collect(t: &Tensor, topo: &mut Vec<GradFn>, visited: &mut HashSet<usize>) {
        if let Some(gf) = t.grad_fn() {
            let k = gf.key();
            if visited.insert(k) {
                for p in gf.parents() { collect(&p, topo, visited); }
                topo.push(gf);
            }
        }
    }
    collect(root, &mut topo, &mut visited);

    let mut out_grads: HashMap<usize, Vec<Option<GradBuf>>> = HashMap::new();
    let seed_gb = prefer_device_buf(root, seed.clone());
    accumulate_into_tensor(root, &seed_gb)?;
    push_grad_to_node(root, seed_gb, &mut out_grads);

    for gf in topo.into_iter().rev() {
        let key = gf.key();
        let num = gf.num_outputs();
        let grads_vec = out_grads.remove(&key).unwrap_or_else(|| vec![None; num]);

        if gf.supports_device() {
            if let Some(dev_outs) = gf.backward_multi_dev(&grads_vec) {
                for (p, maybe_g) in gf.parents().into_iter().zip(dev_outs.into_iter()) {
                    if let Some(gb) = maybe_g {
                        accumulate_into_tensor(&p, &gb)?;
                        if p.grad_fn().is_some() { push_grad_to_node(&p, gb, &mut out_grads); }
                    }
                }
                continue;
            }
        }

        let host_in: Vec<Option<ArrayD<f32>>> = grads_vec.into_iter().map(|og| {
            match og {
                Some(GradBuf::Host(h)) => Some(h),
                #[cfg(all(feature="mps", target_os="macos"))]
                Some(GradBuf::Device{arr, ..}) => Some(MpsBackend::new().to_host_f32(&arr).expect("to_host_f32")),
                _ => None
            }
        }).collect();
        let gout = gf.backward_multi(&host_in);
        for (p, maybe_h) in gf.parents().into_iter().zip(gout.into_iter()) {
            if let Some(h) = maybe_h {
                let gb = GradBuf::Host(h);
                accumulate_into_tensor(&p, &gb)?;
                if p.grad_fn().is_some() { push_grad_to_node(&p, gb, &mut out_grads); }
            }
        }
    }
    Ok(())
}

fn prefer_device_buf(t: &Tensor, host: ArrayD<f32>) -> GradBuf {
    match t.device() {
        Device::Mps => {
            #[cfg(all(feature="mps", target_os="macos"))] {
                let be = MpsBackend::new();
                let arr = be.from_host_f32(&host).expect("from_host");
                return GradBuf::Device { arr, shape: t.shape(), device: Device::Mps };
            }
            #[cfg(not(all(feature="mps", target_os="macos")))] { GradBuf::Host(host) }
        }
        _ => GradBuf::Host(host),
    }
}

fn accumulate_into_tensor(t: &Tensor, g: &GradBuf) -> Result<()> {
    match g {
        GradBuf::Host(h) => t.accumulate_host_grad(h),
        #[cfg(all(feature="mps", target_os="macos"))]
        GradBuf::Device{arr, device, ..} => match device {
            Device::Mps => t.accumulate_device_grad(arr),
            _ => t.accumulate_host_grad(&MpsBackend::new().to_host_f32(arr)?),
        },
        #[cfg(not(all(feature="mps", target_os="macos")))]
        _ => t.accumulate_host_grad(match g { GradBuf::Host(h) => h, _ => unreachable!() }),
    }
}

fn push_grad_to_node(t: &Tensor, g: GradBuf, out: &mut HashMap<usize, Vec<Option<GradBuf>>>) {
    let Some(gf) = t.grad_fn() else { return; };
    let key = gf.key();
    let num = t.num_outputs();
    let idx = t.out_index();
    let entry = out.entry(key).or_insert_with(|| vec![None; num]);
    match &entry[idx] {
        None => entry[idx] = Some(g),
        Some(GradBuf::Host(prev)) => {
            match g {
                GradBuf::Host(h) => entry[idx] = Some(GradBuf::Host(prev.clone() + &h)),
                #[cfg(all(feature="mps", target_os="macos"))]
                GradBuf::Device{arr, ..} => {
                    let be = MpsBackend::new();
                    let h = be.to_host_f32(&arr).expect("to_host for accum");
                    entry[idx] = Some(GradBuf::Host(prev.clone() + &h));
                }
                #[cfg(not(all(feature="mps", target_os="macos")))]
                _ => {}
            }
        }
        Some(GradBuf::Device{arr: prev_arr, ..}) => {
            #[cfg(all(feature="mps", target_os="macos"))] {
                let be = MpsBackend::new();
                let d_new = match g {
                    GradBuf::Host(h) => be.from_host_f32(&h).expect("from_host"),
                    GradBuf::Device{arr, ..} => arr,
                };
                let sum = be.add(prev_arr, &d_new).expect("device add");
                entry[idx] = Some(GradBuf::Device { arr: sum, shape: t.shape(), device: Device::Mps });
            }
            #[cfg(not(all(feature="mps", target_os="macos")))] {
                if let GradBuf::Host(h) = g { entry[idx] = Some(GradBuf::Host(h)); }
            }
        }
    }
}

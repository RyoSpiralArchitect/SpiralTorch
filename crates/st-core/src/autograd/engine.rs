
use std::collections::{HashMap, HashSet};
use ndarray::{ArrayD, IxDyn};

use crate::{Tensor, error::Result};
use super::{GradFn, GradBuf};
use crate::device::Device;

#[cfg(all(feature="mps", target_os="macos"))]
use crate::backend::{Backend, MpsBackend, BackendArrayF32};

/// Engine entry point: run backprop from `root` given a seed gradient (host).
pub fn run_backward(root: &Tensor, seed: ArrayD<f32>) -> Result<()> {
    // Build topological order of GradFn
    let mut topo: Vec<GradFn> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();

    fn collect_from_tensor(t: &Tensor, topo: &mut Vec<GradFn>, visited: &mut HashSet<usize>) {
        if let Some(gf) = t.grad_fn() {
            let key = gf.key();
            if !visited.contains(&key) {
                visited.insert(key);
                for p in gf.parents() { collect_from_tensor(&p, topo, visited); }
                topo.push(gf);
            }
        }
    }
    collect_from_tensor(root, &mut topo, &mut visited);

    // Pending output-grads per node key (device-capable nodes keep device when possible).
    let mut out_grads: HashMap<usize, Vec<Option<GradBuf>>> = HashMap::new();

    // Seed-accumulate into root tensor (prefer device if root is device).
    let seed_buf = prefer_device_buf(root, seed.clone());
    accumulate_into_tensor(root, &seed_buf)?;
    // Also enqueue into root's producer node if any.
    if root.grad_fn().is_some() {
        push_grad_to_node(root, seed_buf, &mut out_grads);
    } else {
        return Ok(());
    }

    // Execute in reverse topo order
    for gf in topo.into_iter().rev() {
        let key = gf.key();
        let num = gf.num_outputs();
        let grads_vec = out_grads.remove(&key).unwrap_or_else(|| vec![None; num]);

        if gf.supports_device() {
            if let Some(douts) = gf.backward_multi_dev(&grads_vec) {
                let parents = gf.parents();
                for (p, maybe_g) in parents.into_iter().zip(douts.into_iter()) {
                    if let Some(gb) = maybe_g {
                        accumulate_into_tensor(&p, &gb)?;
                        if p.grad_fn().is_some() {
                            push_grad_to_node(&p, gb, &mut out_grads);
                        }
                    }
                }
                continue;
            }
        }

        // Host fallback: convert inputs to host, call host backward, accumulate
        let host_in: Vec<Option<ArrayD<f32>>> = grads_vec.into_iter().map(|og| {
            match og {
                Some(GradBuf::Host(h)) => Some(h),
                #[cfg(all(feature="mps", target_os="macos"))]
                Some(GradBuf::Device{arr, ..}) => {
                    let be = MpsBackend::new();
                    Some(be.to_host_f32(&arr).expect("to_host_f32 in engine"))
                }
                _ => None,
            }
        }).collect();

        let gout = gf.backward_multi(&host_in);
        let parents = gf.parents();
        for (p, maybe_h) in parents.into_iter().zip(gout.into_iter()) {
            if let Some(h) = maybe_h {
                let gb = GradBuf::Host(h.clone());
                accumulate_into_tensor(&p, &gb)?;
                if p.grad_fn().is_some() { push_grad_to_node(&p, gb, &mut out_grads); }
            }
        }
    }

    Ok(())
}

/// Prefer device grad if tensor lives on device and feature is available.
fn prefer_device_buf(t: &Tensor, host: ArrayD<f32>) -> GradBuf {
    match t.device() {
        Device::Mps => {
            #[cfg(all(feature="mps", target_os="macos"))] {
                let be = MpsBackend::new();
                let arr = be.from_host_f32(&host).expect("engine from_host");
                return GradBuf::Device { arr, shape: t.shape(), device: Device::Mps };
            }
            #[cfg(not(all(feature="mps", target_os="macos")))] {
                GradBuf::Host(host)
            }
        }
        _ => GradBuf::Host(host)
    }
}

/// Accumulate gradient into a Tensor (device-first when possible).
fn accumulate_into_tensor(t: &Tensor, g: &GradBuf) -> Result<()> {
    match g {
        GradBuf::Host(h) => t.accumulate_host_grad(h),
        #[cfg(all(feature="mps", target_os="macos"))]
        GradBuf::Device{arr, device, ..} => {
            match device {
                Device::Mps => t.accumulate_device_grad(arr),
                _ => t.accumulate_host_grad(&t.to_host_array(arr)?),
            }
        }
        #[cfg(not(all(feature="mps", target_os="macos")))]
        _ => t.accumulate_host_grad(match g { GradBuf::Host(h) => h, _ => unreachable!() }),
    }
}

/// Push gradient into the producing node's output slot (device-capable nodes keep device).
fn push_grad_to_node(t: &Tensor, g: GradBuf, out: &mut HashMap<usize, Vec<Option<GradBuf>>>) {
    let Some(gf) = t.grad_fn() else { return; };
    let key = gf.key();
    let num = t.num_outputs();
    let idx = t.out_index();
    let entry = out.entry(key).or_insert_with(|| vec![None; num]);

    // If node supports device, try to keep device representation.
    let keep_device = gf.supports_device();

    match (&entry[idx], keep_device) {
        (None, _) => entry[idx] = Some(g),
        (Some(GradBuf::Host(prev)), false) => {
            match g {
                GradBuf::Host(h) => entry[idx] = Some(GradBuf::Host(prev.clone() + &h)),
                #[cfg(all(feature="mps", target_os="macos"))]
                GradBuf::Device{arr, ..} => {
                    let be = MpsBackend::new();
                    let h = be.to_host_f32(&arr).expect("to_host for accumulation");
                    entry[idx] = Some(GradBuf::Host(prev.clone() + &h));
                }
                #[cfg(not(all(feature="mps", target_os="macos")))]
                _ => {}
            }
        }
        (Some(GradBuf::Host(prev)), true) => {
            // Convert both to device and add on device.
            #[cfg(all(feature="mps", target_os="macos"))] {
                let be = MpsBackend::new();
                let d_prev = be.from_host_f32(prev).expect("from_host");
                let d_new = match g {
                    GradBuf::Host(h) => be.from_host_f32(&h).expect("from_host"),
                    GradBuf::Device{arr, ..} => arr,
                };
                let sum = be.add(&d_prev, &d_new).expect("device add");
                entry[idx] = Some(GradBuf::Device { arr: sum, shape: t.shape(), device: Device::Mps });
            }
            #[cfg(not(all(feature="mps", target_os="macos")))] {
                if let GradBuf::Host(h) = g { entry[idx] = Some(GradBuf::Host(prev.clone() + &h)); }
            }
        }
        (Some(GradBuf::Device{arr: prev_arr, ..}), true) => {
            #[cfg(all(feature="mps", target_os="macos"))] {
                let be = MpsBackend::new();
                let d_new = match g {
                    GradBuf::Host(h) => be.from_host_f32(&h).expect("from_host"),
                    GradBuf::Device{arr, ..} => arr,
                };
                let sum = be.add(prev_arr, &d_new).expect("add");
                entry[idx] = Some(GradBuf::Device { arr: sum, shape: t.shape(), device: Device::Mps });
            }
            #[cfg(not(all(feature="mps", target_os="macos")))] {
                // fallback to host addition
                if let GradBuf::Host(h) = g {
                    let be = MpsBackend::new(); // not compiled
                    let _ = be; // silence
                    // unreachable in non-mps build
                    entry[idx] = Some(GradBuf::Host(h));
                }
            }
        }
        (Some(GradBuf::Device{arr: prev_arr, ..}), false) => {
            // Node not device-capable: convert to host and add there.
            #[cfg(all(feature="mps", target_os="macos"))] {
                let be = MpsBackend::new();
                let prev_h = be.to_host_f32(prev_arr).expect("to_host");
                let new_h = match g {
                    GradBuf::Host(h) => h,
                    GradBuf::Device{arr, ..} => be.to_host_f32(&arr).expect("to_host"),
                };
                entry[idx] = Some(GradBuf::Host(prev_h + &new_h));
            }
            #[cfg(not(all(feature="mps", target_os="macos")))] {
                if let GradBuf::Host(h) = g { entry[idx] = Some(GradBuf::Host(h)); }
            }
        }
    }
}

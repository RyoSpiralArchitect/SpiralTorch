// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#[derive(Clone, Debug)]
pub struct TopKShard<T> {
    pub vals: Vec<T>,
    pub idxs: Vec<i32>,
}

use st_tensor::{emit_tensor_op, emit_tensor_op_meta};

fn finite_meta_f32(value: f32) -> serde_json::Value {
    serde_json::Value::from(if value.is_finite() { value as f64 } else { 0.0 })
}

fn l1_energy(values: &[f32]) -> f32 {
    values
        .iter()
        .filter(|value| value.is_finite())
        .map(|value| value.abs())
        .sum()
}

pub fn merge_two_shards_f32(a: &TopKShard<f32>, b: &TopKShard<f32>, k: usize) -> TopKShard<f32> {
    let input_count = a.vals.len() + b.vals.len();
    let index_count = a.idxs.len() + b.idxs.len();
    let dropped_shape_mismatch =
        a.vals.len().saturating_sub(a.idxs.len()) + b.vals.len().saturating_sub(b.idxs.len());
    let dropped_non_finite = a
        .vals
        .iter()
        .chain(b.vals.iter())
        .filter(|value| !value.is_finite())
        .count();
    let input_energy = l1_energy(&a.vals) + l1_energy(&b.vals);

    let mut pairs: Vec<(f32, i32)> = a
        .vals
        .iter()
        .cloned()
        .zip(a.idxs.iter().cloned())
        .chain(b.vals.iter().cloned().zip(b.idxs.iter().cloned()))
        .filter(|(value, _)| value.is_finite())
        .collect();
    let finite_pair_count = pairs.len();
    pairs.sort_by(|x, y| y.0.total_cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
    let mut out_vals = Vec::with_capacity(k);
    let mut out_idx = Vec::with_capacity(k);
    for (v, i) in pairs.into_iter().take(k) {
        out_vals.push(v);
        out_idx.push(i);
    }
    let output_energy = l1_energy(&out_vals);
    emit_tensor_op(
        "distributed_topk_merge",
        &[input_count.max(1), index_count.max(1)],
        &[out_vals.len().max(1), 2],
    );
    emit_tensor_op_meta("distributed_topk_merge", || {
        serde_json::json!({
            "backend": "cpu",
            "requested_backend": "auto",
            "kind": "st_core_distributed_topk_merge",
            "merge_backend": "cpu_sort",
            "sort_backend": "cpu_total_cmp",
            "merge_mode": "pair_filter_sort_truncate",
            "route_blocker": "host_pair_sort_and_truncate",
            "k": k,
            "input_count": input_count,
            "index_count": index_count,
            "finite_pair_count": finite_pair_count,
            "output_count": out_vals.len(),
            "dropped_non_finite": dropped_non_finite,
            "dropped_shape_mismatch": dropped_shape_mismatch,
            "truncated": out_vals.len() < input_count.saturating_sub(dropped_non_finite),
            "input_l1_energy": finite_meta_f32(input_energy),
            "output_l1_energy": finite_meta_f32(output_energy),
            "retained_energy_ratio": finite_meta_f32(if input_energy > f32::EPSILON {
                output_energy / input_energy
            } else {
                1.0
            }),
            "top_value": finite_meta_f32(out_vals.first().copied().unwrap_or(0.0)),
            "top_index": out_idx.first().copied().unwrap_or(-1),
            "estimated_filter_values": input_count,
            "estimated_sort_items": finite_pair_count,
            "estimated_output_values": out_vals.len(),
        })
    });
    TopKShard {
        vals: out_vals,
        idxs: out_idx,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn merge_two_shards_drops_non_finite_and_emits_meta() {
        let _lock = crate::telemetry::tensor_observer_lock();
        let events = Arc::new(Mutex::new(Vec::new()));
        let captured = events.clone();
        let previous = st_tensor::set_tensor_op_meta_observer(Some(Arc::new(move |event| {
            captured
                .lock()
                .unwrap()
                .push((event.op_name, event.data.clone()));
        })));

        let a = TopKShard {
            vals: vec![0.5, f32::NAN, 0.2],
            idxs: vec![5, 99, 2],
        };
        let b = TopKShard {
            vals: vec![0.7, f32::INFINITY, -0.1],
            idxs: vec![7, 100, -1],
        };
        let merged = merge_two_shards_f32(&a, &b, 3);
        st_tensor::set_tensor_op_meta_observer(previous);

        assert_eq!(merged.vals, vec![0.7, 0.5, 0.2]);
        assert_eq!(merged.idxs, vec![7, 5, 2]);
        let events = events.lock().unwrap();
        let meta = events
            .iter()
            .find(|(op_name, data)| {
                *op_name == "distributed_topk_merge"
                    && data["kind"] == "st_core_distributed_topk_merge"
            })
            .expect("distributed topk merge metadata event");
        assert_eq!(meta.1["backend"], "cpu");
        assert_eq!(meta.1["merge_backend"], "cpu_sort");
        assert_eq!(meta.1["sort_backend"], "cpu_total_cmp");
        assert_eq!(meta.1["merge_mode"], "pair_filter_sort_truncate");
        assert_eq!(meta.1["route_blocker"], "host_pair_sort_and_truncate");
        assert_eq!(meta.1["k"], 3);
        assert_eq!(meta.1["input_count"], 6);
        assert_eq!(meta.1["finite_pair_count"], 4);
        assert_eq!(meta.1["output_count"], 3);
        assert_eq!(meta.1["dropped_non_finite"], 2);
        assert_eq!(meta.1["top_index"], 7);
        assert_eq!(meta.1["estimated_filter_values"], 6);
        assert_eq!(meta.1["estimated_sort_items"], 4);
        assert_eq!(meta.1["estimated_output_values"], 3);
        assert!(meta.1["retained_energy_ratio"].as_f64().unwrap() > 0.0);
    }
}

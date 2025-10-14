use std::fmt;

/// Enumerates the backends that participate in the unified heuristic chooser.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// WebGPU / WGPU backend.
    Wgpu,
    /// CUDA backend.
    Cuda,
    /// ROCm HIP backend.
    Hip,
    /// CPU fallback.
    Cpu,
}

/// Minimal capability description shared across backends.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DeviceCaps {
    pub backend: BackendKind,
    pub subgroup: bool,
    pub lane_width: u32,
    pub max_workgroup: u32,
    pub shared_mem_per_workgroup: Option<u32>,
}

impl fmt::Debug for DeviceCaps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceCaps")
            .field("backend", &self.backend)
            .field("subgroup", &self.subgroup)
            .field("lane_width", &self.lane_width)
            .field("max_workgroup", &self.max_workgroup)
            .field("shared_mem_per_workgroup", &self.shared_mem_per_workgroup)
            .finish()
    }
}

impl DeviceCaps {
    /// Default per-lane quota used by the heuristics when picking merge kernels.
    #[inline]
    pub fn default_lane_quota(k: u32) -> u32 {
        if k >= 64 {
            32
        } else if k >= 16 {
            16
        } else {
            8
        }
    }

    fn new(backend: BackendKind, lane_width: u32) -> Self {
        Self {
            backend,
            subgroup: matches!(
                backend,
                BackendKind::Wgpu | BackendKind::Cuda | BackendKind::Hip
            ),
            lane_width: lane_width.max(1),
            max_workgroup: 1,
            shared_mem_per_workgroup: None,
        }
    }

    pub fn wgpu(lane_width: u32, subgroup: bool, max_workgroup: u32) -> Self {
        Self {
            subgroup,
            max_workgroup: max_workgroup.max(1),
            ..Self::new(BackendKind::Wgpu, lane_width)
        }
    }

    pub fn cuda(lane_width: u32, max_block: u32, shared_mem_per_block: Option<u32>) -> Self {
        Self {
            max_workgroup: max_block.max(1),
            shared_mem_per_workgroup: shared_mem_per_block,
            ..Self::new(BackendKind::Cuda, lane_width)
        }
    }

    pub fn hip(lane_width: u32, max_block: u32, shared_mem_per_block: Option<u32>) -> Self {
        Self {
            max_workgroup: max_block.max(1),
            shared_mem_per_workgroup: shared_mem_per_block,
            ..Self::new(BackendKind::Hip, lane_width)
        }
    }

    pub fn cpu() -> Self {
        Self {
            subgroup: false,
            lane_width: 1,
            max_workgroup: 128,
            shared_mem_per_workgroup: None,
            backend: BackendKind::Cpu,
        }
    }

    pub fn with_subgroup(mut self, subgroup: bool) -> Self {
        self.subgroup = subgroup;
        self
    }

    pub fn with_max_workgroup(mut self, max_workgroup: u32) -> Self {
        self.max_workgroup = max_workgroup.max(1);
        self
    }

    pub fn with_shared_mem(mut self, shared_mem: Option<u32>) -> Self {
        self.shared_mem_per_workgroup = shared_mem;
        self
    }

    /// Recommended workgroup size for a particular problem size.
    pub fn recommended_workgroup_for_rows(&self, rows: u32) -> u32 {
        let lanes = self.lane_width.max(1);
        let mut target = if self.subgroup {
            lanes.saturating_mul(8)
        } else {
            lanes.saturating_mul(4)
        };
        target = target.max(lanes).min(self.max_workgroup.max(lanes));

        let mut aligned = target;
        if rows > 0 && rows < target {
            aligned = align_up(rows.max(lanes), lanes);
        }
        let min_group = if matches!(self.backend, BackendKind::Cpu) {
            128
        } else {
            128.max(lanes)
        };
        aligned = aligned.max(min_group);
        aligned.clamp(lanes, self.max_workgroup.max(lanes))
    }

    /// Recommended workgroup without row knowledge (nominal occupancy target).
    pub fn recommended_workgroup(&self) -> u32 {
        self.recommended_workgroup_for_rows(self.max_workgroup)
    }

    /// Recommended tile (TopK) and compaction tile (MidK/BottomK) for a column count.
    pub fn recommended_tiles(&self, cols: u32) -> (u32, u32) {
        let sweep = self.recommended_sweep_tile(cols);
        let compaction = self.recommended_compaction_tile(cols).min(sweep);
        (sweep, compaction)
    }

    pub fn recommended_kl(&self, k: u32) -> u32 {
        let lanes = self.lane_width.max(1);
        let mut out = 8;
        if k >= lanes * 2 {
            out = 16;
        }
        if k >= lanes * 4 {
            out = 32;
        }
        if matches!(self.backend, BackendKind::Cpu) {
            out = out.min(16);
        }
        out
    }

    pub fn preferred_k_loop(&self, k: u32) -> u32 {
        self.recommended_kl(k)
    }

    pub fn preferred_channel(&self, cols: u32) -> u32 {
        if cols <= 16_384 {
            0
        } else if matches!(self.backend, BackendKind::Cpu) {
            4_096
        } else if let Some(shared) = self.shared_mem_per_workgroup {
            if shared < 48 * 1024 {
                4_096
            } else {
                8_192
            }
        } else {
            8_192
        }
    }

    pub fn recommended_sweep_tile(&self, cols: u32) -> u32 {
        if matches!(self.backend, BackendKind::Cpu) {
            return if cols > 8_192 { 256 } else { 128 };
        }
        let lanes = self.lane_width.max(1);
        let occ = if self.subgroup { lanes * 8 } else { lanes * 4 };
        let mut target = (cols / 8).max(occ).max(256);
        if let Some(shared) = self.shared_mem_per_workgroup {
            if shared < 32 * 1024 {
                target = target.min(2_048);
            }
        }
        quantize_tile(target, 8_192)
    }

    pub fn recommended_compaction_tile(&self, cols: u32) -> u32 {
        if matches!(self.backend, BackendKind::Cpu) {
            return if cols > 16_384 { 256 } else { 128 };
        }
        let lanes = self.lane_width.max(1);
        let occ = if self.subgroup { lanes * 4 } else { lanes * 2 };
        let mut target = (cols / 16).max(occ).max(256);
        if let Some(shared) = self.shared_mem_per_workgroup {
            if shared < 32 * 1024 {
                target = target.min(1_024);
            }
        }
        quantize_tile(target, 4_096)
    }

    pub fn recommended_channel_stride(&self, cols: u32) -> u32 {
        if cols > 262_144 {
            16_384
        } else if cols > 65_536 {
            8_192
        } else if cols > 16_384 {
            4_096
        } else {
            0
        }
    }

    pub fn preferred_merge_kind(&self, k: u32) -> u32 {
        match self.backend {
            BackendKind::Wgpu => {
                if self.subgroup && k <= 128 {
                    2
                } else if k <= 2_048 {
                    1
                } else {
                    0
                }
            }
            BackendKind::Cuda => {
                if k <= self.lane_width * 4 {
                    2
                } else if k <= 4_096 {
                    1
                } else {
                    0
                }
            }
            BackendKind::Hip => {
                if k <= self.lane_width * 2 {
                    2
                } else if k <= 4_096 {
                    1
                } else {
                    0
                }
            }
            BackendKind::Cpu => 0,
        }
    }

    pub fn preferred_substrategy(&self, merge_kind: u32, k: u32) -> u32 {
        match merge_kind {
            2 => {
                if k <= 128 {
                    4 // warp_heap
                } else {
                    5 // warp_bitonic
                }
            }
            1 => {
                if k <= 1_024 {
                    1 // heap
                } else {
                    2 // kway
                }
            }
            _ => 3, // bitonic
        }
    }

    pub fn prefers_two_stage(&self, rows: u32, cols: u32, k: u32) -> bool {
        let cols_heavy = cols > 32_768;
        let k_heavy = k > self.lane_width.max(1) * 4;
        let rows_small = rows < 256;
        match self.backend {
            BackendKind::Wgpu => cols_heavy || (k_heavy && !rows_small),
            BackendKind::Cuda | BackendKind::Hip => {
                cols_heavy || (k_heavy && rows > self.lane_width.max(1) * 8)
            }
            BackendKind::Cpu => cols_heavy && rows > 512,
        }
    }

    pub fn occupancy_hint(&self, threads_per_group: u32, shared_mem_bytes: Option<u32>) -> f32 {
        if self.max_workgroup == 0 {
            return 0.0;
        }
        let threads = threads_per_group.min(self.max_workgroup).max(1);
        let occ_threads = threads as f32 / self.max_workgroup as f32;
        let occ_shared = match (self.shared_mem_per_workgroup, shared_mem_bytes) {
            (Some(budget), Some(req)) if budget > 0 && req > 0 => {
                (budget as f32 / req as f32).min(1.0)
            }
            _ => 1.0,
        };
        (occ_threads.min(1.0) * 0.7) + (occ_shared.min(1.0) * 0.3)
    }
}

impl Default for DeviceCaps {
    fn default() -> Self {
        Self::cpu()
    }
}

fn align_up(value: u32, align: u32) -> u32 {
    if align <= 1 {
        return value;
    }
    let remainder = value % align;
    if remainder == 0 {
        value
    } else {
        value + (align - remainder)
    }
}

fn quantize_tile(target: u32, limit: u32) -> u32 {
    const CANDS: [u32; 7] = [256, 512, 1_024, 2_048, 4_096, 8_192, 16_384];
    let mut best = 256;
    let mut best_diff = u32::MAX;
    for &cand in &CANDS {
        if cand > limit {
            break;
        }
        let diff = target.abs_diff(cand);
        if diff < best_diff || (diff == best_diff && cand > best) {
            best = cand;
            best_diff = diff;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_respect_defaults() {
        let wgpu = DeviceCaps::wgpu(32, true, 256);
        assert_eq!(wgpu.backend, BackendKind::Wgpu);
        assert!(wgpu.subgroup);
        assert_eq!(wgpu.lane_width, 32);
        assert_eq!(wgpu.max_workgroup, 256);
        assert_eq!(wgpu.shared_mem_per_workgroup, None);

        let cuda = DeviceCaps::cuda(32, 1_024, Some(64 * 1_024));
        assert_eq!(cuda.backend, BackendKind::Cuda);
        assert!(cuda.subgroup);
        assert_eq!(cuda.max_workgroup, 1_024);
        assert_eq!(cuda.shared_mem_per_workgroup, Some(64 * 1_024));

        let cpu = DeviceCaps::cpu();
        assert_eq!(cpu.backend, BackendKind::Cpu);
        assert!(!cpu.subgroup);
        assert_eq!(cpu.lane_width, 1);
    }

    #[test]
    fn recommended_helpers_follow_backend_logic() {
        let cuda = DeviceCaps::cuda(32, 1_024, Some(64 * 1_024));
        assert_eq!(cuda.recommended_workgroup_for_rows(4_096), 256);
        assert_eq!(cuda.recommended_workgroup_for_rows(32), 128);
        assert!(cuda.recommended_sweep_tile(200_000) >= 2_048);
        assert!(cuda.recommended_compaction_tile(200_000) >= 512);
        assert!(cuda.prefers_two_stage(4_096, 100_000, 256));

        let cpu = DeviceCaps::cpu();
        assert_eq!(cpu.recommended_workgroup(), 128);
        assert_eq!(cpu.recommended_sweep_tile(4_096), 128);
        assert_eq!(cpu.recommended_compaction_tile(20_000), 256);
        assert!(!cpu.prefers_two_stage(128, 8_000, 64));
    }

    #[test]
    fn merge_preferences_match_reference() {
        let wgpu = DeviceCaps::wgpu(32, true, 256);
        assert_eq!(wgpu.preferred_merge_kind(64), 2);
        assert_eq!(wgpu.preferred_merge_kind(2_048), 1);
        assert_eq!(wgpu.preferred_merge_kind(8_192), 0);

        let hip = DeviceCaps::hip(32, 1_024, Some(64 * 1_024));
        assert_eq!(hip.preferred_merge_kind(32), 2);
        assert_eq!(hip.preferred_merge_kind(2_048), 1);
        assert_eq!(hip.preferred_merge_kind(8_192), 0);
    }

    #[test]
    fn occupancy_hint_prefers_full_groups() {
        let caps = DeviceCaps::cuda(32, 1_024, Some(64 * 1_024));
        let high = caps.occupancy_hint(1_024, Some(48 * 1_024));
        let low = caps.occupancy_hint(128, Some(48 * 1_024));
        assert!(high > low);
    }
}

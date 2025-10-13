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
///
/// The struct intentionally keeps only a handful of knobs that influence the
/// heuristic chooser.  Higher level backends can always extend this data via
/// wrapper structs, but the unified heuristics only care about the fields that
/// affect merge-kind decisions and tile sizing.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct DeviceCaps {
    /// Backend discriminator so we can inject backend-specific biases.
    pub backend: BackendKind,
    /// Whether the backend supports subgroup/warp execution that maps to the
    /// "warp" merge strategy.
    pub subgroup: bool,
    /// Native lane width for subgroup/warp style operations.
    pub lane_width: u32,
    /// Maximum workgroup size that the backend can launch efficiently.
    pub max_workgroup: u32,
    /// Optional shared memory available per workgroup (bytes).
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

    /// Creates a new capability descriptor for the requested backend.
    fn new(backend: BackendKind, lane_width: u32) -> Self {
        Self {
            backend,
            subgroup: false,
            lane_width: lane_width.max(1),
            max_workgroup: 1,
            shared_mem_per_workgroup: None,
        }
    }

    /// WebGPU/WGPU capabilities.
    pub fn wgpu(lane_width: u32, subgroup: bool, max_workgroup: u32) -> Self {
        Self {
            subgroup,
            max_workgroup: max_workgroup.max(1),
            ..Self::new(BackendKind::Wgpu, lane_width)
        }
    }

    /// CUDA capabilities.
    pub fn cuda(lane_width: u32, max_block: u32, shared_mem_per_block: Option<u32>) -> Self {
        Self {
            subgroup: true,
            max_workgroup: max_block.max(1),
            shared_mem_per_workgroup: shared_mem_per_block,
            ..Self::new(BackendKind::Cuda, lane_width)
        }
    }

    /// HIP capabilities.
    pub fn hip(lane_width: u32, max_block: u32, shared_mem_per_block: Option<u32>) -> Self {
        Self {
            subgroup: true,
            max_workgroup: max_block.max(1),
            shared_mem_per_workgroup: shared_mem_per_block,
            ..Self::new(BackendKind::Hip, lane_width)
        }
    }

    /// CPU fallback capabilities.
    pub fn cpu() -> Self {
        Self {
            max_workgroup: 128,
            ..Self::new(BackendKind::Cpu, 1)
        }
    }

    /// Builder style helper to override the subgroup flag.
    pub fn with_subgroup(mut self, subgroup: bool) -> Self {
        self.subgroup = subgroup;
        self
    }

    /// Builder style helper to override the maximum workgroup size.
    pub fn with_max_workgroup(mut self, max_workgroup: u32) -> Self {
        self.max_workgroup = max_workgroup.max(1);
        self
    }

    /// Builder style helper to override the shared memory budget.
    pub fn with_shared_mem(mut self, shared_mem: Option<u32>) -> Self {
        self.shared_mem_per_workgroup = shared_mem;
        self
    }

    /// Derive the merge-kind preference that matches the fallback heuristic.
    pub fn preferred_merge_kind(&self, k: u32) -> u32 {
        match self.backend {
            BackendKind::Wgpu => {
                if self.subgroup && k <= 128 {
                    2
                } else if k <= 2048 {
                    1
                } else {
                    0
                }
            }
            BackendKind::Cuda => {
                if k <= self.lane_width * 4 {
                    2
                } else if k <= 4096 {
                    1
                } else {
                    0
                }
            }
            BackendKind::Hip => {
                if k <= self.lane_width * 2 {
                    2
                } else if k <= 4096 {
                    1
                } else {
                    0
                }
            }
            BackendKind::Cpu => 0,
        }
    }

    /// Derive the merge sub-strategy to pair with `preferred_merge_kind`.
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
                if k <= 1024 {
                    1 // heap
                } else {
                    2 // kway
                }
            }
            _ => 3, // bitonic
        }
    }

    /// Whether two‑stage compaction should be enabled for the given problem.
    pub fn prefers_two_stage(&self, cols: u32, k: u32) -> bool {
        if !(cols > 32_768 || k > 128) {
            return false;
        }
        if let Some(limit) = self.shared_mem_per_workgroup {
            let lane_width = self.lane_width.max(1);
            let default_wg = if self.subgroup { 256 } else { 128 };
            let wg = self.max_workgroup.min(default_wg).max(lane_width);
            let lane_groups = (wg + lane_width - 1) / lane_width;
            let lanes_total = lane_groups * lane_width;
            let lane_quota = Self::default_lane_quota(k);
            let required = lanes_total.saturating_mul(lane_quota).saturating_mul(8);
            (required as u64) <= (limit as u64)
        } else {
            true
        }
    }
}

impl Default for DeviceCaps {
    fn default() -> Self {
        Self::cpu()
    }
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

        let cuda = DeviceCaps::cuda(32, 1024, Some(64 * 1024));
        assert_eq!(cuda.backend, BackendKind::Cuda);
        assert!(cuda.subgroup);
        assert_eq!(cuda.max_workgroup, 1024);
        assert_eq!(cuda.shared_mem_per_workgroup, Some(64 * 1024));

        let cpu = DeviceCaps::cpu();
        assert_eq!(cpu.backend, BackendKind::Cpu);
        assert!(!cpu.subgroup);
        assert_eq!(cpu.lane_width, 1);
    }

    #[test]
    fn merge_preferences_track_existing_logic() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        assert_eq!(caps.preferred_merge_kind(64), 2);
        assert_eq!(caps.preferred_merge_kind(512), 1);
        assert_eq!(caps.preferred_merge_kind(4096), 0);

        let cuda = DeviceCaps::cuda(32, 1024, None);
        assert_eq!(cuda.preferred_merge_kind(64), 2);
        assert_eq!(cuda.preferred_merge_kind(2048), 1);
        assert_eq!(cuda.preferred_merge_kind(8192), 0);

        let hip = DeviceCaps::hip(32, 1024, None);
        assert_eq!(hip.preferred_merge_kind(32), 2);
        assert_eq!(hip.preferred_merge_kind(2048), 1);
        assert_eq!(hip.preferred_merge_kind(8192), 0);
    }

    #[test]
    fn substrategy_matches_merge_kind() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        assert_eq!(caps.preferred_substrategy(2, 64), 4);
        assert_eq!(caps.preferred_substrategy(2, 1024), 5);
        assert_eq!(caps.preferred_substrategy(1, 512), 1);
        assert_eq!(caps.preferred_substrategy(1, 4096), 2);
        assert_eq!(caps.preferred_substrategy(0, 512), 3);
    }

    #[test]
    fn two_stage_matches_reference_logic() {
        let caps = DeviceCaps::wgpu(32, true, 256);
        assert!(caps.prefers_two_stage(40_000, 64));
        assert!(caps.prefers_two_stage(2_000, 256));
        assert!(!caps.prefers_two_stage(10_000, 64));
    }

    #[test]
    fn two_stage_respects_shared_memory_budget() {
        let caps = DeviceCaps::cuda(32, 1024, Some(32 * 1024));
        assert!(!caps.prefers_two_stage(100_000, 256));

        let generous = DeviceCaps::cuda(32, 1024, Some(128 * 1024));
        assert!(generous.prefers_two_stage(100_000, 256));
    }
}

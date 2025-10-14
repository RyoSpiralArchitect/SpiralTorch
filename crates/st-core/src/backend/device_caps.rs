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

    /// Recommended workgroup size for a particular problem size.
    ///
    /// The logic prefers to scale with the native subgroup width while making
    /// sure the group size never exceeds either the advertised maximum or the
    /// number of rows that the kernel needs to process.
    pub fn recommended_workgroup(&self, rows: u32) -> u32 {
        let lanes = self.lane_width.max(1);
        let mut target = if self.subgroup {
            lanes.saturating_mul(8)
        } else {
            lanes.saturating_mul(4)
        };
        target = target.max(64).min(self.max_workgroup.max(64));

        if rows > 0 && rows < target {
            // Align the workgroup with the hardware lane width so we keep
            // coalesced accesses while avoiding oversubscribing the SM.
            let align = lanes;
            let aligned_rows = ((rows + align - 1) / align) * align;
            aligned_rows.max(align).max(64).min(target)
        } else {
            target
        }
    }

    /// Recommended tile and compaction tile for the given column count.
    ///
    /// The heuristic aligns the tile to the hardware lane width and shrinks it
    /// if shared memory pressure would make the default tile unrealistic.
    pub fn recommended_tiles(&self, cols: u32) -> (u32, u32) {
        const TILE_OPTIONS: [u32; 7] = [256, 512, 1024, 2048, 4096, 8192, 16384];
        let lanes = self.lane_width.max(1);
        let mut tile = TILE_OPTIONS
            .into_iter()
            .filter(|opt| cols >= *opt)
            .last()
            .unwrap_or(256);

        // Respect the lane width and workgroup limits.
        tile = align_up(tile, lanes);
        tile = tile.min(self.max_workgroup.max(lanes).saturating_mul(lanes));
        tile = tile.max(align_up(128, lanes));

        if let Some(shared) = self.shared_mem_per_workgroup {
            // Assume Top-K stores approximately two fp32 values per lane in
            // shared memory. Stay conservative to avoid spilling.
            let approx_bytes_per_lane = 8;
            let denom = approx_bytes_per_lane.max(1) * lanes.max(1);
            if denom > 0 {
                let capacity = (shared / denom).max(lanes);
                tile = tile
                    .min(align_up(capacity, lanes).max(lanes))
                    .max(align_up(128, lanes));
            }
        }

        let ctile_base = if cols > 65_536 {
            1024
        } else if cols > 8_192 {
            512
        } else {
            256
        };
        let mut ctile = ctile_base.min(tile);

        if let Some(shared) = self.shared_mem_per_workgroup {
            // Compaction uses wider staging buffers, so give it extra headroom
            // when the shared memory budget is tight.
            if shared < 48 * 1024 {
                ctile = ctile.min(512);
            }
        }

        ctile = align_up(ctile.max(128), lanes).min(tile.max(lanes));

        (tile, ctile)
    }

    /// Preferred inner K loop depth for the given reduction size.
    pub fn preferred_k_loop(&self, k: u32) -> u32 {
        let lanes = self.lane_width.max(1);
        if k > lanes * 4 {
            32
        } else if k > lanes * 2 {
            16
        } else {
            8
        }
    }

    /// Preferred compaction channel count given the column pressure.
    pub fn preferred_channel(&self, cols: u32) -> u32 {
        if cols <= 16_384 {
            0
        } else if self.backend == BackendKind::Cpu {
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
    pub fn prefers_two_stage(&self, rows: u32, cols: u32, k: u32) -> bool {
        let col_heavy = cols > 32_768;
        let k_heavy = k > self.lane_width.max(1) * 4;
        let small_rows = rows < 256;
        match self.backend {
            BackendKind::Wgpu => col_heavy || (k_heavy && !small_rows),
            BackendKind::Cuda | BackendKind::Hip => {
                col_heavy || (k_heavy && rows > self.lane_width.max(1) * 8)
            }
            BackendKind::Cpu => col_heavy && rows > 512,
        }
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
        assert!(caps.prefers_two_stage(1_024, 40_000, 64));
        assert!(caps.prefers_two_stage(4_096, 2_000, 256));
        assert!(!caps.prefers_two_stage(128, 10_000, 64));
    }

    #[test]
    fn recommended_workgroup_tracks_rows() {
        let caps = DeviceCaps::cuda(32, 1024, Some(96 * 1024));
        assert_eq!(caps.recommended_workgroup(4096), 256);
        assert_eq!(caps.recommended_workgroup(96), 96);
    }

    #[test]
    fn recommended_tiles_obey_shared_memory_limits() {
        let caps = DeviceCaps::cuda(32, 1024, Some(32 * 1024));
        let (tile, ctile) = caps.recommended_tiles(200_000);
        assert!(tile <= 512);
        assert!(ctile <= tile);
        let wide = DeviceCaps::cuda(32, 1024, Some(96 * 1024));
        let (tile_wide, _) = wide.recommended_tiles(200_000);
        assert!(tile_wide > tile);
    }

    #[test]
    fn preferred_helpers_follow_lane_width() {
        let caps = DeviceCaps::hip(64, 1024, Some(64 * 1024));
        assert_eq!(caps.preferred_k_loop(64), 8);
        assert_eq!(caps.preferred_k_loop(256), 16);
        assert_eq!(caps.preferred_k_loop(512), 32);
        assert_eq!(caps.preferred_channel(8_000), 0);
        assert_eq!(caps.preferred_channel(64_000), 8_192);
    }
}

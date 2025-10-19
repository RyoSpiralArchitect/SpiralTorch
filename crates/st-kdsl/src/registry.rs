// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Autotuning registry helpers and telemetry structures used by SpiralTorch's
//! WGPU-first roadmap.  The utilities here provide a common key format for
//! caching tuned kernels across driver updates and store dispatch metrics that
//! downstream schedulers can analyse.

use std::fmt;

use crate::tile::TileConfig;

/// Describes the hardware identity relevant for kernel autotuning.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DeviceProfile {
    pub vendor: String,
    pub device_id: u32,
    pub subgroup_size: u32,
    /// Shared memory per workgroup expressed in kilobytes.
    pub shared_kb: u32,
    pub driver: String,
}

impl DeviceProfile {
    pub fn new(
        vendor: impl Into<String>,
        device_id: u32,
        subgroup_size: u32,
        shared_kb: u32,
        driver: impl Into<String>,
    ) -> Self {
        Self {
            vendor: vendor.into(),
            device_id,
            subgroup_size,
            shared_kb,
            driver: driver.into(),
        }
    }

    fn encode_component(value: &str) -> String {
        value
            .chars()
            .map(|ch| match ch {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' => ch,
                ' ' => '-',
                _ => '_',
            })
            .collect::<String>()
    }

    pub fn encode(&self) -> String {
        format!(
            "{}|{:04x}|{}|{}|{}",
            Self::encode_component(&self.vendor),
            self.device_id,
            self.subgroup_size,
            self.shared_kb,
            Self::encode_component(&self.driver)
        )
    }
}

/// Identifies a shader or kernel variant that participates in autotuning.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelProfile {
    pub shader_revision: u64,
    pub op_signature: String,
}

impl KernelProfile {
    pub fn new(shader_revision: u64, op_signature: impl Into<String>) -> Self {
        Self {
            shader_revision,
            op_signature: op_signature.into(),
        }
    }

    pub fn encode(&self) -> String {
        format!("{:016x}|{}", self.shader_revision, self.op_signature)
    }
}

/// Compound key that joins the device and kernel identity.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AutotuneKey {
    pub device: DeviceProfile,
    pub kernel: KernelProfile,
}

impl AutotuneKey {
    pub fn new(device: DeviceProfile, kernel: KernelProfile) -> Self {
        Self { device, kernel }
    }

    pub fn encode(&self) -> String {
        format!("{}|{}", self.device.encode(), self.kernel.encode())
    }
}

impl fmt::Display for AutotuneKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.encode())
    }
}

/// Per-dispatch telemetry captured by the runtime.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TelemetrySample {
    /// Effective throughput in tera-FLOPs per second.
    pub tflops: f32,
    /// Observed memory bandwidth in gigabytes per second.
    pub bandwidth_gbps: f32,
    /// Occupancy estimate in the range [0, 1].
    pub occupancy: f32,
    /// Tile picked by the autotuner, if known.
    pub tile: Option<TileConfig>,
    /// Whether this dispatch triggered a regression fallback.
    pub regress: bool,
}

impl TelemetrySample {
    pub fn new(
        tflops: f32,
        bandwidth_gbps: f32,
        occupancy: f32,
        tile: Option<TileConfig>,
        regress: bool,
    ) -> Self {
        Self {
            tflops,
            bandwidth_gbps,
            occupancy,
            tile,
            regress,
        }
    }

    pub fn is_regression(&self) -> bool {
        self.regress
    }
}

/// Aggregate statistics computed across telemetry samples.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TelemetrySummary {
    pub avg_tflops: f32,
    pub avg_bandwidth_gbps: f32,
    pub avg_occupancy: f32,
    /// Fraction of dispatches that regressed.
    pub regression_rate: f32,
}

/// Rolling log of telemetry samples.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TelemetryLog {
    entries: Vec<TelemetrySample>,
}

impl TelemetryLog {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn push(&mut self, sample: TelemetrySample) {
        self.entries.push(sample);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &TelemetrySample> {
        self.entries.iter()
    }

    pub fn best_by_tflops(&self) -> Option<&TelemetrySample> {
        self.entries.iter().max_by(|a, b| {
            a.tflops
                .partial_cmp(&b.tflops)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    pub fn summary(&self) -> Option<TelemetrySummary> {
        if self.entries.is_empty() {
            return None;
        }
        let mut total_tflops = 0.0f32;
        let mut total_bw = 0.0f32;
        let mut total_occ = 0.0f32;
        let mut regressions = 0usize;
        for entry in &self.entries {
            total_tflops += entry.tflops;
            total_bw += entry.bandwidth_gbps;
            total_occ += entry.occupancy;
            if entry.regress {
                regressions += 1;
            }
        }
        let count = self.entries.len() as f32;
        Some(TelemetrySummary {
            avg_tflops: total_tflops / count,
            avg_bandwidth_gbps: total_bw / count,
            avg_occupancy: total_occ / count,
            regression_rate: regressions as f32 / count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_device_profile() {
        let device = DeviceProfile::new("NVIDIA Corporation", 0x2684, 32, 128, "551.86");
        assert_eq!(device.encode(), "NVIDIA-Corporation|2684|32|128|551.86");
    }

    #[test]
    fn encodes_autotune_key() {
        let device = DeviceProfile::new("Apple", 0x1234, 32, 64, "14.4");
        let kernel = KernelProfile::new(3, "gemm_f16_tile64x128");
        let key = AutotuneKey::new(device, kernel);
        assert_eq!(
            key.encode(),
            "Apple|1234|32|64|14.4|0000000000000003|gemm_f16_tile64x128"
        );
        assert_eq!(format!("{}", key), key.encode());
    }

    #[test]
    fn aggregates_telemetry() {
        let tile = TileConfig {
            workgroup: (64, 2),
            tile_m: 128,
            tile_n: 64,
            tile_k: 32,
            vector: 4,
            stages: 2,
        };
        let mut log = TelemetryLog::new();
        log.push(TelemetrySample::new(85.0, 900.0, 0.78, Some(tile), false));
        log.push(TelemetrySample::new(82.0, 880.0, 0.75, None, true));
        let summary = log.summary().expect("summary");
        assert!((summary.avg_tflops - 83.5).abs() < 1e-3);
        assert!((summary.avg_bandwidth_gbps - 890.0).abs() < 1e-3);
        assert!((summary.avg_occupancy - 0.765).abs() < 1e-3);
        assert!((summary.regression_rate - 0.5).abs() < 1e-6);
        let best = log.best_by_tflops().unwrap();
        assert_eq!(best.tflops, 85.0);
        assert_eq!(best.tile.unwrap().tile_m, 128);
    }
}

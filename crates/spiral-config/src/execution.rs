//! Shared execution-policy configuration for SpiralTorch runtimes.

/// Environment variable that disables accelerator-to-CPU fallback when enabled.
pub const STRICT_ACCELERATOR_ENV: &str = "SPIRALTORCH_STRICT_GPU";

/// Environment variable controlling the minimum tensor size for WGPU utility kernels.
pub const TENSOR_UTIL_WGPU_MIN_VALUES_ENV: &str = "SPIRALTORCH_TENSOR_UTIL_WGPU_MIN_VALUES";

/// Default minimum tensor size for routing utility kernels to WGPU.
pub const DEFAULT_TENSOR_UTIL_WGPU_MIN_VALUES: usize = 1024;

/// Whether an accelerator execution failure may fall back to a software path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AcceleratorFallback {
    /// Preserve availability by allowing a compatible software fallback.
    #[default]
    Allow,
    /// Surface accelerator failures instead of silently changing execution backends.
    Forbid,
}

impl AcceleratorFallback {
    /// Reads the process-level fallback contract.
    pub fn from_env() -> Self {
        Self::from_value(std::env::var(STRICT_ACCELERATOR_ENV).ok().as_deref())
    }

    /// Parses the stable boolean vocabulary used by SpiralTorch environment flags.
    pub fn from_value(value: Option<&str>) -> Self {
        if matches!(value, Some("1" | "true" | "TRUE")) {
            Self::Forbid
        } else {
            Self::Allow
        }
    }

    /// Returns true when software fallback is permitted.
    pub const fn allows_fallback(self) -> bool {
        matches!(self, Self::Allow)
    }

    /// Returns true when accelerator execution is required.
    pub const fn is_strict(self) -> bool {
        matches!(self, Self::Forbid)
    }

    /// Stable label for telemetry and cross-language reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Forbid => "forbid",
        }
    }
}

/// Process-level inputs used to build a deterministic execution plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionConfig {
    pub accelerator_fallback: AcceleratorFallback,
    pub tensor_util_wgpu_min_values: usize,
}

impl ExecutionConfig {
    /// Builds an explicit configuration without consulting global process state.
    pub const fn new(
        accelerator_fallback: AcceleratorFallback,
        tensor_util_wgpu_min_values: usize,
    ) -> Self {
        Self {
            accelerator_fallback,
            tensor_util_wgpu_min_values,
        }
    }

    /// Captures the process environment once so downstream execution stays stable.
    pub fn from_env() -> Self {
        Self {
            accelerator_fallback: AcceleratorFallback::from_env(),
            tensor_util_wgpu_min_values: std::env::var(TENSOR_UTIL_WGPU_MIN_VALUES_ENV)
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_TENSOR_UTIL_WGPU_MIN_VALUES),
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self::new(
            AcceleratorFallback::Allow,
            DEFAULT_TENSOR_UTIL_WGPU_MIN_VALUES,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_parser_preserves_the_existing_flag_contract() {
        for value in [Some("1"), Some("true"), Some("TRUE")] {
            assert_eq!(
                AcceleratorFallback::from_value(value),
                AcceleratorFallback::Forbid
            );
        }
        for value in [None, Some("0"), Some("false"), Some("yes"), Some("True")] {
            assert_eq!(
                AcceleratorFallback::from_value(value),
                AcceleratorFallback::Allow
            );
        }
    }

    #[test]
    fn default_config_is_non_strict_and_uses_the_shared_utility_threshold() {
        let config = ExecutionConfig::default();
        assert!(config.accelerator_fallback.allows_fallback());
        assert_eq!(
            config.tensor_util_wgpu_min_values,
            DEFAULT_TENSOR_UTIL_WGPU_MIN_VALUES
        );
    }
}

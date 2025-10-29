use rand::{rngs::StdRng, SeedableRng};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{OnceLock, RwLock};

/// Unified deterministic runtime configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeterminismConfig {
    /// Whether deterministic execution is enabled globally.
    pub enabled: bool,
    /// Base seed used to derive per-component seeds.
    pub base_seed: u64,
    /// If true the runtime should clamp schedulers to deterministic stepping.
    pub fix_scheduler: bool,
    /// If true reductions should run sequentially to ensure stable ordering.
    pub fix_reduction: bool,
}

impl DeterminismConfig {
    /// Builds a configuration snapshot from environment variables.
    fn from_env() -> Self {
        let enabled = std::env::var("SPIRAL_DETERMINISTIC")
            .ok()
            .map(|v| match v.as_str() {
                "0" | "false" | "False" | "off" | "OFF" => false,
                _ => true,
            })
            .unwrap_or(false);

        let base_seed = std::env::var("SPIRAL_DETERMINISTIC_SEED")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(42);

        let fix_scheduler = std::env::var("SPIRAL_DETERMINISTIC_SCHEDULER")
            .ok()
            .map(|v| matches!(v.as_str(), "1" | "true" | "True" | "on" | "ON"))
            .unwrap_or(enabled);

        let fix_reduction = std::env::var("SPIRAL_DETERMINISTIC_REDUCTION")
            .ok()
            .map(|v| matches!(v.as_str(), "1" | "true" | "True" | "on" | "ON"))
            .unwrap_or(enabled);

        Self {
            enabled,
            base_seed,
            fix_scheduler,
            fix_reduction,
        }
    }

    /// Derives a deterministic seed for a given component label.
    pub fn seed_for<L: Hash>(&self, label: L) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.base_seed.hash(&mut hasher);
        label.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Clone, Debug)]
struct DeterminismState {
    config: DeterminismConfig,
}

impl DeterminismState {
    fn new(config: DeterminismConfig) -> Self {
        Self { config }
    }
}

static STATE: OnceLock<RwLock<DeterminismState>> = OnceLock::new();

fn state() -> &'static RwLock<DeterminismState> {
    STATE.get_or_init(|| {
        let cfg = DeterminismConfig::from_env();
        apply_process_hints(&cfg, None);
        RwLock::new(DeterminismState::new(cfg))
    })
}

/// Returns the lazily initialised deterministic configuration.
pub fn config() -> DeterminismConfig {
    state().read().unwrap().config.clone()
}

/// Overrides the deterministic configuration. Intended for tests.
pub fn configure(cfg: DeterminismConfig) -> DeterminismConfig {
    let lock = state();
    let mut guard = lock.write().unwrap();
    apply_process_hints(&cfg, Some(&guard.config));
    guard.config = cfg.clone();
    cfg
}

fn apply_process_hints(cfg: &DeterminismConfig, previous: Option<&DeterminismConfig>) {
    let reduction_active = cfg.enabled && cfg.fix_reduction;
    if reduction_active {
        // Hint Rayon before any pools are built. This is best-effort; if a pool
        // already exists the environment change is harmless but ineffectual.
        std::env::set_var("RAYON_NUM_THREADS", "1");
    } else if previous.map_or(false, |prev| prev.enabled && prev.fix_reduction) {
        std::env::remove_var("RAYON_NUM_THREADS");
    }

    if cfg.enabled {
        std::env::set_var("SPIRAL_DETERMINISTIC_ACTIVE", "1");
    } else if previous.map_or(false, |prev| prev.enabled) {
        std::env::remove_var("SPIRAL_DETERMINISTIC_ACTIVE");
    }
}

/// Returns a RNG derived from the provided label. When determinism is disabled
/// this falls back to a random seed from the operating system.
pub fn rng_from_label(label: &str) -> StdRng {
    let cfg = config();
    if cfg.enabled {
        StdRng::seed_from_u64(cfg.seed_for(label))
    } else {
        StdRng::from_entropy()
    }
}

/// Returns a RNG seeded from an optional explicit seed, respecting deterministic
/// overrides when the seed is not provided.
pub fn rng_from_optional(seed: Option<u64>, label: &str) -> StdRng {
    match seed {
        Some(value) => StdRng::seed_from_u64(value),
        None => rng_from_label(label),
    }
}

/// Returns whether reductions should be forced to run sequentially.
pub fn lock_reduction_order() -> bool {
    let cfg = config();
    cfg.enabled && cfg.fix_reduction
}

/// Returns whether schedulers should clamp their adaptive behaviour.
pub fn lock_scheduler() -> bool {
    let cfg = config();
    cfg.enabled && cfg.fix_scheduler
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
    use std::sync::{Mutex, OnceLock};

    #[derive(Clone)]
    struct ConfigGuard(DeterminismConfig);

    impl ConfigGuard {
        fn snapshot() -> Self {
            Self(config())
        }
    }

    impl Drop for ConfigGuard {
        fn drop(&mut self) {
            configure(self.0.clone());
        }
    }

    fn with_env(vars: &[(&str, Option<&str>)], test: impl FnOnce()) {
        static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
        let _lock = GUARD.get_or_init(|| Mutex::new(())).lock().unwrap();

        let snapshot: Vec<(String, Option<String>)> = vars
            .iter()
            .map(|(key, value)| {
                let previous = std::env::var(key).ok();
                match value {
                    Some(val) => std::env::set_var(key, val),
                    None => std::env::remove_var(key),
                }
                ((*key).to_string(), previous)
            })
            .collect();

        let result = catch_unwind(AssertUnwindSafe(test));

        for (key, value) in snapshot {
            match value {
                Some(val) => std::env::set_var(&key, val),
                None => std::env::remove_var(&key),
            }
        }

        if let Err(err) = result {
            resume_unwind(err);
        }
    }

    #[test]
    fn defaults_disable_determinism() {
        with_env(
            &[
                ("SPIRAL_DETERMINISTIC", None),
                ("SPIRAL_DETERMINISTIC_SEED", None),
                ("SPIRAL_DETERMINISTIC_SCHEDULER", None),
                ("SPIRAL_DETERMINISTIC_REDUCTION", None),
            ],
            || {
                let cfg = DeterminismConfig::from_env();
                assert!(!cfg.enabled);
                assert_eq!(cfg.base_seed, 42);
                assert!(!cfg.fix_scheduler);
                assert!(!cfg.fix_reduction);
            },
        );
    }

    #[test]
    fn explicit_enables_override_defaults() {
        with_env(
            &[
                ("SPIRAL_DETERMINISTIC", Some("1")),
                ("SPIRAL_DETERMINISTIC_SEED", Some("1337")),
                ("SPIRAL_DETERMINISTIC_SCHEDULER", Some("0")),
                ("SPIRAL_DETERMINISTIC_REDUCTION", Some("true")),
            ],
            || {
                let cfg = DeterminismConfig::from_env();
                assert!(cfg.enabled);
                assert_eq!(cfg.base_seed, 1337);
                assert!(!cfg.fix_scheduler);
                assert!(cfg.fix_reduction);
            },
        );
    }

    #[test]
    fn textual_false_values_disable_flags() {
        with_env(&[("SPIRAL_DETERMINISTIC", Some("off"))], || {
            let cfg = DeterminismConfig::from_env();
            assert!(!cfg.enabled);
        });
    }

    #[test]
    fn derived_seeds_are_stable_per_label() {
        with_env(
            &[
                ("SPIRAL_DETERMINISTIC", Some("1")),
                ("SPIRAL_DETERMINISTIC_SEED", Some("99")),
            ],
            || {
                let cfg = DeterminismConfig::from_env();
                let alpha_first = cfg.seed_for("alpha");
                let alpha_second = cfg.seed_for("alpha");
                let beta = cfg.seed_for("beta");
                assert_eq!(alpha_first, alpha_second);
                assert_ne!(alpha_first, beta);
            },
        );
    }

    #[test]
    fn scheduler_and_reduction_default_to_enabled_when_unspecified() {
        with_env(
            &[
                ("SPIRAL_DETERMINISTIC", Some("1")),
                ("SPIRAL_DETERMINISTIC_SCHEDULER", None),
                ("SPIRAL_DETERMINISTIC_REDUCTION", None),
            ],
            || {
                let cfg = DeterminismConfig::from_env();
                assert!(cfg.fix_scheduler);
                assert!(cfg.fix_reduction);
            },
        );
    }

    #[test]
    fn configure_overrides_runtime_state_and_hints() {
        let _guard = ConfigGuard::snapshot();

        configure(DeterminismConfig {
            enabled: true,
            base_seed: 7,
            fix_scheduler: false,
            fix_reduction: true,
        });

        let cfg = config();
        assert!(cfg.enabled);
        assert_eq!(cfg.base_seed, 7);
        assert!(matches!(std::env::var("SPIRAL_DETERMINISTIC_ACTIVE"), Ok(val) if val == "1"));
        assert!(matches!(std::env::var("RAYON_NUM_THREADS"), Ok(val) if val == "1"));

        configure(DeterminismConfig {
            enabled: false,
            base_seed: 123,
            fix_scheduler: false,
            fix_reduction: false,
        });

        let cfg = config();
        assert!(!cfg.enabled);
        assert!(std::env::var("SPIRAL_DETERMINISTIC_ACTIVE").is_err());
        assert!(std::env::var("RAYON_NUM_THREADS").is_err());
    }
}

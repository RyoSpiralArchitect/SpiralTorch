use std::time::SystemTime;
#[cfg(any(target_arch = "wasm32", test))]
use std::time::{Duration, UNIX_EPOCH};

/// Host wall time for ecosystem observations, never a monotonic deadline clock.
pub(crate) fn system_time_now() -> SystemTime {
    #[cfg(not(target_arch = "wasm32"))]
    {
        SystemTime::now()
    }
    #[cfg(target_arch = "wasm32")]
    {
        // std::time::SystemTime::now() aborts on wasm32-unknown-unknown.
        system_time_from_unix_millis(js_sys::Date::now())
            .expect("JavaScript Date.now() must return a representable timestamp")
    }
}

#[cfg(any(target_arch = "wasm32", test))]
fn system_time_from_unix_millis(millis: f64) -> Option<SystemTime> {
    let duration = Duration::try_from_secs_f64(millis.abs() / 1_000.0).ok()?;
    if millis < 0.0 {
        UNIX_EPOCH.checked_sub(duration)
    } else {
        UNIX_EPOCH.checked_add(duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_milliseconds_preserve_epoch_sign_and_fraction() {
        assert_eq!(system_time_from_unix_millis(0.0), Some(UNIX_EPOCH));
        assert_eq!(system_time_from_unix_millis(-0.0), Some(UNIX_EPOCH));
        let delta = Duration::from_micros(1_000_250);
        assert_eq!(
            system_time_from_unix_millis(1_000.25),
            Some(UNIX_EPOCH + delta)
        );
        assert_eq!(
            system_time_from_unix_millis(-1_000.25),
            Some(UNIX_EPOCH - delta)
        );
        for millis in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            assert!(system_time_from_unix_millis(millis).is_none());
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_clock_retains_system_time_semantics() {
        let before = SystemTime::now();
        let actual = system_time_now();
        let after = SystemTime::now();
        assert!(before <= actual && actual <= after);
    }
}

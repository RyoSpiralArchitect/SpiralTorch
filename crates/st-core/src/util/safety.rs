use std::sync::Once;
static INIT: Once = Once::new();

/// Install a panic hook that suppresses backtrace when ST_NO_TRACEBACK=1.
pub fn install_panic_sanitizer() {
    INIT.call_once(|| {
        let no_tb = std::env::var("ST_NO_TRACEBACK").ok().map(|v| v=="1").unwrap_or(false);
        if no_tb {
            std::panic::set_hook(Box::new(|info| {
                eprintln!("[SpiralTorch] Panic: {}", info);
                eprintln!("(Set ST_NO_TRACEBACK=0 for detailed backtrace)");
            }));
        }
    });
}

/// Guard a computation; on error, produce fallback value.
#[macro_export]
macro_rules! guard_or {
    ($expr:expr, $fallback:expr) => { match std::panic::catch_unwind(|| { $expr }) {
        Ok(v) => v,
        Err(_) => { $fallback }
    }};
}

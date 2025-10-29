use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use std::io::IsTerminal;

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Registry};

static INITIALISED: OnceLock<()> = OnceLock::new();
static CHROME_GUARD: OnceLock<Mutex<Option<tracing_chrome::FlushGuard>>> = OnceLock::new();

/// Configures the global tracing subscriber.
pub fn init_tracing() -> Result<(), InitError> {
    INITIALISED
        .set(())
        .map_err(|_| InitError::AlreadyInitialised)?;

    let ansi = std::io::stdout().is_terminal();

    match chrome_trace_path()? {
        Some(path) => {
            let filter =
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            let fmt_layer = tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_ansi(ansi);
            let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
                .file(path)
                .include_args(true)
                .build();
            if let Some(cell) = CHROME_GUARD.get() {
                if let Ok(mut slot) = cell.lock() {
                    *slot = Some(guard);
                }
            } else {
                let _ = CHROME_GUARD.set(Mutex::new(Some(guard)));
            }
            Registry::default()
                .with(filter)
                .with(fmt_layer)
                .with(chrome_layer)
                .init();
        }
        None => {
            let filter =
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
            let fmt_layer = tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_ansi(ansi);
            Registry::default().with(filter).with(fmt_layer).init();
        }
    }

    Ok(())
}

fn chrome_trace_path() -> Result<Option<PathBuf>, InitError> {
    match std::env::var("SPIRAL_TRACE_CHROME") {
        Ok(raw) if !raw.trim().is_empty() => Ok(Some(PathBuf::from(raw))),
        Ok(_) => Ok(None),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(err) => Err(InitError::Env(err)),
    }
}

/// Errors emitted when configuring the tracing subscriber.
#[derive(Debug, thiserror::Error)]
pub enum InitError {
    #[error("tracing has already been initialised")]
    AlreadyInitialised,
    #[error("failed to read SPIRAL_TRACE_CHROME: {0}")]
    Env(std::env::VarError),
}

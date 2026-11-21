//! Shared no-clobber commit boundary; trainers still own their math and payloads.
use super::CausalLmResult;
use st_core::runtime::AtomicNewDestination;
use std::{fs, io, path::Path};

pub(super) const MAX_INTERVAL: u64 = 1024;

/// A complete checkpoint was installed but publication did not finish.
/// The live cursor did not advance. Verify any retained file with this pin
/// before using it for an explicit recovery.
#[derive(Debug)]
pub struct CheckpointPublicationUncertain {
    checkpoint_id: String,
    source: io::Error,
}

impl CheckpointPublicationUncertain {
    pub(super) fn from_installed_file(checkpoint_id: String, source: io::Error) -> Self {
        Self {
            checkpoint_id,
            source,
        }
    }

    pub fn checkpoint_id(&self) -> &str {
        &self.checkpoint_id
    }
}

impl std::fmt::Display for CheckpointPublicationUncertain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "checkpoint installed but publication incomplete (pin {}): {}",
            self.checkpoint_id, self.source
        )
    }
}

impl std::error::Error for CheckpointPublicationUncertain {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

pub(super) fn save_with_status<P: AsRef<[u8]>>(
    destination: &AtomicNewDestination,
    (bytes, pin): (P, String),
) -> CausalLmResult<String> {
    match destination.publish(bytes.as_ref()) {
        Ok(()) => Ok(pin),
        Err(source) if source.destination_installed() => Err(Box::new(
            CheckpointPublicationUncertain::from_installed_file(pin, source.into_io_error()),
        )),
        Err(source) => Err(source.into_io_error().into()),
    }
}

/// A frozen checkpoint and destination. Dropping it does not publish a file.
pub struct CausalLmPreparedCheckpointFile<P> {
    snapshot: (P, String),
    destination: AtomicNewDestination,
}

impl<P: AsRef<[u8]>> CausalLmPreparedCheckpointFile<P> {
    pub(super) fn stage(
        path: &Path,
        snapshot: impl FnOnce() -> CausalLmResult<(P, String)>,
    ) -> CausalLmResult<Self> {
        let destination = AtomicNewDestination::admit(path)?;
        Ok(Self {
            destination,
            snapshot: snapshot()?,
        })
    }

    pub fn checkpoint_id(&self) -> &str {
        &self.snapshot.1
    }

    /// Install the frozen checkpoint without overwriting an existing name.
    pub fn publish(self) -> CausalLmResult<String> {
        self.publish_with(save_with_status)
    }

    fn publish_with(
        self,
        publish: impl FnOnce(&AtomicNewDestination, (P, String)) -> CausalLmResult<String>,
    ) -> CausalLmResult<String> {
        publish(&self.destination, self.snapshot)
    }
}

pub(super) struct PreparedCheckpoint<'a, S, R, P> {
    live: &'a mut S,
    next: Option<S>,
    reports: Vec<R>,
    file: CausalLmPreparedCheckpointFile<P>,
}

impl<'a, S, R, P: AsRef<[u8]>> PreparedCheckpoint<'a, S, R, P> {
    pub(super) fn stage(
        live: &'a mut S,
        path: &Path,
        prepare: impl FnOnce(&S) -> CausalLmResult<(Option<S>, Vec<R>, (P, String))>,
    ) -> CausalLmResult<Self> {
        let destination = AtomicNewDestination::admit(path)?;
        let (next, reports, snapshot) = prepare(live)?;
        Ok(Self {
            live,
            next,
            reports,
            file: CausalLmPreparedCheckpointFile {
                snapshot,
                destination,
            },
        })
    }

    pub(super) fn reports(&self) -> &[R] {
        &self.reports
    }

    pub(super) fn checkpoint_id(&self) -> &str {
        self.file.checkpoint_id()
    }

    pub(super) fn publish(self) -> CausalLmResult<(Vec<R>, String)> {
        self.publish_with(save_with_status)
    }

    pub(super) fn publish_with(
        self,
        publish: impl FnOnce(&AtomicNewDestination, (P, String)) -> CausalLmResult<String>,
    ) -> CausalLmResult<(Vec<R>, String)> {
        let pin = self.file.publish_with(publish)?;
        if let Some(next) = self.next {
            *self.live = next;
        }
        Ok((self.reports, pin))
    }
}

pub(super) fn count(cursor: u64, total: u64, stop: u64) -> CausalLmResult<usize> {
    stop.checked_sub(cursor)
        .filter(|n| (1..=MAX_INTERVAL).contains(n) && stop <= total)
        .map(|n| n as usize)
        .ok_or_else(|| {
            "checkpoint interval must advance 1..=1024 updates within the trainer".into()
        })
}

pub(super) fn admit_new(path: &Path) -> io::Result<()> {
    // Advisory only; publication still atomically rejects racing names.
    match fs::symlink_metadata(path) {
        Ok(_) => return Err(io::ErrorKind::AlreadyExists.into()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !fs::metadata(parent)?.is_dir() {
        return Err(io::ErrorKind::NotADirectory.into());
    }
    Ok(())
}

pub(super) fn save<P: AsRef<[u8]>>(
    path: &Path,
    (payload, pin): (P, String),
) -> CausalLmResult<String> {
    st_core::runtime::atomic_create_new(path, payload.as_ref())?;
    Ok(pin)
}

pub(super) fn transaction<S: Clone, T>(
    state: &mut S,
    prepare_and_publish: impl FnOnce(&mut S) -> CausalLmResult<T>,
) -> CausalLmResult<T> {
    let mut next = state.clone();
    let result = prepare_and_publish(&mut next)?;
    *state = next;
    Ok(result)
}

pub(super) fn steps<S, R>(
    state: &mut S,
    count: usize,
    step: impl Fn(&mut S) -> CausalLmResult<Option<R>>,
) -> CausalLmResult<Vec<R>> {
    (0..count)
        .map(|_| step(state)?.ok_or_else(|| "unexpected end of training segment".into()))
        .collect()
}

pub(super) fn advance_new<S: Clone, R, P: AsRef<[u8]>>(
    state: &mut S,
    count: usize,
    path: &Path,
    step: impl Fn(&mut S) -> CausalLmResult<Option<R>>,
    snapshot: impl FnOnce(&S) -> CausalLmResult<(P, String)>,
) -> CausalLmResult<(Vec<R>, String)> {
    advance_new_with(state, count, path, step, snapshot, save)
}

pub(super) fn advance_new_with<S: Clone, R, P: AsRef<[u8]>>(
    state: &mut S,
    count: usize,
    path: &Path,
    step: impl Fn(&mut S) -> CausalLmResult<Option<R>>,
    snapshot: impl FnOnce(&S) -> CausalLmResult<(P, String)>,
    publish: impl FnOnce(&Path, (P, String)) -> CausalLmResult<String>,
) -> CausalLmResult<(Vec<R>, String)> {
    admit_new(path)?;
    transaction(state, |next| {
        let reports = steps(next, count, step)?;
        Ok((reports, publish(path, snapshot(next)?)?))
    })
}

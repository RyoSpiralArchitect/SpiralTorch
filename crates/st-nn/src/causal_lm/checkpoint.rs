//! Shared no-clobber commit boundary; trainers still own their math and payloads.
use super::CausalLmResult;
use std::{fs, io, path::Path};

pub(super) const MAX_INTERVAL: u64 = 1024;

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
    admit_new(path)?;
    transaction(state, |next| {
        let reports = steps(next, count, step)?;
        Ok((reports, save(path, snapshot(next)?)?))
    })
}

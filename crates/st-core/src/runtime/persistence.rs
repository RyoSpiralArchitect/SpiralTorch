// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Staged file publication for Rust-owned runtime state.

#[cfg(unix)]
use std::fs::File;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

const MAX_TEMPFILE_ATTEMPTS: u32 = 128;

/// Write a complete byte snapshot and atomically replace `path`.
///
/// Callers own parent-directory creation and payload validation. A successful
/// return means the temporary file was flushed and synced, the destination was
/// atomically replaced, and the parent directory was synced on Unix.
pub(crate) fn atomic_write(path: &Path, bytes: &[u8]) -> io::Result<()> {
    publish_with(path, false, |f| f.write_all(bytes))
}

/// Publish complete bytes at a new name in a caller-controlled directory.
/// Existing names, including symlinks, are not replaced. Unix requires
/// hard-link support; an error after publication may leave the complete file.
pub fn atomic_create_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    publish_with(path, true, |f| f.write_all(bytes))
}

fn publish_with(
    path: &Path,
    create_only: bool,
    write: impl FnOnce(&mut fs::File) -> io::Result<()>,
) -> io::Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "runtime-state".into());
    let mut last_collision = None;
    for nonce in 0..MAX_TEMPFILE_ATTEMPTS {
        let temporary = parent.join(format!(".{file_name}.{}.{}.tmp", std::process::id(), nonce));
        let mut options = OpenOptions::new();
        options.create_new(true).write(true);
        #[cfg(unix)]
        if create_only {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = match options.open(&temporary) {
            Ok(file) => file,
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                last_collision = Some(source);
                continue;
            }
            Err(source) => return Err(source),
        };
        let result = (|| {
            write(&mut file)?;
            file.flush()?;
            file.sync_all()?;
            close_before_replace(file);
            install(&temporary, path, create_only)?;
            #[cfg(unix)]
            File::open(parent)?.sync_all()?;
            Ok::<(), io::Error>(())
        })();
        if let Err(source) = result {
            let _ = fs::remove_file(&temporary);
            return Err(source);
        }
        return Ok(());
    }
    Err(last_collision.unwrap_or_else(|| {
        io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve an atomic runtime-state file",
        )
    }))
}

fn close_before_replace(file: fs::File) {
    #[cfg(not(target_arch = "wasm32"))]
    drop(file);
    #[cfg(target_arch = "wasm32")]
    let _ = file;
}

#[cfg(not(windows))]
fn install(source: &Path, destination: &Path, create_only: bool) -> io::Result<()> {
    if create_only {
        fs::hard_link(source, destination)?;
        fs::remove_file(source)
    } else {
        fs::rename(source, destination)
    }
}

#[cfg(windows)]
fn install(source: &Path, destination: &Path, create_only: bool) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source = source
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    // Both paths are NUL-terminated and remain alive for the duration of the call.
    let replaced = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_WRITE_THROUGH
                | if create_only {
                    0
                } else {
                    MOVEFILE_REPLACE_EXISTING
                },
        )
    };
    if replaced == 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_write_never_publishes_a_checkpoint_or_changes_its_predecessor() {
        let dir = tempfile::tempdir().unwrap();
        let previous = dir.path().join("previous.json");
        let failed = dir.path().join("next.json");
        atomic_create_new(&previous, b"retained").unwrap();
        let result = publish_with(&failed, true, |f| {
            f.write_all(b"partial")?;
            assert!(!failed.exists());
            Err(io::Error::other("injected write failure"))
        });
        assert!(result.is_err());
        assert!(!failed.exists());
        assert_eq!(fs::read(previous).unwrap(), b"retained");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn concurrent_publishers_keep_one_complete_winner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint");
        let gate = std::sync::Barrier::new(2);
        let results = std::thread::scope(|s| {
            let writers = [b"first".as_slice(), b"second".as_slice()].map(|bytes| {
                let (gate, path) = (&gate, &path);
                s.spawn(move || {
                    gate.wait();
                    (bytes, atomic_create_new(path, bytes))
                })
            });
            writers.map(|w| w.join().unwrap())
        });
        let winners: Vec<_> = results.iter().filter(|(_, r)| r.is_ok()).collect();
        assert_eq!(winners.len(), 1);
        assert_eq!(fs::read(&path).unwrap(), winners[0].0);
        assert_eq!(
            results
                .iter()
                .filter(|(_, r)| r
                    .as_ref()
                    .is_err_and(|e| e.kind() == io::ErrorKind::AlreadyExists))
                .count(),
            1
        );
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[cfg(unix)]
    #[test]
    fn new_files_are_private_and_dangling_destinations_are_not_followed() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("private");
        let link = dir.path().join("dangling");
        symlink(&file, &link).unwrap();
        assert_eq!(
            atomic_create_new(&link, b"x").unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert!(!file.exists());
        atomic_create_new(&file, b"safe").unwrap();
        assert_eq!(fs::metadata(file).unwrap().permissions().mode() & 0o077, 0);
    }

    #[test]
    fn atomically_creates_and_replaces_complete_snapshots() {
        let temp = tempfile::tempdir().expect("temporary directory");
        let path = temp.path().join("state.json");
        atomic_write(&path, b"first\n").expect("first snapshot");
        assert_eq!(fs::read(&path).expect("first bytes"), b"first\n");
        atomic_write(&path, b"second\n").expect("replacement snapshot");
        assert_eq!(fs::read(&path).expect("second bytes"), b"second\n");
        assert_eq!(fs::read_dir(temp.path()).expect("directory").count(), 1);
    }
}

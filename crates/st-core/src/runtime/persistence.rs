// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

//! Shared durable file replacement for Rust-owned runtime state.

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
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let file_name = path
        .file_name()
        .map(|name| name.to_string_lossy())
        .unwrap_or_else(|| "runtime-state".into());
    let mut last_collision = None;
    for nonce in 0..MAX_TEMPFILE_ATTEMPTS {
        let temporary = parent.join(format!(".{file_name}.{}.{}.tmp", std::process::id(), nonce));
        let mut file = match OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)
        {
            Ok(file) => file,
            Err(source) if source.kind() == io::ErrorKind::AlreadyExists => {
                last_collision = Some(source);
                continue;
            }
            Err(source) => return Err(source),
        };
        let result = (|| {
            file.write_all(bytes)?;
            file.flush()?;
            file.sync_all()?;
            drop(file);
            atomic_replace(&temporary, path)?;
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

#[cfg(not(windows))]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn atomic_replace(source: &Path, destination: &Path) -> io::Result<()> {
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
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
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

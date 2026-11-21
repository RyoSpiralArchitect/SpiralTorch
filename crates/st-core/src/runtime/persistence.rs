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

mod destination;
pub use destination::AtomicNewDestination;

/// An atomic publication error with the visibility of the new destination.
#[derive(Debug)]
pub struct AtomicPublishError {
    source: io::Error,
    destination_installed: bool,
}

impl AtomicPublishError {
    /// The complete file was installed, but publication did not finish.
    pub fn destination_installed(&self) -> bool {
        self.destination_installed
    }

    pub fn into_io_error(self) -> io::Error {
        self.source
    }
}

impl From<io::Error> for AtomicPublishError {
    fn from(source: io::Error) -> Self {
        Self {
            source,
            destination_installed: false,
        }
    }
}

impl std::fmt::Display for AtomicPublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.source.fmt(f)
    }
}

impl std::error::Error for AtomicPublishError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

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

/// Publish without replacing an existing file, distinguishing post-install errors.
pub fn atomic_create_new_with_status(path: &Path, bytes: &[u8]) -> Result<(), AtomicPublishError> {
    publish_detailed(path, true, |f| f.write_all(bytes), sync_parent)
}

fn publish_with(
    path: &Path,
    create_only: bool,
    write: impl FnOnce(&mut fs::File) -> io::Result<()>,
) -> io::Result<()> {
    publish_detailed(path, create_only, write, sync_parent)
        .map_err(AtomicPublishError::into_io_error)
}

fn publish_detailed(
    path: &Path,
    create_only: bool,
    write: impl FnOnce(&mut fs::File) -> io::Result<()>,
    finish: impl FnOnce(&Path) -> io::Result<()>,
) -> Result<(), AtomicPublishError> {
    let path = std::path::absolute(path)?;
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
            Err(source) => return Err(source.into()),
        };
        let mut destination_installed = false;
        let mut owns_temporary_name = true;
        let result = (|| {
            write(&mut file)?;
            file.flush()?;
            file.sync_all()?;
            #[cfg(not(unix))]
            close_before_replace(file);
            install(&temporary, &path, create_only)?;
            destination_installed = true;
            #[cfg(unix)]
            if !create_only {
                // A rename onto a hard link to the same inode is a no-op.
                // Keep the original descriptor alive while checking the name.
                owns_temporary_name = false;
                owns_temporary_name = staging_name_refers_to_file(&temporary, &file)?;
            }
            #[cfg(not(unix))]
            {
                owns_temporary_name = cfg!(not(windows)) && create_only;
            }
            if owns_temporary_name {
                fs::remove_file(&temporary)?;
            }
            // Rename or unlink released the name; another publisher may reuse it.
            owns_temporary_name = false;
            #[cfg(unix)]
            drop(file);
            finish(parent)?;
            Ok::<(), io::Error>(())
        })();
        if let Err(source) = result {
            if owns_temporary_name {
                let _ = fs::remove_file(&temporary);
            }
            return Err(AtomicPublishError {
                source,
                destination_installed,
            });
        }
        return Ok(());
    }
    Err(last_collision
        .unwrap_or_else(|| {
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not reserve an atomic runtime-state file",
            )
        })
        .into())
}

fn sync_parent(parent: &Path) -> io::Result<()> {
    #[cfg(unix)]
    return File::open(parent)?.sync_all();
    #[cfg(not(unix))]
    {
        let _ = parent;
        Ok(())
    }
}

#[cfg(unix)]
fn staging_name_refers_to_file(path: &Path, file: &File) -> io::Result<bool> {
    use std::os::unix::fs::MetadataExt;
    match fs::symlink_metadata(path) {
        Ok(named) => {
            let staged = file.metadata()?;
            Ok((named.dev(), named.ino()) == (staged.dev(), staged.ino()))
        }
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(source) => Err(source),
    }
}

#[cfg(not(unix))]
fn close_before_replace(file: fs::File) {
    #[cfg(not(target_arch = "wasm32"))]
    drop(file);
    #[cfg(target_arch = "wasm32")]
    let _ = file;
}

#[cfg(not(windows))]
fn install(source: &Path, destination: &Path, create_only: bool) -> io::Result<()> {
    if create_only {
        fs::hard_link(source, destination)
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

    #[cfg(unix)]
    #[test]
    fn publication_reuses_the_staging_descriptor_slot_for_parent_sync() {
        const CHILD: &str = "ST_PERSISTENCE_ONE_DESCRIPTOR_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let child = std::process::Command::new("/bin/sh")
                .args(["-c", "ulimit -n 32 && exec \"$@\"", "persistence-test"])
                .arg(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    concat!(
                        module_path!(),
                        "::publication_reuses_the_staging_descriptor_slot_for_parent_sync"
                    )
                    .trim_start_matches("st_core::"),
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                child.status.success(),
                "{}",
                String::from_utf8_lossy(&child.stdout)
            );
            assert!(String::from_utf8_lossy(&child.stdout).contains("1 passed"));
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("checkpoint");
        let mut occupied = Vec::new();
        while let Ok(file) = File::open("/dev/null") {
            occupied.push(file);
            assert!(
                occupied.len() <= 32,
                "child descriptor limit was not applied"
            );
        }
        drop(occupied.pop().expect("at least one slot was available"));
        let result = atomic_create_new_with_status(&destination, b"complete");
        drop(occupied);
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(fs::read(destination).unwrap(), b"complete");
    }

    #[cfg(unix)]
    #[test]
    fn relative_publication_is_not_redirected_by_a_working_directory_change() {
        const CHILD: &str = "ST_PERSISTENCE_CHANGE_DIRECTORY_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    concat!(
                        module_path!(),
                        "::relative_publication_is_not_redirected_by_a_working_directory_change"
                    )
                    .trim_start_matches("st_core::"),
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                child.status.success(),
                "{}",
                String::from_utf8_lossy(&child.stdout)
            );
            assert!(String::from_utf8_lossy(&child.stdout).contains("1 passed"));
            return;
        }
        let initial_directory = std::env::current_dir().unwrap();
        let original = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        std::env::set_current_dir(original.path()).unwrap();
        let result = publish_detailed(
            Path::new("checkpoint"),
            true,
            |file| {
                file.write_all(b"complete")?;
                std::env::set_current_dir(elsewhere.path())
            },
            sync_parent,
        );
        std::env::set_current_dir(initial_directory).unwrap();
        assert!(result.is_ok(), "{result:?}");
        assert_eq!(
            fs::read(original.path().join("checkpoint")).unwrap(),
            b"complete"
        );
        assert_eq!(fs::read_dir(original.path()).unwrap().count(), 1);
        assert_eq!(fs::read_dir(elsewhere.path()).unwrap().count(), 0);
    }

    #[test]
    fn partial_write_never_publishes_a_checkpoint_or_changes_its_predecessor() {
        let dir = tempfile::tempdir().unwrap();
        let previous = dir.path().join("previous.json");
        let failed = dir.path().join("next.json");
        atomic_create_new(&previous, b"retained").unwrap();
        let result = publish_detailed(
            &failed,
            true,
            |f| {
                f.write_all(b"partial")?;
                assert!(!failed.exists());
                Err(io::Error::other("injected write failure"))
            },
            sync_parent,
        );
        assert!(!result.unwrap_err().destination_installed());
        assert!(!failed.exists());
        assert_eq!(fs::read(previous).unwrap(), b"retained");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    #[test]
    fn failure_after_install_reports_visibility_without_clobbering() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("next.bin");
        let error = publish_detailed(
            &dest,
            true,
            |f| f.write_all(b"complete"),
            |_| Err(io::Error::other("injected directory sync failure")),
        )
        .unwrap_err();
        assert!(error.destination_installed());
        assert_eq!(fs::read(&dest).unwrap(), b"complete");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
        let collision = atomic_create_new_with_status(&dest, b"other").unwrap_err();
        assert!(!collision.destination_installed());
        assert_eq!(fs::read(&dest).unwrap(), b"complete");
    }

    #[test]
    fn post_install_sync_failure_keeps_the_next_writers_temporary_file() {
        for create_only in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let dest = dir.path().join("checkpoint");
            if !create_only {
                fs::write(&dest, b"previous").unwrap();
            }
            let temporary = dir
                .path()
                .join(format!(".checkpoint.{}.0.tmp", std::process::id()));
            let error = publish_detailed(
                &dest,
                create_only,
                |f| f.write_all(b"installed"),
                |_| {
                    // The first publisher has released this name before syncing.
                    OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .open(&temporary)
                        .unwrap()
                        .write_all(b"next writer")
                        .unwrap();
                    Err(io::Error::other("injected directory sync failure"))
                },
            )
            .unwrap_err();
            assert!(error.destination_installed());
            assert_eq!(fs::read(&dest).unwrap(), b"installed");
            assert_eq!(fs::read(&temporary).unwrap(), b"next writer");
        }
    }

    #[cfg(unix)]
    #[test]
    fn replacement_cleans_up_a_staging_name_left_by_same_inode_rename() {
        for fail_sync in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let dest = dir.path().join("checkpoint");
            let temporary = dir
                .path()
                .join(format!(".checkpoint.{}.0.tmp", std::process::id()));
            let result = publish_detailed(
                &dest,
                false,
                |f| {
                    f.write_all(b"installed")?;
                    fs::hard_link(&temporary, &dest)
                },
                |_| {
                    if fail_sync {
                        Err(io::Error::other("injected directory sync failure"))
                    } else {
                        Ok(())
                    }
                },
            );
            if fail_sync {
                assert!(result.unwrap_err().destination_installed());
            } else {
                result.unwrap();
            }
            assert_eq!(fs::read(&dest).unwrap(), b"installed");
            assert!(!temporary.exists());
        }
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

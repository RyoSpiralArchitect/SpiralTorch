use super::AtomicPublishError;
#[cfg(unix)]
use super::MAX_TEMPFILE_ATTEMPTS;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// A new filename in a directory admitted before checkpoint computation.
///
/// The directory must be controlled by the caller. On Unix, file operations
/// are relative to the held directory, not repeated resolutions of its path.
pub struct AtomicNewDestination {
    path: PathBuf,
    #[cfg(any(unix, windows))]
    directory: fs::File,
    #[cfg(not(unix))]
    resolved_parent: PathBuf,
}

impl AtomicNewDestination {
    /// Capture the destination without creating a file or staging name.
    pub fn admit(path: &Path) -> io::Result<Self> {
        let path = std::path::absolute(path)?;
        let raw_name = path
            .as_os_str()
            .as_encoded_bytes()
            .rsplit(|&byte| byte == b'/' || (cfg!(windows) && byte == b'\\'))
            .next();
        if path.file_name().is_none() || matches!(raw_name, Some(b"" | b"." | b"..")) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "checkpoint destination must name a file",
            ));
        }
        #[cfg(windows)]
        if path.file_name().unwrap().as_encoded_bytes().contains(&b':') {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "checkpoint destination must name a file, not a drive or stream",
            ));
        }
        let parent = path.parent().expect("absolute path with filename");
        #[cfg(unix)]
        let directory = {
            use rustix::fs::{open, Mode, OFlags};
            fs::File::from(open(
                parent,
                OFlags::RDONLY | OFlags::DIRECTORY | OFlags::CLOEXEC,
                Mode::empty(),
            )?)
        };
        #[cfg(not(unix))]
        let resolved_parent = fs::canonicalize(parent)?;
        #[cfg(windows)]
        let directory = open_directory(&resolved_parent, false)?;
        let destination = Self {
            path,
            #[cfg(any(unix, windows))]
            directory,
            #[cfg(not(unix))]
            resolved_parent,
        };
        destination.check_parent()?;
        destination.check_new_name()?;
        Ok(destination)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Publish complete bytes without replacing any existing name.
    pub fn publish(&self, bytes: &[u8]) -> Result<(), AtomicPublishError> {
        self.publish_with(|file| file.write_all(bytes))
    }

    fn parent(&self) -> &Path {
        self.path.parent().expect("absolute path with filename")
    }

    fn check_parent(&self) -> io::Result<()> {
        let changed =
            || io::Error::other("checkpoint destination directory changed after admission");
        let check = || -> io::Result<bool> {
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                let admitted = self.directory.metadata()?;
                let current = fs::metadata(self.parent())?;
                Ok((admitted.dev(), admitted.ino()) == (current.dev(), current.ino()))
            }
            #[cfg(windows)]
            {
                let admitted = directory_id(&self.directory)?;
                let current = open_directory(self.parent(), true)?;
                Ok(directory_id(&current)? == admitted)
            }
            #[cfg(not(any(unix, windows)))]
            {
                Ok(fs::canonicalize(self.parent())? == self.resolved_parent)
            }
        };
        match check() {
            Ok(true) => Ok(()),
            Ok(false) => Err(changed()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Err(changed()),
            Err(error) => Err(error),
        }
    }

    fn check_new_name(&self) -> io::Result<()> {
        #[cfg(unix)]
        let occupied = rustix::fs::statat(
            &self.directory,
            self.path.file_name().unwrap(),
            rustix::fs::AtFlags::SYMLINK_NOFOLLOW,
        )
        .map(|_| ())
        .map_err(io::Error::from);
        #[cfg(not(unix))]
        let occupied =
            fs::symlink_metadata(self.resolved_parent.join(self.path.file_name().unwrap()))
                .map(|_| ());
        match occupied {
            Ok(()) => Err(io::ErrorKind::AlreadyExists.into()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error),
        }
    }

    #[cfg(unix)]
    fn publish_with(
        &self,
        write: impl FnOnce(&mut fs::File) -> io::Result<()>,
    ) -> Result<(), AtomicPublishError> {
        use rustix::fs::{linkat, openat, unlinkat, AtFlags, Mode, OFlags};
        self.check_parent()?;
        let name = self.path.file_name().unwrap();
        for nonce in 0..MAX_TEMPFILE_ATTEMPTS {
            let temporary = format!(
                ".{}.{}.{}.tmp",
                name.to_string_lossy(),
                std::process::id(),
                nonce
            );
            let mut file = match openat(
                &self.directory,
                temporary.as_str(),
                OFlags::WRONLY | OFlags::CREATE | OFlags::EXCL | OFlags::CLOEXEC,
                Mode::RUSR | Mode::WUSR,
            ) {
                Ok(fd) => fs::File::from(fd),
                Err(rustix::io::Errno::EXIST) => continue,
                Err(error) => return Err(io::Error::from(error).into()),
            };
            let mut installed = false;
            let mut owns_temporary_name = true;
            let result = (|| {
                write(&mut file)?;
                file.flush()?;
                file.sync_all()?;
                drop(file);
                self.check_parent()?;
                linkat(
                    &self.directory,
                    temporary.as_str(),
                    &self.directory,
                    name,
                    AtFlags::empty(),
                )?;
                installed = true;
                unlinkat(&self.directory, temporary.as_str(), AtFlags::empty())?;
                owns_temporary_name = false;
                self.directory.sync_all()?;
                self.check_parent()?;
                Ok::<(), io::Error>(())
            })();
            return result.map_err(|source| {
                if owns_temporary_name {
                    let _ = unlinkat(&self.directory, temporary.as_str(), AtFlags::empty());
                }
                AtomicPublishError {
                    source,
                    destination_installed: installed,
                }
            });
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not reserve an atomic runtime-state file",
        )
        .into())
    }

    #[cfg(not(unix))]
    fn publish_with(
        &self,
        write: impl FnOnce(&mut fs::File) -> io::Result<()>,
    ) -> Result<(), AtomicPublishError> {
        self.check_parent()?;
        let path = self.resolved_parent.join(self.path.file_name().unwrap());
        super::publish_detailed(&path, true, write, super::sync_parent)?;
        self.check_parent().map_err(|source| AtomicPublishError {
            source,
            destination_installed: true,
        })
    }
}

#[cfg(windows)]
fn open_directory(path: &Path, share_delete: bool) -> io::Result<fs::File> {
    use std::os::windows::fs::OpenOptionsExt;
    use windows_sys::Win32::Storage::FileSystem::{
        FILE_FLAG_BACKUP_SEMANTICS, FILE_READ_ATTRIBUTES, FILE_SHARE_DELETE, FILE_SHARE_READ,
        FILE_SHARE_WRITE,
    };
    let file = fs::OpenOptions::new()
        .access_mode(FILE_READ_ATTRIBUTES)
        .share_mode(
            FILE_SHARE_READ | FILE_SHARE_WRITE | if share_delete { FILE_SHARE_DELETE } else { 0 },
        )
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)?;
    if !file.metadata()?.is_dir() {
        return Err(io::ErrorKind::NotADirectory.into());
    }
    Ok(file)
}

#[cfg(all(test, windows))]
#[test]
fn windows_checkpoint_admits_files_but_not_stream_names() {
    let root = tempfile::tempdir().unwrap();
    let destination = AtomicNewDestination::admit(&root.path().join("next.json")).unwrap();
    destination.publish(b"checkpoint").unwrap();
    assert_eq!(
        fs::read(root.path().join("next.json")).unwrap(),
        b"checkpoint"
    );
    let stream = AtomicNewDestination::admit(&root.path().join("next.json:metadata"));
    assert!(matches!(stream, Err(error) if error.kind() == io::ErrorKind::InvalidInput));
}

#[cfg(windows)]
fn directory_id(file: &fs::File) -> io::Result<(u32, u32, u32)> {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileInformationByHandle, BY_HANDLE_FILE_INFORMATION,
    };
    let mut info = std::mem::MaybeUninit::<BY_HANDLE_FILE_INFORMATION>::uninit();
    // The handle is borrowed for the call. Success initializes the output.
    let result = unsafe { GetFileInformationByHandle(file.as_raw_handle(), info.as_mut_ptr()) };
    if result == 0 {
        return Err(io::Error::last_os_error());
    }
    let info = unsafe { info.assume_init() };
    Ok((
        info.dwVolumeSerialNumber,
        info.nFileIndexHigh,
        info.nFileIndexLow,
    ))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn directory_replacement_during_write_cleans_only_the_admitted_directory() {
        let root = tempfile::tempdir().unwrap();
        let current = root.path().join("current");
        let rotated = root.path().join("rotated");
        fs::create_dir(&current).unwrap();
        let destination = AtomicNewDestination::admit(&current.join("next")).unwrap();
        let error = destination
            .publish_with(|file| {
                fs::rename(&current, &rotated)?;
                fs::create_dir(&current)?;
                file.write_all(b"sample bytes")
            })
            .unwrap_err();
        assert!(!error.destination_installed());
        assert!(error.to_string().contains("destination directory changed"));
        assert_eq!(fs::read_dir(&current).unwrap().count(), 0);
        assert_eq!(fs::read_dir(&rotated).unwrap().count(), 0);
    }
}

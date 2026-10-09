#[cfg(not(target_arch = "wasm32"))]
use st_nn::resident::{ByteCorpusStudy, BYTE_CORPUS_STUDY_MAX_BYTES};

#[cfg(not(target_arch = "wasm32"))]
fn read_request(path: &std::path::Path) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use std::io::Read;
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // A FIFO must not block in open before descriptor validation runs.
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > BYTE_CORPUS_STUDY_MAX_BYTES as u64 {
        return Err("request must be a regular file of at most 64 MiB".into());
    }
    let mut input = Vec::new();
    // The file can grow after metadata was read; retain the bound during I/O.
    file.take(BYTE_CORPUS_STUDY_MAX_BYTES as u64 + 1)
        .read_to_end(&mut input)?;
    if input.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
        return Err("request grew beyond 64 MiB".into());
    }
    Ok(input)
}

#[cfg(not(target_arch = "wasm32"))]
struct Args {
    request: std::path::PathBuf,
    resume: Option<std::path::PathBuf>,
    output: Option<std::path::PathBuf>,
    stop: Option<usize>,
}

#[cfg(not(target_arch = "wasm32"))]
impl Args {
    fn parse(
        args: impl IntoIterator<Item = std::ffi::OsString>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut args = args.into_iter();
        let mut result = Self {
            request: args.next().ok_or("usage: resident_byte_learning <request.json> [--stop-after N] [--resume checkpoint.json] [--checkpoint-out NEW.json]")?.into(),
            resume: None, output: None, stop: None,
        };
        while let Some(flag) = args.next() {
            let value = args.next().ok_or("missing flag value")?;
            match flag.to_str() {
                Some("--resume") if result.resume.is_none() => result.resume = Some(value.into()),
                Some("--checkpoint-out") if result.output.is_none() => {
                    result.output = Some(value.into())
                }
                Some("--stop-after") if result.stop.is_none() => {
                    result.stop = Some(value.to_str().ok_or("invalid stop cursor")?.parse()?);
                }
                _ => return Err("unknown or duplicate flag".into()),
            }
        }
        if (result.stop.is_some() || result.resume.is_some()) && result.output.is_none() {
            return Err("stop/resume requires --checkpoint-out with a new file path".into());
        }
        if let Some(path) = &result.output {
            if std::fs::symlink_metadata(path).is_ok() {
                return Err("checkpoint output already exists; choose a new file".into());
            }
            if !output_parent(path).is_dir() {
                return Err("checkpoint output directory does not exist".into());
            }
        }
        Ok(result)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn output_parent(path: &std::path::Path) -> &std::path::Path {
    path.parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."))
}

#[cfg(not(target_arch = "wasm32"))]
fn save_checkpoint(path: &std::path::Path, json: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    let mut temporary = tempfile::NamedTempFile::new_in(output_parent(path))?;
    temporary.write_all(json.as_bytes())?;
    temporary.as_file().sync_all()?;
    // Commit only complete bytes and never replace a request or earlier checkpoint.
    temporary.persist_noclobber(path)?;
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse(std::env::args_os().skip(1))?;
    let input = read_request(&args.request)?;
    let study = ByteCorpusStudy::from_json(&input)?;
    let resume = args
        .resume
        .as_ref()
        .map(|path| study.checkpoint_from_json(&read_request(path)?))
        .transpose()?;
    let stop = args.stop.unwrap_or(study.total_updates());
    if args.output.is_some() {
        study.validate_segment(resume.as_ref(), stop)?;
    }
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.corpus.learning")?;
    let report = if let Some(output) = &args.output {
        let segment = pollster::block_on(study.advance(runtime, resume.as_ref(), stop))?;
        save_checkpoint(output, &segment.checkpoint.to_json()?)?;
        segment.report
    } else {
        pollster::block_on(study.run(runtime))?
    };
    println!("{report}");
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #[test]
    fn checkpoint_save_commits_complete_bytes_and_never_clobbers() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoint.json");
        super::save_checkpoint(&path, "original").unwrap();
        assert!(super::save_checkpoint(&path, "replacement").is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "original");
        assert_eq!(std::fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn cli_rejects_ambiguous_or_unsaved_segments_before_gpu() {
        for args in [
            vec![],
            vec!["request", "--stop-after", "1"],
            vec!["request", "--resume", "old.json"],
            vec!["request", "--stop-after", "1", "--stop-after", "2"],
            vec!["request", "--typo", "1"],
            vec!["request", "--resume"],
        ] {
            assert!(super::Args::parse(args.into_iter().map(std::ffi::OsString::from)).is_err());
        }
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("new.json");
        let args = [
            std::ffi::OsString::from("request"),
            "--checkpoint-out".into(),
            output.clone().into(),
        ];
        assert!(super::Args::parse(args.clone()).is_ok());
        std::fs::write(output, "preserve").unwrap();
        assert!(super::Args::parse(args).is_err());
    }

    #[test]
    fn oversized_and_non_file_inputs_fail_before_allocation() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file()
            .set_len(super::BYTE_CORPUS_STUDY_MAX_BYTES as u64 + 1)
            .unwrap();
        assert!(super::read_request(file.path()).is_err());
        let directory = tempfile::tempdir().unwrap();
        assert!(super::read_request(directory.path()).is_err());
    }

    #[test]
    fn bounded_reader_preserves_exact_request_bytes() {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(b"{\"request\":1}\n").unwrap();
        assert_eq!(
            super::read_request(file.path()).unwrap(),
            b"{\"request\":1}\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn fifo_input_is_rejected_without_waiting_for_a_writer() {
        use std::os::unix::ffi::OsStrExt;
        use std::time::{Duration, Instant};
        const ENV: &str = "SPIRALTORCH_BYTE_STUDY_FIFO_PROBE";
        if let Some(path) = std::env::var_os(ENV) {
            assert!(super::read_request(std::path::Path::new(&path)).is_err());
            return;
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("request.fifo");
        let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: the NUL-terminated path lives through the call and points
        // inside a private temporary directory owned by this test.
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::fifo_input_is_rejected_without_waiting_for_a_writer",
            ])
            .env(ENV, &path)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(status.success());
                break;
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("FIFO request blocked before being rejected");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

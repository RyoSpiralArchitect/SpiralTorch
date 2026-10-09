#[cfg(not(target_arch = "wasm32"))]
#[path = "support/byte_learning.rs"]
mod learning;

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
    if !metadata.is_file() || metadata.len() > learning::MAX_REQUEST_BYTES as u64 {
        return Err("request must be a regular file of at most 64 MiB".into());
    }
    let mut input = Vec::new();
    // The file can grow after metadata was read; retain the bound during I/O.
    file.take(learning::MAX_REQUEST_BYTES as u64 + 1)
        .read_to_end(&mut input)?;
    if input.len() > learning::MAX_REQUEST_BYTES {
        return Err("request grew beyond 64 MiB".into());
    }
    Ok(input)
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        return Err("usage: resident_byte_learning <request.json>".into());
    }
    let input = read_request(std::path::Path::new(&args[0]))?;
    learning::validate_request(&input)?;
    let (runtime, _) =
        st_backend_wgpu::runtime::ensure_default_runtime_blocking("byte.corpus.learning")?;
    let report = pollster::block_on(learning::run(runtime, &input))?;
    println!("{report}");
    Ok(())
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #[test]
    fn oversized_and_non_file_inputs_fail_before_allocation() {
        let file = tempfile::NamedTempFile::new().unwrap();
        file.as_file()
            .set_len(super::learning::MAX_REQUEST_BYTES as u64 + 1)
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

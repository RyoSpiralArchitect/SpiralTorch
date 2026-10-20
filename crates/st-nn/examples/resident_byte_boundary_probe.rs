//! Explicitly instrumented first-update capture; never a release qualification.
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use st_backend_wgpu::runtime;
    use st_nn::resident::{ByteCorpusStudy, BYTE_CORPUS_STUDY_MAX_BYTES};
    use std::fs::{self, OpenOptions};
    use std::io::{Read, Write};
    use std::path::Path;

    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn read(path: &Path) -> Result<Vec<u8>> {
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK);
        }
        let file = options.open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() > BYTE_CORPUS_STUDY_MAX_BYTES as u64 {
            return Err("expected a regular file of at most 64 MiB".into());
        }
        let mut bytes = Vec::new();
        file.take(BYTE_CORPUS_STUDY_MAX_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
            return Err("file grew beyond 64 MiB".into());
        }
        Ok(bytes)
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn executable_sha256() -> Result<String> {
        let mut file = fs::File::open(std::env::current_exe()?)?;
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let size = file.read(&mut buffer)?;
            if size == 0 {
                return Ok(format!("{:x}", hash.finalize()));
            }
            hash.update(&buffer[..size]);
        }
    }

    fn save(path: &Path, bytes: &[u8]) -> Result<()> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }

    fn save_json(dir: &Path, name: &str, value: &Value) -> Result<()> {
        save(&dir.join(name), &serde_json::to_vec_pretty(value)?)
    }

    fn finish(
        dir: &Path,
        result: Result<Value>,
        reference: &str,
        read_boundary: impl FnOnce() -> Result<String>,
    ) -> Result<()> {
        let run_error = result.as_ref().err().map(ToString::to_string);
        let outcome = match result {
            Ok(report) => json!({
                "status": "completed", "all_cases_completed": true, "report": report
            }),
            Err(error) => json!({
                "status": "error", "all_cases_completed": false, "error": error.to_string()
            }),
        };
        // Never let a diagnostic readback overwrite or erase the study outcome.
        save_json(dir, "run-outcome.json", &outcome)?;
        let boundary = read_boundary();
        let (equal, boundary_error, boundary_hash) = match boundary {
            Ok(model) => {
                save(&dir.join("observed-boundary.json"), model.as_bytes())?;
                (
                    Some(model == reference),
                    None,
                    Some(sha256(model.as_bytes())),
                )
            }
            Err(error) => (None, Some(error.to_string()), None),
        };
        save_json(
            dir,
            "summary.json",
            &json!({
                "schema": "spiraltorch.byte_boundary_probe.v1",
                "release_ready": false,
                "run_error": run_error,
                "boundary_error": boundary_error,
                "boundary_equal_to_reference": equal,
                "observed_boundary_sha256": boundary_hash,
                "reference_boundary_sha256": sha256(reference.as_bytes()),
                "scope": "Instrumented run. GPU copies and resource lifetimes differ from run(). \
                    Exact canonical checkpoint comparison, not tolerance or root-cause evidence. \
                    Compare the saved report with the original failure and resumed report separately."
            }),
        )?;
        if let Some(error) = run_error {
            return Err(format!("observed study failed: {error}; see saved outcome").into());
        }
        if let Some(error) = boundary_error {
            return Err(format!("boundary read failed: {error}; study outcome preserved").into());
        }
        if equal != Some(true) {
            return Err(
                "first boundary differs; retain all evidence, do not rerun until green".into(),
            );
        }
        println!("First-boundary capture completed; release hold unchanged.");
        Ok(())
    }

    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if args.len() != 4 {
            return Err(
                "usage: resident_byte_boundary_probe REQUEST FIRST_CHECKPOINT CASE NEW_OUTPUT_DIR"
                    .into(),
            );
        }
        let request = read(Path::new(&args[0]))?;
        let first_bytes = read(Path::new(&args[1]))?;
        let name = args[2].to_str().ok_or("case name must be UTF-8")?;
        let dir = Path::new(&args[3]);
        let study = ByteCorpusStudy::from_json(&request)?;
        let first = study.checkpoint_from_json(&first_bytes)?;
        if study.total_updates() < 2 || first.completed_updates() != 1 {
            return Err(
                "requires at least two updates and a request-bound update-1 checkpoint".into(),
            );
        }
        study.validate_segment(None, study.total_updates())?;
        let reference = study.case_checkpoint(&first, name)?.to_json()?;
        let executable_hash = executable_sha256()?;

        // Preflight is CPU-only, and an existing output is never overwritten.
        fs::create_dir(dir)?;
        save(&dir.join("request.json"), &request)?;
        save(&dir.join("reference-first-checkpoint.json"), &first_bytes)?;
        save(&dir.join("reference-boundary.json"), reference.as_bytes())?;
        save_json(
            dir,
            "invocation.json",
            &json!({
                "schema": "spiraltorch.byte_boundary_invocation.v1",
                "request_sha256": study.request_sha256(),
                "reference_first_checkpoint_sha256": sha256(&first_bytes),
                "case": name, "requested_capture_after_update": 1, "all_cases_requested": true,
                "boundary_read_policy": "after_run_returns",
                "executable_sha256": executable_hash,
                "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
                "crate_version": env!("CARGO_PKG_VERSION"),
                "pid": std::process::id(),
                "started_unix_ms": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?.as_millis(),
                "source_identity": "Record commit, patch, toolchain and build command separately.",
                "release_ready": false
            }),
        )?;
        let mut snapshot = None;
        let result = (|| -> Result<Value> {
            let (runtime, _) = runtime::ensure_default_runtime_blocking("byte.boundary.probe")?;
            save_json(
                dir,
                "runtime.json",
                &json!({"adapter": format!("{:?}", runtime.adapter_info())}),
            )?;
            pollster::block_on(
                study.run_with_update_observer(runtime, |case, step, model| {
                    if case == name && step == 1 {
                        if snapshot.is_some() {
                            return Err("selected first boundary observed more than once".into());
                        }
                        snapshot = Some(model.checkpoint_snapshot()?);
                    }
                    Ok(())
                }),
            )
        })();
        finish(dir, result, &reference, || {
            Ok(snapshot
                .ok_or("selected first boundary was not reached")?
                .read()?
                .to_json()?)
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn output_is_exclusive_and_input_is_bounded_and_regular() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("input");
            save(&path, b"first").unwrap();
            assert!(save(&path, b"replacement").is_err());
            assert_eq!(read(&path).unwrap(), b"first");
            assert!(read(dir.path()).is_err());
            let path = dir.path().join("large");
            fs::File::create(&path)
                .unwrap()
                .set_len(BYTE_CORPUS_STUDY_MAX_BYTES as u64 + 1)
                .unwrap();
            assert!(read(&path).is_err());
        }

        #[test]
        fn failing_boundary_keeps_successful_primary_report() {
            let dir = tempfile::tempdir().unwrap();
            let report = json!({"original": "full outcome"});
            let error = finish(dir.path(), Ok(report.clone()), "{}", || {
                let saved: Value =
                    serde_json::from_slice(&read(&dir.path().join("run-outcome.json")).unwrap())
                        .unwrap();
                assert_eq!(saved["report"], report);
                assert_eq!(saved["all_cases_completed"], true);
                Err("injected readback failure".into())
            })
            .unwrap_err();
            assert!(error.to_string().contains("injected readback failure"));
            let summary: Value =
                serde_json::from_slice(&read(&dir.path().join("summary.json")).unwrap()).unwrap();
            assert_eq!(summary["release_ready"], false);
            assert_eq!(summary["boundary_equal_to_reference"], Value::Null);
            assert_eq!(summary["run_error"], Value::Null);
        }

        #[test]
        fn study_failure_remains_primary_even_with_equal_boundary() {
            let dir = tempfile::tempdir().unwrap();
            let error = finish(
                dir.path(),
                Err("original study failure".into()),
                "{}",
                || {
                    assert!(dir.path().join("run-outcome.json").is_file());
                    let saved: Value = serde_json::from_slice(
                        &read(&dir.path().join("run-outcome.json")).unwrap(),
                    )
                    .unwrap();
                    assert_eq!(saved["all_cases_completed"], false);
                    Ok("{}".into())
                },
            )
            .unwrap_err();
            assert!(error.to_string().contains("original study failure"));
        }

        #[test]
        fn mismatch_is_not_accepted_or_tolerance_adjusted() {
            let dir = tempfile::tempdir().unwrap();
            assert!(finish(dir.path(), Ok(json!({})), "{\"weight\":1.0}", || {
                Ok("{\"weight\":1.0000001}".into())
            })
            .is_err());
            let summary: Value =
                serde_json::from_slice(&read(&dir.path().join("summary.json")).unwrap()).unwrap();
            assert_eq!(summary["boundary_equal_to_reference"], false);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::run()
}

#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("resident_byte_boundary_probe requires native files; the update observer itself is available to WASM clients");
}

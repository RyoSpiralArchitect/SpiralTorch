//! One selected-case diagnostic invocation: continuous then original split resume.
//! Usage: resident_case24_diagnostic REQUEST FIRST_CHECKPOINT NEW_OUTPUT_DIRECTORY
//! A supervising process must enforce the separately approved 90-minute deadline.
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use st_backend_wgpu::runtime;
    use st_nn::resident::{
        ByteCorpusCase24Observation, ByteCorpusStudy, BYTE_CORPUS_STUDY_MAX_BYTES,
    };
    use std::{
        fs::{self, OpenOptions},
        io::{Read, Write},
        path::Path,
    };
    type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
    const CASE: &str = "mixture-11-3-train-query-frozen";

    fn read(path: &Path) -> Result<Vec<u8>> {
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NONBLOCK);
        }
        let file = options.open(path)?;
        if !file.metadata()?.is_file()
            || file.metadata()?.len() > BYTE_CORPUS_STUDY_MAX_BYTES as u64
        {
            return Err("input must be a regular file of at most64MiB".into());
        }
        let mut bytes = Vec::new();
        file.take(BYTE_CORPUS_STUDY_MAX_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > BYTE_CORPUS_STUDY_MAX_BYTES {
            return Err("input grew beyond64MiB".into());
        }
        Ok(bytes)
    }
    fn hash(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }
    fn save(path: &Path, bytes: &[u8]) -> Result<()> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }
    fn save_json(path: &Path, value: &Value) -> Result<()> {
        save(path, &serde_json::to_vec_pretty(value)?)
    }
    fn retain_outputs(
        dir: &Path,
        label: &str,
        observed: ByteCorpusCase24Observation,
        reference: &str,
    ) -> Result<bool> {
        // Persist original execution outcome before any diagnostic copy/map.
        save_json(
            &dir.join(format!("{label}-outcome.json")),
            &observed.outcome,
        )?;
        let completed = observed.outcome["status"] == "completed";
        let mut evidence =
            json!({"trajectory":label,"execution_completed":completed,"release_ready":false});
        for (name, handles) in [
            ("first", observed.first),
            ("final", observed.final_checkpoint),
        ] {
            if let Some(handles) = handles {
                let result =
                    (|| -> Result<String> { Ok(handles.snapshot()?.read()?.to_json()?) })();
                match result {
                    Ok(raw) => {
                        save(
                            &dir.join(format!("{label}-{name}-checkpoint.json")),
                            raw.as_bytes(),
                        )?;
                        evidence[format!("{name}_sha256")] = json!(hash(raw.as_bytes()));
                        if name == "first" {
                            evidence["first_equals_original_split_exact_json"] =
                                json!(raw == reference);
                        }
                    }
                    Err(error) => evidence[format!("{name}_read_error")] = json!(error.to_string()),
                }
            }
        }
        if let Some(graph) = observed.graph {
            match graph.read().and_then(|g| g.to_json()) {
                Ok(raw) => {
                    save(
                        &dir.join(format!("{label}-update2-graph.json")),
                        raw.as_bytes(),
                    )?;
                    evidence["graph_sha256"] = json!(hash(raw.as_bytes()));
                }
                Err(error) => evidence["graph_read_error"] = json!(error.to_string()),
            }
        }
        save_json(&dir.join(format!("{label}-evidence.json")), &evidence)?;
        Ok(completed)
    }
    pub fn run() -> Result<()> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        if args.len() != 3 {
            return Err(
                "usage: resident_case24_diagnostic REQUEST FIRST_CHECKPOINT NEW_OUTPUT_DIRECTORY"
                    .into(),
            );
        }
        if args.iter().any(|p| !Path::new(p).is_absolute()) {
            return Err("all paths must be absolute".into());
        }
        let request = read(Path::new(&args[0]))?;
        let first_bytes = read(Path::new(&args[1]))?;
        let study = ByteCorpusStudy::from_json(&request)?;
        let first = study.checkpoint_from_json(&first_bytes)?;
        study.validate_case24_diagnostic(&first)?;
        let reference = study.case_checkpoint(&first, CASE)?.to_json()?;
        let dir = Path::new(&args[2]);
        fs::create_dir(dir)?;
        save(
            &dir.join("original-selected-first-checkpoint.json"),
            reference.as_bytes(),
        )?;
        save_json(
            &dir.join("invocation.json"),
            &json!({
                "schema":"spiraltorch.case24_invocation.v1","case":CASE,
                "request_sha256":hash(&request),"first_checkpoint_sha256":hash(&first_bytes),
                "selected_reference_sha256":hash(reference.as_bytes()),"trajectories":["continuous","original_split_resume"],
                "executed_cases_per_trajectory":1,"total_executed_updates":3,"retry_policy":"none","max_graph_bytes":67108864,
                "deadline_policy":"external supervisor90minutes","release_ready":false,
                "source_identity":"Record source commit, patch and toolchain separately.",
                "started_unix_ms":std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_millis()
            }),
        )?;
        let execution = (|| -> Result<()> {
            let (runtime, _) = runtime::ensure_default_runtime_blocking("case24.diagnostic")?;
            save_json(
                &dir.join("runtime.json"),
                &json!({"adapter":format!("{:?}",runtime.adapter_info())}),
            )?;
            for (resumed, label) in [(false, "continuous"), (true, "original_split_resume")] {
                let observed = pollster::block_on(study.run_case24_diagnostic(
                    runtime.clone(),
                    &first,
                    resumed,
                ))?;
                if !retain_outputs(dir, label, observed, &reference)? {
                    return Err(
                        "selected trajectory failed; no further trajectory attempted".into(),
                    );
                }
            }
            Ok(())
        })();
        save_json(
            &dir.join("completion.json"),
            &match &execution {
                Ok(()) => json!({"status":"completed","release_ready":false}),
                Err(error) => {
                    json!({"status":"error","error":error.to_string(),"release_ready":false})
                }
            },
        )?;
        execution
    }
}
#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    native::run()
}
#[cfg(target_arch = "wasm32")]
fn main() {
    panic!("native diagnostic only")
}

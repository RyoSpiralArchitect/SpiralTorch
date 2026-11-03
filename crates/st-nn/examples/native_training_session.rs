//! Resume one native SGD update using only local artifacts (no Python or Torch).
//! Usage: native_training_session CONFIG WEIGHTS DOCUMENTS CHECKPOINT PIN OUTPUT [STOP_AFTER]
//! DOCUMENTS is a JSON array of token-ID arrays or an object with token_documents
//! and unshifted labels. OUTPUT is the next checkpoint.
//! Stdout contains the checkpoint identity and the optional update report.
use st_nn::causal_lm::{CausalLmResult, NativeCausalLm, NativeCausalLmTrainingSession};
use st_nn::weights::{SafeTensorArchive, WeightLoadLimits};
use std::fs::File;
use std::io::Read;
use std::path::Path;

fn read_limited(path: &str, limit: usize) -> CausalLmResult<Vec<u8>> {
    let mut bytes = Vec::new();
    File::open(path)?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("input file exceeds byte budget".into());
    }
    Ok(bytes)
}

fn main() -> CausalLmResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(6..=7).contains(&args.len()) {
        return Err(
            "usage: native_training_session CONFIG WEIGHTS DOCUMENTS CHECKPOINT PIN OUTPUT [STOP_AFTER]".into(),
        );
    }
    if Path::new(&args[5]).exists() {
        return Err("output must be a new checkpoint file".into());
    }
    let checkpoint = read_limited(&args[3], NativeCausalLmTrainingSession::MAX_PAYLOAD_BYTES)?;
    let limits = WeightLoadLimits::default();
    #[cfg(not(target_arch = "wasm32"))]
    let archive = SafeTensorArchive::open(&args[1], limits)?;
    #[cfg(target_arch = "wasm32")]
    let archive = SafeTensorArchive::from_owned_bytes(
        read_limited(&args[1], limits.max_file_bytes)?,
        limits,
    )?;
    let base = NativeCausalLm::from_gpt2_json(
        &read_limited(&args[0], 1_048_576)?,
        &archive,
        Default::default(),
    )?;
    drop(archive);
    let resume = NativeCausalLmTrainingSession::prepare_restore(base, &checkpoint, &args[4])?;
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Labeled {
        token_documents: Vec<Vec<usize>>,
        labels: Vec<Vec<i64>>,
    }
    let bytes = read_limited(&args[2], 256 * 1024 * 1024)?;
    let mut session = if resume.has_labels() {
        let data: Labeled = serde_json::from_slice(&bytes)?;
        resume.resume_with_labels(data.token_documents, Some(data.labels))?
    } else {
        resume.resume(serde_json::from_slice(&bytes)?)?
    };
    if let Some(stop) = args.get(6) {
        let (steps, pin) = session.advance_to_checkpoint_new(stop.parse()?, &args[5])?;
        #[derive(serde::Serialize)]
        struct Segment {
            checkpoint_id: String,
            steps: Vec<st_nn::causal_lm::CausalLmTrainingSessionStep>,
        }
        println!(
            "{}",
            serde_json::to_string(&Segment {
                checkpoint_id: pin,
                steps
            })?
        );
    } else {
        let (step, pin) = if session.exhausted() {
            (None, session.save_checkpoint_new(&args[5])?)
        } else {
            let (mut steps, pin) =
                session.advance_to_checkpoint_new(session.completed_updates() + 1, &args[5])?;
            (steps.pop(), pin)
        };
        println!(
            "{}",
            serde_json::json!({"checkpoint_id": pin, "step": step})
        );
    }
    Ok(())
}

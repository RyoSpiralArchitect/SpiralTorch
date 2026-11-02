//! Resume one native SGD update using only local artifacts (no Python or Torch).
//! Usage: native_training_session CONFIG WEIGHTS DOCUMENTS CHECKPOINT PIN OUTPUT
//! DOCUMENTS is a JSON array of token-ID arrays; OUTPUT is the next checkpoint.
//! Stdout contains the checkpoint identity and the optional update report.
use st_nn::causal_lm::{CausalLmResult, NativeCausalLm, NativeCausalLmTrainingSession};
use st_nn::weights::{SafeTensorArchive, WeightLoadLimits};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
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
    if args.len() != 6 {
        return Err(
            "usage: native_training_session CONFIG WEIGHTS DOCUMENTS CHECKPOINT PIN OUTPUT".into(),
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
    let documents: Vec<Vec<usize>> =
        serde_json::from_slice(&read_limited(&args[2], 256 * 1024 * 1024)?)?;
    let mut session = resume.resume(documents)?;
    let step = session.step()?;
    let (payload, pin) = session.checkpoint_with_id()?;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[5])?
        .write_all(payload.as_bytes())?;
    println!(
        "{}",
        serde_json::json!({"checkpoint_id": pin, "step": step})
    );
    Ok(())
}

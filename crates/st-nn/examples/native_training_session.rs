//! Resume one native SGD update using only local artifacts (no Python or Torch).
//! Usage: native_training_session CONFIG WEIGHTS DOCUMENTS CHECKPOINT PIN OUTPUT
//! DOCUMENTS is a JSON array of token-ID arrays; OUTPUT is the next checkpoint.
//! Stdout contains the checkpoint identity and the optional update report.
use st_nn::causal_lm::{CausalLmResult, NativeCausalLm, NativeCausalLmTrainingSession};
use st_nn::weights::{SafeTensorArchive, WeightLoadLimits};
use std::fs;

fn main() -> CausalLmResult<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 6 {
        return Err(
            "usage: native_training_session CONFIG WEIGHTS DOCUMENTS CHECKPOINT PIN OUTPUT".into(),
        );
    }
    let archive =
        SafeTensorArchive::from_owned_bytes(fs::read(&args[1])?, WeightLoadLimits::default())?;
    let base = NativeCausalLm::from_gpt2_json(&fs::read(&args[0])?, &archive, Default::default())?;
    let documents: Vec<Vec<usize>> = serde_json::from_slice(&fs::read(&args[2])?)?;
    let mut session =
        NativeCausalLmTrainingSession::from_json(base, documents, &fs::read(&args[3])?, &args[4])?;
    let step = session.step()?;
    fs::write(&args[5], session.to_json()?)?;
    println!(
        "{}",
        serde_json::json!({"checkpoint_id": session.checkpoint_id()?, "step": step})
    );
    Ok(())
}

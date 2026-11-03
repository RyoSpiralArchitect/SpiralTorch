use super::*;
use std::{fs, io, path::Path};

const RECEIPT_SCHEMA: &str = "spiraltorch.native_corpus.observed_checkpoint.v1";
const MAX_RECEIPT_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Serialize)]
pub struct CausalLmObservedCheckpointFiles {
    pub file: String,
    pub checkpoint_id: String,
    pub updates: u64,
    pub observation_file: String,
    pub observation_id: String,
}

#[derive(Serialize)]
struct CheckpointRef<'a> {
    file: &'a str,
    checkpoint_id: &'a str,
    updates: u64,
}

#[derive(Serialize)]
struct Receipt<'a> {
    schema: &'static str,
    checkpoint: CheckpointRef<'a>,
    observation: &'a CausalLmObservedTrainingStep,
}

impl NativeCausalLmTrainer {
    pub const MAX_PAYLOAD_BYTES: usize = MAX_CHECKPOINT_BYTES;
    pub const MAX_CHECKPOINT_INTERVAL: u64 = checkpoint::MAX_INTERVAL;

    /// Save this exact geometry, AdamW state and cursor without advancing.
    pub fn save_checkpoint_new(&self, path: impl AsRef<Path>) -> CausalLmResult<String> {
        checkpoint::save(path.as_ref(), self.checkpoint_with_id()?)
    }

    /// Commit only after the unchanged v1 checkpoint is complete at a new name.
    pub fn advance_to_checkpoint_new(
        &mut self,
        stop_after: u64,
        path: impl AsRef<Path>,
    ) -> CausalLmResult<(Vec<CausalLmTrainingStep>, String)> {
        let n = checkpoint::count(self.updates, self.total_updates(), stop_after)?;
        checkpoint::advance_new(self, n, path.as_ref(), Self::step, Self::checkpoint_with_id)
    }

    /// As above, but observe the final step once and publish its receipt last.
    /// Requires a private, caller-controlled directory. I/O failure leaves the
    /// live model, moments, clocks and cursor unchanged. Never overwrite or
    /// automatically adopt an orphan checkpoint after a failed publication.
    pub fn advance_to_observed_checkpoint_new(
        &mut self,
        stop_after: u64,
        directory: impl AsRef<Path>,
    ) -> CausalLmResult<(Vec<CausalLmTrainingStep>, CausalLmObservedCheckpointFiles)> {
        self.observed_with(
            stop_after,
            directory.as_ref(),
            st_core::runtime::atomic_create_new,
        )
    }

    pub(in crate::causal_lm) fn observed_with(
        &mut self,
        stop: u64,
        dir: &Path,
        mut write: impl FnMut(&Path, &[u8]) -> io::Result<()>,
    ) -> CausalLmResult<(Vec<CausalLmTrainingStep>, CausalLmObservedCheckpointFiles)> {
        let n = checkpoint::count(self.updates, self.total_updates(), stop)?;
        let meta = fs::symlink_metadata(dir)?;
        if !meta.is_dir() {
            return Err("use a private run directory, not a link or file".into());
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if meta.permissions().mode() & 0o077 != 0 {
                return Err("use a private run directory (0700 on POSIX)".into());
            }
        }
        let file = format!("checkpoint-{stop:06}.json");
        let observation_file = format!("observation-{stop:06}.json");
        let path = dir.join(&file);
        let receipt_path = dir.join(&observation_file);
        checkpoint::admit_new(&path)?;
        checkpoint::admit_new(&receipt_path)?;
        checkpoint::transaction(self, |next| {
            let mut reports = checkpoint::steps(next, n - 1, Self::step)?;
            let observed = next
                .step_observed()?
                .ok_or("unexpected end of observed segment")?;
            let (payload, checkpoint_id) = next.checkpoint_with_id()?;
            let mut receipt = serde_json::to_vec(&Receipt {
                schema: RECEIPT_SCHEMA,
                checkpoint: CheckpointRef {
                    file: &file,
                    checkpoint_id: &checkpoint_id,
                    updates: stop,
                },
                observation: &observed,
            })?;
            receipt.push(b'\n');
            if receipt.len() > MAX_RECEIPT_BYTES {
                return Err("observation exceeds receipt budget".into());
            }
            let files = CausalLmObservedCheckpointFiles {
                file,
                checkpoint_id,
                updates: stop,
                observation_file,
                observation_id: identity(&receipt),
            };
            reports.push(observed.step);
            write(&path, payload.as_bytes())?;
            write(&receipt_path, &receipt)?;
            Ok((reports, files))
        })
    }
}

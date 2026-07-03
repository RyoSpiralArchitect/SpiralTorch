// ============================================================================
//  SpiralReality Proprietary
//  Copyright (c) 2025 SpiralReality. All Rights Reserved.
//
//  NOTICE: This file contains confidential and proprietary information of
//  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
//  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
//  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
//
//  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
//  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
//  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
//  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
//  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

//! End-to-end fine-tuning contract smoke.
//!
//! This intentionally stays tiny: it proves that a run can start from a source
//! checkpoint, freeze a backbone-style parameter, fine-tune the remaining head,
//! and audit the outcome without relying on Python or external ML stacks.

use st_core::backend::device_caps::DeviceCaps;
use st_nn::{
    dataset_from_vec, load_json_checked, save_json, Linear, MeanSquaredError, Module,
    ModuleTrainer, RoundtableConfig, Tensor,
};
use st_tensor::pure::{PureResult, TensorError};

fn dataset(offset: f32) -> PureResult<Vec<(Tensor, Tensor)>> {
    Ok(vec![
        (
            Tensor::from_vec(1, 2, vec![0.0, 1.0])?,
            Tensor::from_vec(1, 2, vec![offset, 1.0 + offset])?,
        ),
        (
            Tensor::from_vec(1, 2, vec![1.0, 0.0])?,
            Tensor::from_vec(1, 2, vec![1.0 + offset, offset])?,
        ),
    ])
}

fn train_once(
    trainer: &mut ModuleTrainer,
    model: &mut Linear,
    data: Vec<(Tensor, Tensor)>,
) -> PureResult<f32> {
    let schedule = trainer.roundtable(1, 2, RoundtableConfig::default());
    let loader = dataset_from_vec(data).batched(2);
    let mut loss = MeanSquaredError::new();
    let stats = trainer.train_epoch(model, &mut loss, loader, &schedule)?;
    Ok(stats.average_loss)
}

fn contract_error(message: impl Into<String>) -> TensorError {
    TensorError::IoError {
        message: message.into(),
    }
}

fn main() -> PureResult<()> {
    let caps = DeviceCaps::wgpu(32, true, 256);
    let mut trainer = ModuleTrainer::new(caps, -1.0, 0.05, 0.01);
    trainer.set_max_grad_norm(Some(1.0))?;
    let mut source = Linear::new("layer", 2, 2)?;
    trainer.prepare(&mut source)?;
    let source_loss = train_once(&mut trainer, &mut source, dataset(0.0)?)?;

    let checkpoint = std::env::temp_dir().join(format!(
        "spiraltorch_finetune_contract_{}.json",
        std::process::id()
    ));
    save_json(&source, &checkpoint)?;

    let mut target = Linear::new("layer", 2, 2)?;
    trainer.prepare(&mut target)?;
    let load = load_json_checked(&mut target, &checkpoint)?;
    let _ = std::fs::remove_file(&checkpoint);
    if !load.matched {
        return Err(contract_error("source checkpoint fingerprint mismatch"));
    }

    let frozen_weight_params = target.set_parameters_trainable_by_suffix("::weight", false)?;
    let boosted_bias_params = target.set_parameters_learning_rate_scale_by_suffix("::bias", 1.1)?;
    let ft_ready_state = target.state_dict()?;
    let ft_ready_resume = trainer.resume_fingerprint(&target)?;
    let mut resumed = Linear::new("layer", 2, 2)?;
    trainer.prepare(&mut resumed)?;
    let resume_load = resumed.load_state_dict_checked(&ft_ready_state)?;
    resumed.set_parameters_trainable_by_suffix("::weight", false)?;
    resumed.set_parameters_learning_rate_scale_by_suffix("::bias", 1.1)?;
    let resume_audit = trainer.audit_resume_fingerprint(&resumed, &ft_ready_resume)?;
    if !resume_load.matched || !resume_audit.matched {
        return Err(contract_error("FT-ready resume fingerprint mismatch"));
    }
    let before_ft = target.state_dict()?;
    let reload_loss = train_once(&mut trainer, &mut target, dataset(0.5)?)?;
    let movement = target.audit_parameter_movement(&before_ft, 1e-8)?;
    if !movement.frozen_stable() {
        return Err(contract_error("frozen parameter moved during fine-tune"));
    }
    if !movement.trainable_movement_observed() {
        return Err(contract_error(
            "no trainable parameter moved during fine-tune",
        ));
    }

    println!("source_loss={source_loss:.6}");
    println!("reload_loss={reload_loss:.6}");
    println!("source_hash={}", load.source.hash);
    println!("loaded_hash={}", load.loaded.hash);
    println!("load_matched={}", load.matched);
    println!(
        "resume_hash={} trainer_hash={} parameter_training_hash={} resume_matched={}",
        ft_ready_resume.hash,
        ft_ready_resume.trainer.hash,
        ft_ready_resume.parameter_training.hash,
        resume_audit.matched
    );
    println!(
        "movement_status={} frozen_stable={} trainable_moved={} frozen_weight_params={} boosted_bias_params={}",
        movement.status(),
        movement.frozen_stable(),
        movement.trainable_movement_observed(),
        frozen_weight_params,
        boosted_bias_params
    );

    Ok(())
}

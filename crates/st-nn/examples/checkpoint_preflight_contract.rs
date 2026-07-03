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
//  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
//  IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
//  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ============================================================================

//! External-checkpoint preflight contract for adapter fine-tuning.
//!
//! This keeps the non-scratch FT handoff honest before real LLM checkpoints are
//! connected: external keys are mapped into SpiralTorch module keys, a
//! transposed LM-head weight is adapted, a shorter bias is overlap-copied and
//! zero-padded, and the preflight report explains every source key/transform.
//! A second HF/PyTorch-style key layout mirrors the Python preflight helper so
//! the Rust backend and Python facade stay aligned before larger LLM imports.

use st_nn::{
    Linear, LoraLinear, Module, StateCompatibilityReport, StateKeyMapRule, StateTensorTransform,
    Tensor,
};
use st_tensor::pure::{PureResult, TensorError};
use std::collections::HashMap;

const VOCAB: usize = 8;
const HIDDEN: usize = 4;
const TARGET_CLASSES: usize = 6;
const EXTERNAL_CLASSES: usize = 4;

fn contract_error(message: impl Into<String>) -> TensorError {
    TensorError::IoError {
        message: message.into(),
    }
}

fn external_state() -> PureResult<HashMap<String, Tensor>> {
    let embed = Linear::new("embed", VOCAB, HIDDEN)?;
    let head = Linear::new("head", HIDDEN, TARGET_CLASSES)?;
    let embed_state = embed.state_dict()?;
    let head_state = head.state_dict()?;

    Ok(HashMap::from([
        (
            "model.embed.weight".to_string(),
            embed_state["embed::weight"].clone(),
        ),
        (
            "model.embed.bias".to_string(),
            embed_state["embed::bias"].clone(),
        ),
        (
            "model.lm_head.weight".to_string(),
            head_state["head::weight"].transpose(),
        ),
        (
            "model.lm_head.bias".to_string(),
            Tensor::from_vec(1, EXTERNAL_CLASSES, vec![0.1, -0.2, 0.3, -0.4])?,
        ),
        (
            "model.unused.layernorm.weight".to_string(),
            Tensor::from_vec(1, 2, vec![1.0, 1.0])?,
        ),
    ]))
}

fn key_rules() -> HashMap<String, StateKeyMapRule> {
    HashMap::from([
        (
            "model.embed.weight".to_string(),
            StateKeyMapRule::new("embed::weight"),
        ),
        (
            "model.embed.bias".to_string(),
            StateKeyMapRule::new("embed::bias"),
        ),
        (
            "model.lm_head.weight".to_string(),
            StateKeyMapRule::with_transform("head::weight", StateTensorTransform::Transpose),
        ),
        (
            "model.lm_head.bias".to_string(),
            StateKeyMapRule::with_transform("head::bias", StateTensorTransform::CopyOverlapZeros),
        ),
    ])
}

fn hf_style_external_state() -> PureResult<HashMap<String, Tensor>> {
    let embed = Linear::new("embed", VOCAB, HIDDEN)?;
    let head = Linear::new("head", HIDDEN, TARGET_CLASSES)?;
    let embed_state = embed.state_dict()?;
    let head_state = head.state_dict()?;

    Ok(HashMap::from([
        (
            "transformer.wte.weight".to_string(),
            embed_state["embed::weight"].clone(),
        ),
        (
            "transformer.wte.bias".to_string(),
            embed_state["embed::bias"].clone(),
        ),
        (
            "lm_head.weight".to_string(),
            head_state["head::weight"].transpose(),
        ),
        (
            "lm_head.bias".to_string(),
            Tensor::from_vec(1, EXTERNAL_CLASSES, vec![0.1, -0.2, 0.3, -0.4])?,
        ),
        (
            "transformer.h.0.ln_1.weight".to_string(),
            Tensor::from_vec(1, 2, vec![1.0, 1.0])?,
        ),
    ]))
}

fn hf_lm_key_rules() -> HashMap<String, StateKeyMapRule> {
    HashMap::from([
        (
            "transformer.wte.weight".to_string(),
            StateKeyMapRule::new("embed::weight"),
        ),
        (
            "transformer.wte.bias".to_string(),
            StateKeyMapRule::new("embed::bias"),
        ),
        (
            "lm_head.weight".to_string(),
            StateKeyMapRule::with_transform("head::weight", StateTensorTransform::Transpose),
        ),
        (
            "lm_head.bias".to_string(),
            StateKeyMapRule::with_transform("head::bias", StateTensorTransform::CopyOverlapZeros),
        ),
    ])
}

fn case_label(source: &str, label: &str) -> String {
    if source == "toy" {
        label.to_string()
    } else {
        format!("{source}_{label}")
    }
}

fn print_report(label: &str, report: &StateCompatibilityReport) {
    println!(
        "preflight_report label={label} compatible={} matched={} missing={} shape_mismatched={} extra={} source_hash={} matched_subset_hash={}",
        report.compatible,
        report.matched,
        report.missing,
        report.shape_mismatched,
        report.extra,
        report.source.hash,
        report.matched_subset.hash
    );
    for entry in &report.entries {
        println!(
            "preflight_entry label={label} name={} status={} source_name={} transform={} expected_shape={:?} source_shape={:?} original_source_shape={:?}",
            entry.name,
            entry.status.as_str(),
            entry.source_name.as_deref().unwrap_or("none"),
            entry.transform.as_str(),
            entry.expected_shape,
            entry.source_shape,
            entry.original_source_shape
        );
    }
}

fn require_report(label: &str, report: &StateCompatibilityReport) -> PureResult<()> {
    if !report.compatible {
        return Err(contract_error(format!(
            "{label} checkpoint preflight failed: missing={} shape_mismatched={}",
            report.missing, report.shape_mismatched
        )));
    }
    Ok(())
}

fn run_preflight_case(
    source: &str,
    external: &HashMap<String, Tensor>,
    rules: &HashMap<String, StateKeyMapRule>,
) -> PureResult<()> {
    let mut embed = Linear::new("embed", VOCAB, HIDDEN)?;
    let embed_report = embed.state_dict_compatibility_with_key_rules(external, rules)?;
    let embed_label = case_label(source, "embed");
    print_report(&embed_label, &embed_report);
    require_report(&embed_label, &embed_report)?;
    let embed_load = embed.load_state_dict_subset_adapted_checked(external, rules)?;
    if !embed_load.matched {
        return Err(contract_error("embed adapted checkpoint load mismatch"));
    }

    let mut head = LoraLinear::new("head", HIDDEN, TARGET_CLASSES, 2, 8.0)?;
    let head_report = head.base_state_dict_compatibility_with_key_rules(external, rules)?;
    let head_label = case_label(source, "lora_head_base");
    print_report(&head_label, &head_report);
    require_report(&head_label, &head_report)?;
    let head_load = head.load_base_from_state_dict_adapted(external, rules)?;
    if !head_load.matched {
        return Err(contract_error(
            "LoRA head base adapted checkpoint load mismatch",
        ));
    }

    let head_state = head.base_state_dict();
    let padded_bias = &head_state["head::bias"];
    if padded_bias.data() != &[0.1, -0.2, 0.3, -0.4, 0.0, 0.0] {
        return Err(contract_error(
            "LoRA head bias was not zero-padded as expected",
        ));
    }
    println!(
        "checkpoint_preflight_contract_case source={source} status=ok external_keys={} embed_loaded={} head_loaded={}",
        external.len(),
        embed_load.matched,
        head_load.matched
    );
    Ok(())
}

fn main() -> PureResult<()> {
    let external = external_state()?;
    let rules = key_rules();
    run_preflight_case("toy", &external, &rules)?;

    let hf_external = hf_style_external_state()?;
    let hf_rules = hf_lm_key_rules();
    run_preflight_case("hf_style", &hf_external, &hf_rules)?;

    println!("checkpoint_preflight_contract status=ok cases=2");
    Ok(())
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// © 2025 Ryo ∴ SpiralArchitect (kishkavsesvit@icloud.com)
// Part of SpiralTorch — Licensed under AGPL-3.0-or-later.
// Unauthorized derivative works or closed redistribution prohibited under AGPL §13.

#![cfg(feature = "cuda")]

use crate::backend::cuda_loader::{self, CudaModule};
use cudarc::driver::{LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use std::convert::TryFrom;
use std::sync::OnceLock;

const MODULE_NAME: &str = "spiraltorch_attention";
const KERNEL_NAME: &str = "scaled_dot_attention_kernel";
const MODULE_KERNELS: &[&str] = &[KERNEL_NAME];
const CUDA_SOURCE: &str = include_str!("cuda_attention.cu");
const THREADS_PER_BLOCK: u32 = 128;
const WARP_LANES: u32 = 32;
const SCRATCH_SIZE: usize = (THREADS_PER_BLOCK as usize / WARP_LANES as usize).max(1);

static COMPILED_PTX: OnceLock<cudarc::nvrtc::Ptx> = OnceLock::new();
static CUDA_MODULE: OnceLock<CudaModule> = OnceLock::new();

/// Configuration describing how to launch the CUDA attention kernel.
#[derive(Clone, Copy, Debug)]
pub struct AttentionConfig {
    /// Batch size.
    pub batch: u32,
    /// Number of attention heads.
    pub heads: u32,
    /// Number of Z-space slices to treat as independent contexts.
    pub z_levels: u32,
    /// Sequence length per head.
    pub sequence_len: u32,
    /// Feature dimension per head.
    pub head_dim: u32,
    /// Optional scale applied to the dot-product (defaults to 1/sqrt(head_dim)).
    pub scale: Option<f32>,
    /// Whether to apply causal masking (mask out keys greater than the current query).
    pub causal: bool,
}

impl AttentionConfig {
    /// Total number of independent contexts (batch × heads × z_levels).
    pub fn contexts(&self) -> usize {
        (self.batch as usize)
            .saturating_mul(self.heads as usize)
            .saturating_mul(self.z_levels as usize)
    }

    /// Returns the softmax scaling factor.
    pub fn scale(&self) -> f32 {
        match self.scale {
            Some(explicit) => explicit,
            None => {
                let dim = self.head_dim.max(1) as f32;
                dim.recip().sqrt()
            }
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.batch == 0 {
            return Err("attention batch must be greater than zero".to_string());
        }
        if self.heads == 0 {
            return Err("attention heads must be greater than zero".to_string());
        }
        if self.z_levels == 0 {
            return Err("attention z_levels must be greater than zero".to_string());
        }
        if self.sequence_len == 0 {
            return Err("attention sequence length must be greater than zero".to_string());
        }
        if self.head_dim == 0 {
            return Err("attention head dimension must be greater than zero".to_string());
        }
        Ok(())
    }
}

/// Host slices passed to the CUDA attention kernel.
pub struct AttentionSlices<'a> {
    pub queries: &'a [f32],
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub context_lengths: Option<&'a [u32]>,
    pub z_bias: Option<&'a [f32]>,
    pub attn_bias: Option<&'a [f32]>,
    pub attn_probs: Option<&'a mut [f32]>,
    pub output: &'a mut [f32],
}

impl<'a> AttentionSlices<'a> {
    pub fn validate(self, cfg: &AttentionConfig) -> Result<Self, String> {
        cfg.validate()?;
        let contexts = cfg.contexts();
        let seq = cfg.sequence_len as usize;
        let dim = cfg.head_dim as usize;
        let tokens = contexts
            .checked_mul(seq)
            .ok_or_else(|| "attention dimensions overflow rows".to_string())?;
        let expected_len = tokens
            .checked_mul(dim)
            .ok_or_else(|| "attention dimensions overflow elements".to_string())?;

        if self.queries.len() != expected_len {
            return Err(format!(
                "queries length {} does not match contexts×seq×dim {}",
                self.queries.len(),
                expected_len
            ));
        }
        if self.keys.len() != expected_len {
            return Err(format!(
                "keys length {} does not match contexts×seq×dim {}",
                self.keys.len(),
                expected_len
            ));
        }
        if self.values.len() != expected_len {
            return Err(format!(
                "values length {} does not match contexts×seq×dim {}",
                self.values.len(),
                expected_len
            ));
        }
        if self.output.len() != expected_len {
            return Err(format!(
                "output length {} does not match contexts×seq×dim {}",
                self.output.len(),
                expected_len
            ));
        }
        if let Some(lengths) = self.context_lengths {
            if lengths.len() != contexts {
                return Err(format!(
                    "context_lengths length {} does not match contexts {}",
                    lengths.len(),
                    contexts
                ));
            }
            if let Some(invalid) = lengths.iter().position(|&len| len > cfg.sequence_len) {
                return Err(format!(
                    "context_lengths[{}] = {} exceeds sequence length {}",
                    invalid, lengths[invalid], cfg.sequence_len
                ));
            }
        }
        if let Some(bias) = self.z_bias {
            if bias.len() != tokens {
                return Err(format!(
                    "z_bias length {} does not match contexts×seq {}",
                    bias.len(),
                    tokens
                ));
            }
        }
        if let Some(mask) = self.attn_bias {
            let expected = tokens
                .checked_mul(seq)
                .ok_or_else(|| "attention bias dimensions overflow".to_string())?;
            if mask.len() != expected {
                return Err(format!(
                    "attention bias length {} does not match contexts×seq×seq {}",
                    mask.len(),
                    expected
                ));
            }
        }
        if let Some(probs) = self.attn_probs.as_ref() {
            let expected = tokens
                .checked_mul(seq)
                .ok_or_else(|| "attention probability dimensions overflow".to_string())?;
            if probs.len() != expected {
                return Err(format!(
                    "attention probabilities length {} does not match contexts×seq×seq {}",
                    probs.len(),
                    expected
                ));
            }
        }
        Ok(self)
    }
}

/// Runs the scaled dot-product attention kernel on the active CUDA device.
pub fn run_attention(cfg: AttentionConfig, slices: AttentionSlices<'_>) -> Result<(), String> {
    let AttentionSlices {
        queries,
        keys,
        values,
        context_lengths,
        z_bias,
        attn_bias,
        attn_probs,
        output,
    } = slices.validate(&cfg)?;

    let contexts = cfg.contexts();
    let seq = cfg.sequence_len as usize;

    let shared_bytes = shared_mem_for_sequence(seq)?;

    let module = cuda_module()?;
    let device = module.device();
    let func = module.get_func(KERNEL_NAME)?;

    let queries = device
        .htod_sync_copy(queries)
        .map_err(|err| err.to_string())?;
    let keys = device.htod_sync_copy(keys).map_err(|err| err.to_string())?;
    let values = device
        .htod_sync_copy(values)
        .map_err(|err| err.to_string())?;
    let mut output = device
        .alloc_zeros::<f32>(output.len())
        .map_err(|err| err.to_string())?;

    let use_z_bias = z_bias.is_some();
    let z_bias = if let Some(bias) = z_bias {
        device.htod_sync_copy(bias).map_err(|err| err.to_string())?
    } else {
        device
            .alloc_zeros::<f32>(1)
            .map_err(|err| err.to_string())?
    };

    let use_context_lengths = context_lengths.is_some();
    let context_lengths = if let Some(lengths) = context_lengths {
        device
            .htod_sync_copy(lengths)
            .map_err(|err| err.to_string())?
    } else {
        device
            .alloc_zeros::<u32>(1)
            .map_err(|err| err.to_string())?
    };

    let use_attn_bias = attn_bias.is_some();
    let attn_bias = if let Some(mask) = attn_bias {
        device.htod_sync_copy(mask).map_err(|err| err.to_string())?
    } else {
        device
            .alloc_zeros::<f32>(1)
            .map_err(|err| err.to_string())?
    };

    let mut attn_probs = attn_probs;
    let use_attn_probs = attn_probs.is_some();
    let mut attn_probs_device = if use_attn_probs {
        let expected = contexts
            .checked_mul(seq)
            .and_then(|v| v.checked_mul(seq))
            .ok_or_else(|| "attention probabilities dimensions overflow device".to_string())?;
        device
            .alloc_zeros::<f32>(expected)
            .map_err(|err| err.to_string())?
    } else {
        device
            .alloc_zeros::<f32>(1)
            .map_err(|err| err.to_string())?
    };

    let grid_dim = grid_for_contexts(cfg.sequence_len, contexts)?;

    let cfg_launch = LaunchConfig {
        grid_dim,
        block_dim: (THREADS_PER_BLOCK, 1, 1),
        shared_mem_bytes: shared_bytes,
    };

    let contexts_i32 =
        i32::try_from(contexts).map_err(|_| "attention contexts exceed i32::MAX".to_string())?;

    unsafe {
        func.launch(
            cfg_launch,
            (
                &queries,
                &keys,
                &values,
                &context_lengths,
                &z_bias,
                &attn_bias,
                &mut attn_probs_device,
                &mut output,
                contexts_i32,
                cfg.sequence_len as i32,
                cfg.head_dim as i32,
                cfg.scale(),
                use_z_bias as i32,
                use_context_lengths as i32,
                use_attn_bias as i32,
                cfg.causal as i32,
                use_attn_probs as i32,
            ),
        )
        .map_err(|err| err.to_string())?;
    }

    let host_output: Vec<f32> = device
        .dtoh_sync_copy(&output)
        .map_err(|err| err.to_string())?;
    output.copy_from_slice(&host_output);

    if let Some(probs) = attn_probs.as_deref_mut() {
        let host_probs: Vec<f32> = device
            .dtoh_sync_copy(&attn_probs_device)
            .map_err(|err| err.to_string())?;
        probs.copy_from_slice(&host_probs);
    }

    Ok(())
}

fn cuda_module() -> Result<&'static CudaModule, String> {
    let ptx =
        COMPILED_PTX.get_or_try_init(|| compile_ptx(CUDA_SOURCE).map_err(|err| err.to_string()))?;
    CUDA_MODULE.get_or_try_init(|| cuda_loader::load_ptx_module(ptx, MODULE_NAME, MODULE_KERNELS))
}

fn shared_mem_for_sequence(seq_len: usize) -> Result<u32, String> {
    let floats = seq_len
        .checked_add(SCRATCH_SIZE)
        .ok_or_else(|| "shared memory requirements overflow".to_string())?;
    let bytes = floats
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "shared memory requirements overflow bytes".to_string())?;
    let bytes_u32: u32 = bytes
        .try_into()
        .map_err(|_| "shared memory requirement exceeds u32".to_string())?;
    if bytes_u32 > 49_152 {
        return Err(format!(
            "attention sequence length {} requires {} bytes of shared memory (exceeds 48KiB)",
            seq_len, bytes_u32
        ));
    }
    Ok(bytes_u32)
}

fn grid_for_contexts(sequence_len: u32, contexts: usize) -> Result<(u32, u32, u32), String> {
    const MAX_GRID_X: u32 = 2_147_483_647;
    const MAX_GRID_YZ: u32 = 65_535;
    const MAX_GRID_Z: u64 = 65_535;

    if sequence_len == 0 {
        return Err("attention sequence length must be greater than zero".to_string());
    }
    if sequence_len > MAX_GRID_X {
        return Err(format!(
            "attention sequence length {} exceeds CUDA grid.x capacity",
            sequence_len
        ));
    }
    if contexts == 0 {
        return Err("attention contexts must be greater than zero".to_string());
    }

    if contexts <= MAX_GRID_YZ as usize {
        return Ok((sequence_len, contexts as u32, 1));
    }

    let contexts64 = contexts as u64;
    let y = MAX_GRID_YZ as u64;
    let z_needed = (contexts64 + y - 1) / y;
    if z_needed > MAX_GRID_Z {
        return Err("attention contexts exceed CUDA grid capacity".to_string());
    }
    Ok((sequence_len, MAX_GRID_YZ, z_needed as u32));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_scale() {
        let cfg = AttentionConfig {
            batch: 2,
            heads: 4,
            z_levels: 1,
            sequence_len: 16,
            head_dim: 64,
            scale: None,
            causal: false,
        };
        let scale = cfg.scale();
        assert!((scale - (1.0f32 / 64.0f32).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn slices_validate_lengths() {
        let cfg = AttentionConfig {
            batch: 1,
            heads: 1,
            z_levels: 2,
            sequence_len: 3,
            head_dim: 4,
            scale: Some(1.0),
            causal: false,
        };
        let contexts = cfg.contexts();
        let seq = cfg.sequence_len as usize;
        let dim = cfg.head_dim as usize;
        let len = contexts * seq * dim;
        let bias_len = contexts * seq;
        let mask_len = bias_len * seq;
        let mut output = vec![0.0f32; len];
        let slices = AttentionSlices {
            queries: &vec![0.0; len],
            keys: &vec![0.0; len],
            values: &vec![0.0; len],
            context_lengths: Some(&vec![seq as u32; contexts]),
            z_bias: Some(&vec![0.0; bias_len]),
            attn_bias: Some(&vec![0.0; mask_len]),
            attn_probs: None,
            output: &mut output,
        };
        slices.validate(&cfg).unwrap();
    }

    #[test]
    fn slices_validate_attn_probs_length() {
        let cfg = AttentionConfig {
            batch: 1,
            heads: 2,
            z_levels: 1,
            sequence_len: 4,
            head_dim: 8,
            scale: None,
            causal: true,
        };
        let contexts = cfg.contexts();
        let seq = cfg.sequence_len as usize;
        let dim = cfg.head_dim as usize;
        let len = contexts * seq * dim;
        let mut output = vec![0.0f32; len];
        let mut probs = vec![0.0f32; contexts * seq * seq];
        let slices = AttentionSlices {
            queries: &vec![0.0; len],
            keys: &vec![0.0; len],
            values: &vec![0.0; len],
            context_lengths: None,
            z_bias: None,
            attn_bias: None,
            attn_probs: Some(&mut probs),
            output: &mut output,
        };
        slices.validate(&cfg).unwrap();
    }

    #[test]
    fn grid_splits_large_contexts() {
        let result = grid_for_contexts(32, 70_000).unwrap();
        assert_eq!(result.0, 32);
        assert_eq!(result.1, 65_535);
        assert!(result.2 >= 2);
    }
}

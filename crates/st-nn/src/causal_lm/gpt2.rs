//! Checkpoint naming/orientation belongs here, not in the generic decoder loop.
use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use st_tensor::Tile;
use std::collections::{BTreeMap, BTreeSet};

fn yes() -> bool {
    true
}
fn epsilon() -> f32 {
    1e-5
}
fn activation() -> String {
    "gelu_new".into()
}

#[derive(Deserialize)]
struct Gpt2Config {
    model_type: String,
    vocab_size: usize,
    n_positions: usize,
    n_embd: usize,
    n_head: usize,
    n_layer: usize,
    n_inner: Option<usize>,
    #[serde(default = "epsilon")]
    layer_norm_epsilon: f32,
    #[serde(default = "activation")]
    activation_function: String,
    #[serde(default = "yes")]
    scale_attn_weights: bool,
    #[serde(default)]
    scale_attn_by_inverse_layer_idx: bool,
    #[serde(default = "yes")]
    tie_word_embeddings: bool,
    #[serde(default)]
    add_cross_attention: bool,
    #[serde(default)]
    reorder_and_upcast_attn: bool,
    #[serde(default)]
    pruned_heads: BTreeMap<String, Vec<usize>>,
}

fn config(bytes: &[u8], limits: CausalLmLimits) -> CausalLmResult<CausalLmConfig> {
    limits.validate()?;
    if bytes.len() > 131_072 {
        return Err("GPT-2 config exceeds byte budget".into());
    }
    let value: Gpt2Config = serde_json::from_slice(bytes)?;
    if value.model_type != "gpt2"
        || value.activation_function != "gelu_new"
        || value.add_cross_attention
        || value.reorder_and_upcast_attn
        || value.pruned_heads.values().any(|heads| !heads.is_empty())
    {
        return Err(
            "unsupported GPT-2 architecture, activation, pruned heads or attention mode".into(),
        );
    }
    let intermediate = value
        .n_inner
        .unwrap_or(value.n_embd.checked_mul(4).ok_or("MLP width overflow")?);
    if [
        value.vocab_size,
        value.n_positions,
        value.n_embd,
        value.n_head,
        value.n_layer,
        intermediate,
    ]
    .contains(&0)
        || !value.n_embd.is_multiple_of(value.n_head)
        || value.n_layer > 1024
        || !value.layer_norm_epsilon.is_finite()
        || value.layer_norm_epsilon <= 0.
    {
        return Err("invalid native causal LM dimensions or normalization epsilon".into());
    }
    product(
        &[value.n_embd, 3],
        limits.max_parameter_elements,
        "QKV width",
    )?;
    product(
        &[value.vocab_size, value.n_embd],
        limits.max_parameter_elements,
        "token table",
    )?;
    product(
        &[value.n_positions, value.n_embd],
        limits.max_parameter_elements,
        "position table",
    )?;
    product(
        &[value.n_embd, intermediate],
        limits.max_parameter_elements,
        "MLP parameters",
    )?;
    Ok(CausalLmConfig {
        vocabulary: value.vocab_size,
        positions: value.n_positions,
        width: value.n_embd,
        heads: value.n_head,
        layers: value.n_layer,
        intermediate,
        norm_epsilon: value.layer_norm_epsilon,
        scale_attention: value.scale_attn_weights,
        inverse_layer_scale: value.scale_attn_by_inverse_layer_idx,
        tied_head: value.tie_word_embeddings,
    })
}

fn matrix(
    weights: &SafeTensorArchive,
    name: &str,
    rows: usize,
    cols: usize,
) -> CausalLmResult<Tensor> {
    Ok(Tensor::from_vec(rows, cols, weights.f32_values(name)?)?)
}

fn norm(weights: &SafeTensorArchive, prefix: &str, width: usize) -> CausalLmResult<Norm> {
    Ok(Norm {
        gamma: matrix(weights, &format!("{prefix}.weight"), 1, width)?,
        beta: matrix(weights, &format!("{prefix}.bias"), 1, width)?,
    })
}

fn dense(
    weights: &SafeTensorArchive,
    prefix: &str,
    input: usize,
    output: usize,
) -> CausalLmResult<Dense> {
    let weight = matrix(weights, &format!("{prefix}.weight"), input, output)?;
    Ok(Dense {
        weight: PackedB::from_tensor(&weight, Tile::col_major())?,
        backward_weight: Arc::new(OnceLock::new()),
        bias: matrix(weights, &format!("{prefix}.bias"), 1, output)?,
    })
}

fn legacy_mask(weights: &SafeTensorArchive, name: &str, positions: usize) -> CausalLmResult<()> {
    let info = weights.info(name)?;
    if info.shape != [1, 1, positions, positions] {
        return Err(format!("legacy attention mask has wrong shape: {name}").into());
    }
    let valid = if matches!(info.dtype.as_str(), "BOOL" | "U8") {
        weights
            .tensor_bytes(name)?
            .iter()
            .enumerate()
            .all(|(i, &v)| v == u8::from(i % positions <= i / positions))
    } else if matches!(info.dtype.as_str(), "F32" | "F16" | "BF16") {
        weights.f32_values(name)?.iter().enumerate().all(|(i, &v)| {
            v == if i % positions <= i / positions {
                1.
            } else {
                0.
            }
        })
    } else {
        false
    };
    if !valid {
        return Err(
            format!("legacy attention mask is not the canonical causal triangle: {name}").into(),
        );
    }
    Ok(())
}

pub(super) fn load(
    bytes: &[u8],
    weights: &SafeTensorArchive,
    limits: CausalLmLimits,
) -> CausalLmResult<NativeCausalLm> {
    let c = config(bytes, limits)?;
    let names: BTreeSet<_> = weights.names().into_iter().collect();
    let prefix = match (
        names.contains("wte.weight"),
        names.contains("transformer.wte.weight"),
    ) {
        (true, false) => "",
        (false, true) => "transformer.",
        _ => return Err("expected exactly one GPT-2 checkpoint naming convention".into()),
    };
    let mut expected = BTreeMap::from([
        (format!("{prefix}wte.weight"), vec![c.vocabulary, c.width]),
        (format!("{prefix}wpe.weight"), vec![c.positions, c.width]),
        (format!("{prefix}ln_f.weight"), vec![c.width]),
        (format!("{prefix}ln_f.bias"), vec![c.width]),
    ]);
    for index in 0..c.layers {
        let p = format!("{prefix}h.{index}");
        for n in ["ln_1", "ln_2"] {
            for suffix in ["weight", "bias"] {
                expected.insert(format!("{p}.{n}.{suffix}"), vec![c.width]);
            }
        }
        for (n, input, output) in [
            ("attn.c_attn", c.width, c.width * 3),
            ("attn.c_proj", c.width, c.width),
            ("mlp.c_fc", c.width, c.intermediate),
            ("mlp.c_proj", c.intermediate, c.width),
        ] {
            expected.insert(format!("{p}.{n}.weight"), vec![input, output]);
            expected.insert(format!("{p}.{n}.bias"), vec![output]);
        }
    }
    if !c.tied_head || names.contains("lm_head.weight") {
        expected.insert("lm_head.weight".into(), vec![c.vocabulary, c.width]);
    }
    let mut total = 0usize;
    for (name, shape) in &expected {
        let info = weights.info(name)?;
        if info.shape != *shape || !matches!(info.dtype.as_str(), "F32" | "F16" | "BF16") {
            return Err(format!("invalid GPT-2 weight shape or dtype: {name}").into());
        }
        total = total
            .checked_add(info.elements)
            .filter(|&n| n <= limits.max_parameter_elements)
            .ok_or("native causal LM parameters exceed budget")?;
    }
    // Legacy persistent masks are redundant only after their values are checked.
    // Other extras (including unimported LoRA weights) are never silently dropped.
    for name in names.iter().filter(|n| !expected.contains_key(*n)) {
        let layer = (0..c.layers).find(|i| *name == format!("{prefix}h.{i}.attn.bias"));
        if layer.is_some() {
            legacy_mask(weights, name, c.positions)?;
        } else {
            return Err(format!("unconsumed GPT-2 checkpoint tensor: {name}").into());
        }
    }
    let token = matrix(
        weights,
        &format!("{prefix}wte.weight"),
        c.vocabulary,
        c.width,
    )?;
    let position = matrix(
        weights,
        &format!("{prefix}wpe.weight"),
        c.positions,
        c.width,
    )?;
    let head = if c.tied_head {
        if names.contains("lm_head.weight") && weights.f32_values("lm_head.weight")? != token.data()
        {
            return Err("tied lm_head.weight differs from token embedding".into());
        }
        PackedB::from_tensor_transpose(&token, Tile::col_major())?
    } else {
        PackedB::from_tensor_transpose(
            &matrix(weights, "lm_head.weight", c.vocabulary, c.width)?,
            Tile::col_major(),
        )?
    };
    let mut blocks = Vec::with_capacity(c.layers);
    for index in 0..c.layers {
        let p = format!("{prefix}h.{index}");
        blocks.push(DecoderBlock {
            first_norm: norm(weights, &format!("{p}.ln_1"), c.width)?,
            qkv: dense(weights, &format!("{p}.attn.c_attn"), c.width, c.width * 3)?,
            output: dense(weights, &format!("{p}.attn.c_proj"), c.width, c.width)?,
            second_norm: norm(weights, &format!("{p}.ln_2"), c.width)?,
            expand: dense(weights, &format!("{p}.mlp.c_fc"), c.width, c.intermediate)?,
            contract: dense(weights, &format!("{p}.mlp.c_proj"), c.intermediate, c.width)?,
        });
    }
    let final_norm = norm(weights, &format!("{prefix}ln_f"), c.width)?;
    let mut identity = Sha256::new();
    identity.update(b"spiraltorch.native_causal_lm.gpt2.f32.v1\0");
    identity.update(serde_json::to_vec(&c)?);
    identity.update(weights.content_id());
    Ok(NativeCausalLm {
        config: c,
        limits,
        base_id: format!("sha256:{:x}", identity.finalize()),
        token,
        position,
        blocks,
        final_norm,
        head,
        backward_head: Arc::new(OnceLock::new()),
        geometry: None,
    })
}

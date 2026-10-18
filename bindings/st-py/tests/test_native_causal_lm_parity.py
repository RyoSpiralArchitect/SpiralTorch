"""Optional Transformers numerical reference; never imported by the native runtime."""
import json

import pytest

torch = pytest.importorskip("torch")
safe_torch = pytest.importorskip("safetensors.torch")
transformers = pytest.importorskip("transformers")
import spiraltorch as st  # noqa: E402


def pair(*, dtype=torch.float32, tied=True, scaled=True, inverse=False):
    torch.manual_seed(327)
    config = transformers.GPT2Config(vocab_size=19, n_positions=12, n_embd=8, n_head=2,
                                    n_layer=2, n_inner=13, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
                                    tie_word_embeddings=tied, scale_attn_weights=scaled,
                                    scale_attn_by_inverse_layer_idx=inverse)
    config._attn_implementation = "eager"
    reference = transformers.GPT2LMHeadModel(config).eval()
    state = {k: v.to(dtype).clone().contiguous() for k, v in reference.state_dict().items()}
    reference.load_state_dict({k: v.float() for k, v in state.items()})
    weights = st.SafeTensorArchive.from_bytes(safe_torch.save(state))
    native = st.NativeCausalLm.from_gpt2(config.to_json_string(), weights)
    return reference, native


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tied,scaled,inverse", [(True, True, False), (False, True, True), (True, False, False)])
def test_complete_logits_last_projection_and_nll(dtype, tied, scaled, inverse):
    reference, native = pair(dtype=dtype, tied=tied, scaled=scaled, inverse=inverse)
    tokens = torch.tensor([[1, 5, 7, 3], [2, 8, 4, 6]])
    with torch.no_grad():
        expected = reference(tokens, labels=tokens, use_cache=False)
    result = native.forward(tokens.flatten().tolist(), list(tokens.shape))
    actual = torch.frombuffer(result.values_buffer(), dtype=torch.float32).reshape(result.shape)
    torch.testing.assert_close(actual, expected.logits, atol=2e-6, rtol=2e-5)
    last = native.forward(tokens.flatten().tolist(), list(tokens.shape), last_only=True)
    torch.testing.assert_close(torch.frombuffer(last.values_buffer(), dtype=torch.float32).reshape(2, 19),
                               expected.logits[:, -1], atol=2e-6, rtol=2e-5)
    assert native.next_token_ids(tokens.flatten().tolist(), list(tokens.shape)) == expected.logits[:, -1].argmax(-1).tolist()
    assert native.next_token_loss(tokens.flatten().tolist(), list(tokens.shape)) == pytest.approx(expected.loss.item(), abs=5e-6)


@pytest.mark.parametrize("site", ["attn", "mlp", "block"])
def test_native_geometry_site_matches_explicit_torch_output_hook(site):
    reference, native = pair()
    width = 8
    gate = torch.arange(width).float() * .03 - .1
    bias = torch.arange(width).float() * .01 - .03
    path = "transformer.h.0" + ("." + site if site != "block" else "")
    artifact = st.GeometryAdapterArtifact.create_json(json.dumps({
        "schema": "spiraltorch.geometry_adapter_artifact.v1", "base_model_identity": native.base_model_identity,
        "policy": "full_prefix_f32_no_cache.v1", "training": False, "placements": [{
            "path": path, "training": False,
            "recipe": {"schema": "spiraltorch.channel_affine_adapter.v1", "features": width, "strength": .5, "max_values": 1024},
            "tensors": [{"name": name, "shape": [width], "values": values.tolist(), "requires_grad": False, "role": "parameter"}
                        for name, values in [("gate", gate), ("bias", bias)]],
        }],
    }))
    adapted = native.with_geometry(artifact)
    module = reference.transformer.h[0]
    if site != "block":
        module = getattr(module, site)

    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        changed = value + .5 * (value * gate + bias)
        return (changed,) + output[1:] if isinstance(output, tuple) else changed

    handle = module.register_forward_hook(hook)
    tokens = torch.tensor([[1, 5, 7], [2, 8, 4]])
    try:
        with torch.no_grad():
            expected = reference(tokens, use_cache=False).logits
    finally:
        handle.remove()
    result = adapted.forward(tokens.flatten().tolist(), list(tokens.shape))
    actual = torch.frombuffer(result.values_buffer(), dtype=torch.float32).reshape(result.shape)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)

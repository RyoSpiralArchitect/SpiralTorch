"""Optional whole-decoder and cache reference; geometry uses full-prefix Torch hooks."""
import json
from pathlib import Path
import runpy

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
helpers = runpy.run_path(str(Path(__file__).with_name("test_native_causal_lm_learning_parity.py")))


def values(output):
    return torch.frombuffer(output.values_buffer(), dtype=torch.float32).reshape(output.shape)


@pytest.mark.parametrize("kind", (None, *helpers["KINDS"]))
@pytest.mark.parametrize("suffix", ("attn", "mlp", "block"))
def test_cached_native_all_geometries_match_full_prefix_torch(kind, suffix):
    reference, base = helpers["pair"](width=3, heads=1)
    reference.requires_grad_(False)
    native, handles = base, []
    if kind:
        record = json.loads(helpers["fixtures"]["artifact"](kind).state_json())
        record["base_model_identity"] = base.base_model_identity
        record["placements"][0]["path"] = "transformer.h.0" + ("." + suffix if suffix != "block" else "")
        if kind == "affine":
            record["placements"][0]["recipe"]["max_values"] = 96
        state = st.GeometryAdapterArtifact.create_json(json.dumps(record))
        native = base.with_geometry(state)
        stack, handles = helpers["attach"](reference, state)
    context = st.NativeCausalLmContext(native, [1, 2, 5], max_cache_values=1024)
    try:
        for token in (3, 9, 8, 0, None):
            with torch.no_grad():
                expected = reference(torch.tensor([context.token_ids]), use_cache=False).logits[:, -1:]
            torch.testing.assert_close(values(context.logits()), expected, atol=3e-6, rtol=3e-5)
            if token is not None:
                context.append(token)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("tied,scaled,inverse", [(True, True, False), (False, True, True), (True, False, False)])
def test_ordinary_cached_native_matches_torch_cache(dtype, tied, scaled, inverse):
    reference, native = helpers["pair"](dtype=dtype, tied=tied, scaled=scaled, inverse=inverse)
    context = st.NativeCausalLmContext(native, [1, 5, 7])
    with torch.no_grad():
        result = reference(torch.tensor([[1, 5, 7]]), use_cache=True)
        torch.testing.assert_close(values(context.logits()), result.logits[:, -1:], atol=3e-6, rtol=3e-5)
        for token in (3, 9, 8, 0):
            result = reference(torch.tensor([[token]]), past_key_values=result.past_key_values, use_cache=True)
            torch.testing.assert_close(values(context.append(token)), result.logits, atol=3e-6, rtol=3e-5)


def test_joint_causal_and_pointwise_placements_keep_one_frozen_snapshot():
    reference, base = helpers["pair"](width=3, heads=1)
    record = json.loads(helpers["fixtures"]["artifact"]("wave").state_json())
    record["base_model_identity"] = base.base_model_identity
    record["placements"] = []
    for kind, path in [("learned", "transformer.h.0.attn"), ("history_angle", "transformer.h.0.mlp"),
                       ("gated", "transformer.h.1.attn"), ("topos", "transformer.h.1.mlp")]:
        placement = json.loads(helpers["fixtures"]["artifact"](kind).state_json())["placements"][0]
        placement["path"] = path
        record["placements"].append(placement)
    state = st.GeometryAdapterArtifact.create_json(json.dumps(record))
    native = base.with_geometry(state)
    stack, handles = helpers["attach"](reference, state)
    context = st.NativeCausalLmContext(native, [1, 2])
    try:
        for token in (3, 4, 5, 6, 7):
            context.append(token)
            with torch.no_grad():
                expected = reference(torch.tensor([context.token_ids]), use_cache=False).logits[:, -1:]
            torch.testing.assert_close(values(context.logits()), expected, atol=3e-6, rtol=3e-5)
    finally:
        for handle in handles:
            handle.remove()

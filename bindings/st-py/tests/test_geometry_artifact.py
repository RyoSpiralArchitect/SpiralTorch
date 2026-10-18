"""Actual native artifact -> reconstructed HF geometry, not pickle replay."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
BASE = "sha256:" + "a" * 64


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                                right.detach().contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


CASES = [
    ("ChannelAffineAdapter", {}),
    ("WaveGateAdapter", {}),
    ("WaveGateAdapter", {"log_radius": .2, "learnable_radius": True}),
    ("WaveGateAdapter", {"log_radius": .2}),
    ("ToposResonatorAdapter", {}),
    ("EllipticResidualAdapter", {}),
    ("EllipticCausalResidualAdapter", {}),
    ("EllipticGatedCausalResidualAdapter", {"raw_mix": .2}),
    ("EllipticAnchoredResidualAdapter", {"raw_mix": -.2}),
    ("FractionalMemoryAdapter", {"kernel_len": 5}),
    ("FractionalHistoryAdapter", {"kernel_len": 5}),
    ("FractionalL2HistoryAdapter", {"kernel_len": 5, "gain": .7}),
    ("FractionalGainHistoryAdapter", {"kernel_len": 5, "lag_window": (1, 4)}),
    ("FractionalAngleGainHistoryAdapter", {"kernel_len": 5, "initial_angle": .2, "lag_window": (2, 4)}),
]


@pytest.mark.parametrize("name,options", CASES)
def test_all_builtin_variants_preserve_state_forward_vjp_flags_and_rng(name, options):
    torch.manual_seed(13)
    adapter = getattr(st, name)(4, strength=.3, **options).eval()
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.add_(.03)
    next(adapter.parameters()).requires_grad_(False)
    stack = st.GeometryAdapterStack({"block.mlp": adapter})
    artifact = stack.export_artifact(base_model_identity=BASE)
    assert artifact.artifact_id == "sha256:" + hashlib.sha256(artifact.to_json().encode()).hexdigest()
    loaded = st.GeometryAdapterArtifact.from_json(artifact.to_json(), artifact.artifact_id, BASE)
    rng = torch.get_rng_state().clone()
    restored = st.GeometryAdapterStack.from_artifact(loaded, expected_base_model_identity=BASE)
    assert torch.equal(rng, torch.get_rng_state())
    assert equal(stack.state_dict(), restored.state_dict())
    assert [p.requires_grad for p in stack.parameters()] == [p.requires_grad for p in restored.parameters()]
    assert restored.training and not restored.adapters[0].training
    assert artifact.to_json() == restored.export_artifact(base_model_identity=BASE).to_json()
    x = torch.linspace(-.4, .6, 48).reshape(2, 6, 4).requires_grad_()
    y = x.detach().clone().requires_grad_()
    expected, actual = adapter(x), restored.adapters[0](y)
    assert equal(expected, actual)
    expected.square().sum().backward()
    actual.square().sum().backward()
    assert equal(x.grad, y.grad)
    assert all(equal(a.grad, b.grad) for a, b in zip(adapter.parameters(), restored.adapters[0].parameters()))


def test_wrong_base_hash_bad_state_and_active_stack_fail_before_any_reconstruction():
    stack = st.GeometryAdapterStack({"0": st.WaveGateAdapter(4)})
    artifact = stack.export_artifact(base_model_identity=BASE)
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="different base"):
        st.GeometryAdapterStack.from_artifact(artifact, expected_base_model_identity="sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="pinned identity"):
        st.GeometryAdapterArtifact.from_json(artifact.to_json() + " ", artifact.artifact_id, BASE)
    with pytest.raises(ValueError, match="different base"):
        st.GeometryAdapterArtifact.from_json(artifact.to_json(), artifact.artifact_id, "sha256:" + "b" * 64)
    with pytest.raises(TypeError, match="Rust-validated"):
        st.GeometryAdapterStack.from_artifact(json.loads(artifact.to_json()), expected_base_model_identity=BASE)
    assert torch.equal(rng, torch.get_rng_state())
    with stack.attach(torch.nn.Sequential(torch.nn.Identity())):
        with pytest.raises(RuntimeError, match="detach"):
            stack.export_artifact(base_model_identity=BASE)
    stack.double()
    with pytest.raises(TypeError, match="float32"):
        stack.export_artifact(base_model_identity=BASE)
    class Custom(st.WaveGateAdapter):
        pass
    with pytest.raises(TypeError, match="exact built-in"):
        st.GeometryAdapterStack({"0": Custom(4)}).export_artifact(base_model_identity=BASE)
    with pytest.raises(ValueError, match="non-finite|JSON compliant"):
        stack = st.GeometryAdapterStack({"0": st.WaveGateAdapter(4)})
        with torch.no_grad():
            stack.adapters[0].gate[0] = float("nan")
        stack.export_artifact(base_model_identity=BASE)


@pytest.mark.parametrize("architecture", ["gpt2", "llama"])
def test_hf_save_load_inference_and_exact_adamw_continuation(architecture):
    hf = pytest.importorskip("transformers")
    torch.manual_seed(41)
    torch.set_num_threads(2)
    if architecture == "gpt2":
        config = hf.GPT2Config(vocab_size=32, n_positions=8, n_embd=8, n_layer=4, n_head=2,
                              resid_pdrop=0., embd_pdrop=0., attn_pdrop=0., use_cache=False)
        model = hf.GPT2LMHeadModel(config)
        paths = [f"transformer.h.{i}.mlp" for i in range(4)]
    else:
        config = hf.LlamaConfig(vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=4,
                               num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16,
                               attention_dropout=0., use_cache=False)
        model = hf.LlamaForCausalLM(config)
        paths = [f"model.layers.{i}.mlp" for i in range(4)]
    model.eval().requires_grad_(False)
    base_state = copy.deepcopy(model.state_dict())
    # Test-local model identity; production uses the HF runtime identity report.
    digest = hashlib.sha256(config.to_json_string().encode())
    for name, value in base_state.items():
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    base = "sha256:" + digest.hexdigest()
    stack = st.GeometryAdapterStack(dict(zip(paths, (
        st.WaveGateAdapter(8, log_radius=0., learnable_radius=True),
        st.ToposResonatorAdapter(8), st.EllipticAnchoredResidualAdapter(8),
        st.FractionalAngleGainHistoryAdapter(8, kernel_len=5, initial_angle=.2),
    ))))
    optimizer = torch.optim.AdamW(stack.parameters(), lr=.01, foreach=False)
    batch = torch.arange(12).reshape(2, 6) % 32

    def step(geometry, opt):
        with geometry.attach(model):
            opt.zero_grad(set_to_none=True)
            out = model(batch, labels=batch, use_cache=False)
            out.loss.backward()
            gradients = [p.grad.clone() for p in geometry.parameters()]
            opt.step()
            return out.logits.detach().clone(), out.loss.detach().clone(), gradients

    for _ in range(2):
        step(stack, optimizer)
    artifact = stack.export_artifact(base_model_identity=base)
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    rng = torch.get_rng_state().clone()
    restored = st.GeometryAdapterStack.from_artifact(
        st.GeometryAdapterArtifact.from_json(artifact.to_json(), artifact.artifact_id, base),
        expected_base_model_identity=base)
    assert torch.equal(rng, torch.get_rng_state())
    with torch.no_grad(), stack.attach(model):
        logits = model(batch, use_cache=False).logits
    with torch.no_grad(), restored.attach(model):
        assert equal(logits, model(batch, use_cache=False).logits)
    other = torch.optim.AdamW(restored.parameters(), lr=.01, foreach=False)
    other.load_state_dict(optimizer_state)
    expected = step(stack, optimizer)
    actual = step(restored, other)
    assert equal(expected, actual)
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in actual[2])
    assert equal(stack.state_dict(), restored.state_dict())
    assert equal(optimizer.state_dict(), other.state_dict())
    assert equal(base_state, model.state_dict())
    assert all(p.grad is None for p in model.parameters())


def test_reconstruction_in_fresh_isolated_process():
    stack = st.GeometryAdapterStack({"block": st.FractionalAngleGainHistoryAdapter(4, kernel_len=5, lag_window=(1, 4))})
    artifact = stack.export_artifact(base_model_identity=BASE)
    script = """
import json, sys, spiraltorch as st
data = json.load(sys.stdin)
a = st.GeometryAdapterArtifact.from_json(data['json'], data['id'], data['base'])
s = st.GeometryAdapterStack.from_artifact(a, expected_base_model_identity=data['base'])
assert s.export_artifact(base_model_identity=data['base']).to_json() == a.to_json()
print(a.artifact_id)
"""
    run = subprocess.run([sys.executable, "-I", "-B", "-c", script],
                         input=json.dumps({"json": artifact.to_json(), "id": artifact.artifact_id, "base": BASE}),
                         text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == artifact.artifact_id


def test_noncanonical_input_reconstructs_rust_normalized_state():
    stack = st.GeometryAdapterStack({"block": st.FractionalGainHistoryAdapter(4)})
    artifact = stack.export_artifact(base_model_identity=BASE)
    value = json.loads(artifact.to_json())
    value["placements"][0]["recipe"]["lag_window"] = None
    payload = json.dumps(value, indent=2)
    loaded = st.GeometryAdapterArtifact.from_json(payload, "sha256:" + hashlib.sha256(payload.encode()).hexdigest(), BASE)
    assert loaded.to_json() == payload
    restored = st.GeometryAdapterStack.from_artifact(loaded, expected_base_model_identity=BASE)
    assert equal(stack.state_dict(), restored.state_dict())


def test_shared_native_wasm_fixture_reconstructs_all_thirteen_placements():
    path = Path(__file__).resolve().parents[3] / "crates/st-nn/tests/fixtures/geometry_adapter_artifact.json"
    payload = path.read_text()
    expected = json.loads(path.with_suffix(".expected.json").read_text())
    assert "sha256:" + hashlib.sha256(payload.encode()).hexdigest() == expected["fixture_id"]
    artifact = st.GeometryAdapterArtifact.from_json(payload, "sha256:" + hashlib.sha256(payload.encode()).hexdigest(), BASE)
    restored = st.GeometryAdapterStack.from_artifact(artifact, expected_base_model_identity=BASE)
    assert len(restored.adapters) == 13
    assert len(restored.paths) == 13
    canonical = st.GeometryAdapterArtifact.create_json(payload)
    assert canonical.artifact_id == expected["canonical_artifact_id"]
    assert canonical.layout_id == expected["layout_id"]
    assert restored.export_artifact(base_model_identity=BASE).artifact_id == canonical.artifact_id
    assert equal(json.loads(canonical.state_json()), json.loads(restored.export_artifact(base_model_identity=BASE).state_json()))


@pytest.mark.parametrize("name", ["gate", "bias"])
def test_export_rejects_parameter_buffer_role_changes(name):
    adapter = st.WaveGateAdapter(4)
    parameter = getattr(adapter, name)
    delattr(adapter, name)
    adapter.register_buffer(name, parameter.detach().requires_grad_())
    with pytest.raises(ValueError, match="role"):
        st.GeometryAdapterStack({"0": adapter}).export_artifact(base_model_identity=BASE)


@pytest.mark.parametrize("name,options", [
    ("FractionalMemoryAdapter", {"initial_alpha": 2., "step": 1e-30, "kernel_len": 4}),
    ("FractionalHistoryAdapter", {"initial_alpha": 2., "step": 1e-30, "kernel_len": 4}),
    ("FractionalMemoryAdapter", {"initial_alpha": 200., "kernel_len": 256}),
    ("FractionalL2HistoryAdapter", {"initial_alpha": 200., "kernel_len": 256}),
    ("FractionalGainHistoryAdapter", {"initial_alpha": 200., "kernel_len": 256}),
])
def test_export_rejects_input_independent_fractional_overflow(name, options):
    adapter = getattr(st, name)(2, **options)
    with pytest.raises(ValueError, match="f32|finite|represent"):
        st.GeometryAdapterStack({"0": adapter}).export_artifact(base_model_identity=BASE)


def test_normalized_and_empty_history_do_not_inherit_irrelevant_scale_guards():
    for adapter in (
        st.FractionalL2HistoryAdapter(2, initial_alpha=2., step=1e-30, kernel_len=4),
        st.FractionalGainHistoryAdapter(2, initial_alpha=200., kernel_len=256, lag_window=(2, 2)),
        st.FractionalHistoryAdapter(2, initial_alpha=200., step=1e-30, kernel_len=1),
    ):
        artifact = st.GeometryAdapterStack({"0": adapter}).export_artifact(base_model_identity=BASE)
        restored = st.GeometryAdapterStack.from_artifact(artifact, expected_base_model_identity=BASE)
        assert equal(restored.adapters[0](torch.zeros(1, 2, 2)), torch.zeros(1, 2, 2))
    with pytest.raises(ValueError, match="tap budget"):
        st.GeometryAdapterStack({"0": st.FractionalHistoryAdapter(2, kernel_len=65_537)}).export_artifact(base_model_identity=BASE)

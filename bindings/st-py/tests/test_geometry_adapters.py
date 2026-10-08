import copy
from contextlib import contextmanager
import io
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.shape == right.shape and left.dtype == right.dtype
                and torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                                right.contiguous().reshape(-1).view(torch.uint8)))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def simple():
    torch.manual_seed(277)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).eval().requires_grad_(False)
    stack = st.GeometryAdapterStack({"0": st.ToposResonatorAdapter(4), "1": st.WaveGateAdapter(4)})
    return model, stack


def mixed(features, paths):
    return st.GeometryAdapterStack(dict(zip(paths, (
        st.WaveGateAdapter(features, log_radius=0., learnable_radius=True),
        st.ToposResonatorAdapter(features),
        st.EllipticAnchoredResidualAdapter(features),
        st.FractionalAngleGainHistoryAdapter(features, kernel_len=5, initial_angle=.2),
    ))))


@contextmanager
def manual_placement(model, stack):
    targets = []
    try:
        for path, adapter in zip(stack.paths, stack.adapters):
            prefix, child = path.rsplit(".", 1)
            parent = model.get_submodule(prefix)
            original = parent.get_submodule(child)
            parent.add_module(child, torch.nn.Sequential(original, adapter))
            targets.append((parent, child, original))
        yield
    finally:
        for parent, child, original in reversed(targets):
            parent.add_module(child, original)


def tiny_hf(architecture):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(281)
    torch.set_num_threads(2)
    if architecture == "gpt2":
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=32, n_positions=8, n_embd=8, n_layer=4, n_head=2,
            resid_pdrop=0., attn_pdrop=0., embd_pdrop=0., use_cache=False))
        paths = [f"transformer.h.{i}.mlp" for i in range(4)]
    else:
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(
            vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16,
            attention_dropout=0., use_cache=False))
        paths = [f"model.layers.{i}.mlp" for i in range(4)]
    return model.eval().requires_grad_(False), mixed(8, paths)


def update(model, stack, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = model(batch, labels=batch, use_cache=False).loss
    loss.backward()
    gradients = {n: p.grad.clone() for n, p in stack.named_parameters()}
    assert all(torch.isfinite(p).all() for p in gradients.values())
    optimizer.step()
    return loss.detach(), gradients


@pytest.mark.parametrize("architecture", ["gpt2", "llama"])
def test_mixed_rust_adapters_match_manual_placement_and_resume(architecture):
    model, stack = tiny_hf(architecture)
    reference, manual = tiny_hf(architecture)
    base = copy.deepcopy(model.state_dict())
    original_names = list(dict(model.named_parameters()))
    optimizer = torch.optim.Adam(stack.parameters(), lr=.01, foreach=False)
    other = torch.optim.Adam(manual.parameters(), lr=.01, foreach=False)
    batches = [(torch.arange(12).reshape(2, 6) + i) % 32 for i in range(3)]
    with stack.attach(model), manual_placement(reference, manual):
        assert list(dict(model.named_parameters())) == original_names
        for index, batch in enumerate(batches):
            observed = update(model, stack, optimizer, batch)
            expected = update(reference, manual, other, batch)
            assert equal(observed, expected)
            assert equal(stack.state_dict(), manual.state_dict())
            assert equal(optimizer.state_dict(), other.state_dict())
            if index == 1:
                saved = copy.deepcopy({"adapter": stack.state_dict(), "optimizer": optimizer.state_dict()})
            if index == 2:
                assert all(torch.count_nonzero(p) for p in observed[1].values())
    assert equal(base, model.state_dict()) and equal(base, reference.state_dict())
    assert all(p.grad is None for p in model.parameters())
    blob = io.BytesIO()
    torch.save(saved, blob)
    blob.seek(0)
    restored = torch.load(blob, weights_only=True)
    resumed_model, resumed = tiny_hf(architecture)
    resumed.load_state_dict(restored["adapter"])
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=.01, foreach=False)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    with resumed.attach(resumed_model):
        assert equal(update(resumed_model, resumed, resumed_optimizer, batches[2]), observed)
    assert equal(resumed.state_dict(), stack.state_dict())
    assert equal(resumed_optimizer.state_dict(), optimizer.state_dict())


def test_scoped_attachment_preserves_identity_modes_hooks_and_frozen_flags():
    model, stack = simple()
    x = torch.ones(2, 4)
    modules = dict(model.named_modules())
    flags = [(p.requires_grad, p.device, p.dtype) for p in model.parameters()]
    seen = []
    user_hook = model[0].register_forward_hook(lambda _m, _a, out: seen.append(out.shape))
    original_hooks = set(model[0]._forward_hooks)
    before = model(x)
    try:
        for _ in range(2):
            with stack.attach(model) as active:
                assert active is stack and equal(model(x), before)
                assert dict(model.named_modules()) == modules
                assert [(p.requires_grad, p.device, p.dtype) for p in model.parameters()] == flags
                model(x).sum().backward()
            assert set(model[0]._forward_hooks) == original_hooks
            assert not model[1]._forward_hooks and not model._forward_pre_hooks
        assert len(seen) == 5 and not model.training
    finally:
        user_hook.remove()


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("corruption", ["paths", "types", "schema", "missing"])
def test_checkpoint_placement_is_checked_before_parameter_copy(nested, corruption):
    _, stack = simple()
    owner = torch.nn.ModuleDict({"geometry": stack}) if nested else stack
    prefix = "geometry." if nested else ""
    before = copy.deepcopy(owner.state_dict())
    state = copy.deepcopy(before)
    state[prefix + "adapters.0.gate"].fill_(5.)
    if corruption == "missing": del state[prefix + "_extra_state"]
    elif corruption == "paths": state[prefix + "_extra_state"]["paths"].reverse()
    elif corruption == "types": state[prefix + "_extra_state"]["adapter_types"].reverse()
    else: state[prefix + "_extra_state"]["schema"] = "unknown"
    with pytest.raises(ValueError, match="placements or types"):
        owner.load_state_dict(state, strict=False)
    assert equal(before, owner.state_dict())


def test_existing_or_reattached_scope_cannot_accept_stale_backward():
    model, stack = simple()
    with stack.attach(model):
        output = model(torch.ones(2, 4))
    with pytest.raises(RuntimeError, match="finish backward"):
        output.sum().backward()
    with stack.attach(model):
        stale = model(torch.ones(2, 4))
    with stack.attach(model):
        with pytest.raises(RuntimeError, match="finish backward"):
            stale.sum().backward()
        model(torch.ones(2, 4)).sum().backward()


def test_conflicting_ownership_reentrancy_and_loading_are_rejected():
    model, stack = simple()
    state = copy.deepcopy(stack.state_dict())
    other = st.GeometryAdapterStack({"0": st.ToposResonatorAdapter(4)})
    with stack.attach(model):
        with pytest.raises(RuntimeError, match="already attached"):
            with stack.attach(model): pass
        with pytest.raises(RuntimeError, match="active geometry stack"):
            with other.attach(model): pass
        with pytest.raises(RuntimeError, match="detach geometry"):
            stack.load_state_dict(state)
    with other.attach(model): model(torch.ones(2, 4)).sum().backward()
    duplicate = st.ToposResonatorAdapter(4)
    with pytest.raises(ValueError, match="share"):
        st.GeometryAdapterStack({"0": duplicate, "1": duplicate})
    a, b = st.ToposResonatorAdapter(4), st.ToposResonatorAdapter(4)
    b.gate = a.gate
    with pytest.raises(ValueError, match="share"):
        st.GeometryAdapterStack({"0": a, "1": b})
    stolen = st.GeometryAdapterStack({"0": model[1]})
    with pytest.raises(ValueError, match="owned separately"):
        with stolen.attach(model): pass


@pytest.mark.parametrize("failure", ["forward", "registration", "path", "alias"])
def test_every_failure_removes_only_its_own_hooks(failure, monkeypatch):
    model, stack = simple()
    user = model[0].register_forward_hook(lambda *args: None)
    expected = set(model[0]._forward_hooks)
    if failure == "registration":
        def fail(*args, **kwargs): raise RuntimeError("registration failure")
        monkeypatch.setattr(model[1], "register_forward_hook", fail)
    elif failure == "path":
        stack = st.GeometryAdapterStack({"0": st.ToposResonatorAdapter(4), "absent": st.WaveGateAdapter(4)})
    elif failure == "alias":
        model.add_module("alias", model[0])
    try:
        with pytest.raises((RuntimeError, AttributeError, ValueError)):
            with stack.attach(model):
                raise RuntimeError("forward failure")
        assert set(model[0]._forward_hooks) == expected and not model[1]._forward_hooks
        assert not model._forward_pre_hooks and stack._active_token is None
    finally:
        user.remove()


def test_keyword_module_interfaces_and_cache_guards():
    class Keyword(torch.nn.Module):
        def forward(self, *, hidden): return hidden * 2
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Keyword()
            self.config = SimpleNamespace(use_cache=False)
        def forward(self, x, use_cache=None, past_key_values=None):
            return self.block(hidden=x)
    model = Model()
    stack = st.GeometryAdapterStack({"block": st.ToposResonatorAdapter(4)})
    x = torch.ones(2, 4)
    with stack.attach(model):
        assert torch.equal(model(x), x * 2)
        model(x, use_cache=False).sum().backward()
        for args, kwargs in (((x,), {"use_cache": True}), ((x, True), {}),
                             ((x,), {"past_key_values": ()}), ((x, False, ()), {})):
            with pytest.raises(ValueError, match="cached"):
                model(*args, **kwargs)
    for attribute in ("cache", "checkpointing"):
        model.config.use_cache = attribute == "cache"
        model.is_gradient_checkpointing = attribute == "checkpointing"
        with pytest.raises(ValueError):
            with stack.attach(model): pass


def test_forward_cache_defaults_and_mutated_adapter_inventory_are_rejected():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Identity()
        def forward(self, x, use_cache=True):
            return self.block(x)
    model = Model()
    stack = st.GeometryAdapterStack({"block": st.ToposResonatorAdapter(4)})
    with stack.attach(model):
        with pytest.raises(ValueError, match="cached"): model(torch.ones(2, 4))
        model(torch.ones(2, 4), use_cache=False).sum().backward()
        stack.adapters.append(st.ToposResonatorAdapter(4))
        with pytest.raises(RuntimeError, match="changed"): model(torch.ones(2, 4), use_cache=False)


def test_inference_without_gradients_is_supported():
    model, stack = simple()
    with torch.no_grad(), stack.attach(model):
        output = model(torch.ones(2, 4))
    assert not output.requires_grad


def test_data_parallel_wrapper_does_not_silently_leave_adapters_unsynchronized():
    model, _ = simple()
    stack = st.GeometryAdapterStack({"module.0": st.ToposResonatorAdapter(4)})
    with pytest.raises(ValueError, match="parallel wrappers"):
        with stack.attach(torch.nn.DataParallel(model)): pass


@pytest.mark.parametrize("kind", ["tuple", "dtype", "shape", "width"])
def test_incompatible_output_is_not_silently_coerced(kind):
    class Output(torch.nn.Module):
        def forward(self, x): return (x,) if kind == "tuple" else x
    class WrongAdapter(torch.nn.Module):
        def forward(self, x): return x.double() if kind == "dtype" else x[..., :-1]
    model = torch.nn.Sequential(Output())
    adapter = (WrongAdapter() if kind in ("dtype", "shape") else
               st.ToposResonatorAdapter(5 if kind == "width" else 4))
    stack = st.GeometryAdapterStack({"0": adapter})
    with pytest.raises((TypeError, ValueError)):
        with stack.attach(model): model(torch.ones(2, 4))
    assert not model[0]._forward_hooks


@pytest.mark.parametrize("placements", [{}, {"": None}, {"0..x": None}, {"a": None},
                                        {"a": torch.nn.Identity(), "a.b": torch.nn.Identity()}])
def test_invalid_placement_is_rejected(placements):
    with pytest.raises((TypeError, ValueError)): st.GeometryAdapterStack(placements)


def test_offline_example_executes_real_public_api_and_detects_drift(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "tools"))
    spec = importlib.util.spec_from_file_location("geometry_stack_example", root / "bindings/st-py/examples/hf_geometry_adapter_stack.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    config = json.loads((root / "bindings/st-py/examples/hf_geometry_adapter_stack.json").read_text())
    config.update(features=8, block_size=6, learning_rate=.01)
    model, _ = tiny_hf("gpt2")
    tokens = torch.arange(48).reshape(8, 6) % 32
    reference = example.run(model, tokens, config, False)
    public = example.run(model, tokens, config, True)
    resumed = example.run(model, tokens, config, True, saved=reference["midpoint"])
    example.compare(reference, public, resumed, 2)
    public["gradients"][0]["adapters.0.gate"].view(torch.int32)[0] ^= 1
    with pytest.raises(ValueError, match="raw gradients differ"):
        example.compare(reference, public, resumed, 2)
    config["placements"][1]["path"] = config["placements"][0]["path"]
    with pytest.raises(ValueError, match="recipe"): example.make_stack(config)

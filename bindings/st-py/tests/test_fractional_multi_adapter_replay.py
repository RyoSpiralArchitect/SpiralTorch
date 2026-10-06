import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("multi_adapter_probe", ROOT / "tools/probe_fractional_multi_adapter.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def recipe():
    result = json.loads((ROOT / "bindings/st-py/examples/hf_fractional_dual_replay.json").read_text())
    result.update(features=8, block_size=6, learning_rate=.01)
    return result


def setup():
    torch.set_num_threads(2)
    torch.manual_seed(197)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=32, n_positions=8, n_embd=8, n_layer=2, n_head=2,
        resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
    )).eval().requires_grad_(False)
    return model, torch.arange(48).reshape(8, 6) % 32


def run(probe, window="full", **kwargs):
    return probe.learning_run(*setup(), recipe(), window,
                              {"native_sha256": "native-test"}, {"fixture": "tiny-gpt2"}, **kwargs)


@pytest.mark.parametrize("window", ["short", "full"])
def test_two_adapters_execute_joint_vjp_and_exact_checkpoint_continuation(probe, tmp_path, window):
    before = run(probe, window)
    path = tmp_path / "midpoint.pt"
    receipt = probe.save_state(path, before["midpoint"])
    assert receipt["sha256"] == probe.common.digest(path)
    saved = torch.load(path, weights_only=True, map_location="cpu")
    after = run(probe, window)
    resumed = run(probe, window, saved=saved, allowed_native="native-test")
    report = probe.compare_runs(before, after, resumed)
    assert report["status"] == "bitwise_exact" and report["updates_executed"] == 10
    for row in report["records"]:
        assert [c["method"] for c in row["native_calls"]] == ["vjp_buffer", "vjp_parameters_buffer"]
        if row["step"] > 1:
            assert row["native_calls"][0]["input_vjp"]["nonzero"] > 0
            assert all(g["nonzero"] for g in row["gradients"].values())
        else:
            for site in ("site0", "site1"):
                assert row["gradients"][site + ".history_angle"]["nonzero"] == 0
                assert row["gradients"][site + ".log_gain"]["nonzero"] == 0
    assert before["final"]["parameter_names"] == [
        f"site{i}.{p}" for i in range(2) for p in ("gate", "local_gate", "history_angle", "log_gain")]


def test_observation_leaves_public_adapter_math_unchanged(probe):
    observed, plain = run(probe), run(probe, observe=False)
    for name in ("adapter", "optimizer", "rng"):
        assert probe.equal(observed["final"][name], plain["final"][name])
    assert probe.equal(observed["executed_gradients"], plain["executed_gradients"])
    for left, right in zip(observed["final"]["records"], plain["final"]["records"]):
        assert left["loss"] == right["loss"] and right["native_calls"] == []


def test_zeroed_native_input_vjp_is_rejected_and_modules_restored(probe, monkeypatch):
    delegate = probe.ObservedSnapshot.__getattr__
    def corrupt(self, name):
        operation = delegate(self, name)
        def invoke(direction):
            result = operation(direction)
            if name == "vjp_buffer":
                zeros = bytearray(len(result[0]))
                self.events[-1]["input_vjp"] = probe.tensor_receipt(torch.frombuffer(zeros, dtype=torch.float32))
                return zeros, *result[1:]
            return result
        return invoke
    monkeypatch.setattr(probe.ObservedSnapshot, "__getattr__", corrupt)
    model, tokens = setup()
    originals = [model.get_submodule(p) for p in recipe()["blocks"]]
    with pytest.raises(ValueError, match="input VJP never became active"):
        probe.learning_run(model, tokens, recipe(), "full", {"native_sha256": "native-test"}, {})
    assert [model.get_submodule(p) for p in recipe()["blocks"]] == originals
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("corruption", ["binding", "cursor", "names", "order", "recipe", "moment", "runtime", "nan", "step"])
def test_incompatible_saved_states_cannot_resume(probe, corruption):
    saved = run(probe)["midpoint"]
    if corruption == "binding": saved["binding"]["window"] = "short"
    elif corruption == "cursor": saved["cursor"] = 1
    elif corruption == "names": saved["parameter_names"].reverse()
    elif corruption == "order": saved["optimizer"]["param_groups"][0]["params"].reverse()
    elif corruption == "recipe": saved["adapter"]["site1._extra_state"]["kernel"]["kernel_len"] = 3
    elif corruption == "moment": saved["optimizer"]["state"][0]["exp_avg_sq"][0] = -1.
    elif corruption == "runtime": saved["runtime"]["native_sha256"] = "unknown-native"
    elif corruption == "nan": saved["adapter"]["site1.history_angle"].fill_(float("nan"))
    elif corruption == "step": saved["optimizer"]["state"][0]["step"].fill_(1.)
    with pytest.raises(ValueError): run(probe, saved=saved, allowed_native="native-test")


@pytest.mark.parametrize("corruption", ["gradient", "loss", "state", "moment", "calls", "count", "target"])
def test_comparison_rejects_single_bit_and_missing_evidence(probe, corruption):
    before = run(probe)
    after = copy.deepcopy(before)
    resumed = run(probe, saved=before["midpoint"], allowed_native="native-test")
    if corruption == "gradient": after["executed_gradients"][0]["site0.gate"].view(torch.int32)[0] ^= 1
    elif corruption == "loss": after["final"]["records"][0]["loss_value"] += 1e-6
    elif corruption == "state": after["final"]["adapter"]["site0.gate"].view(torch.int32)[0] ^= 1
    elif corruption == "moment": after["final"]["optimizer"]["state"][0]["exp_avg"].view(torch.int32)[0] ^= 1
    elif corruption == "calls": after["final"]["records"][0]["native_calls"].clear()
    elif corruption == "count": after["executed_gradients"].pop()
    elif corruption == "target": resumed["final"]["runtime"]["native_sha256"] = "unexpected"
    with pytest.raises(ValueError): probe.compare_runs(before, after, resumed)


@pytest.mark.parametrize("paths", [["transformer.h.0.mlp", "not_a_module"],
    ["transformer.h.0.mlp", "transformer.h.0.mlp"],
    ["transformer.h.0", "transformer.h.0.mlp"]])
def test_bad_insertion_paths_do_not_leave_a_partial_patch(probe, paths):
    model, train = setup()
    original = model.transformer.h[0].mlp
    config = recipe()
    config["blocks"] = paths
    with pytest.raises((ValueError, AttributeError)):
        probe.learning_run(model, train, config, "full", {"native_sha256": "test"}, {})
    assert model.transformer.h[0].mlp is original


def test_signed_zero_is_not_a_bitwise_match(probe):
    assert not probe.equal(torch.tensor(0.), torch.tensor(-0.))
    assert probe.tensor_receipt(torch.tensor(0.))["sha256"] != probe.tensor_receipt(torch.tensor(-0.))["sha256"]

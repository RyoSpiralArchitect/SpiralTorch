import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=["gain", "angle"])
def completed_study(tmp_path_factory, request):
    family = request.param
    directory = tmp_path_factory.mktemp("gain-verification")
    with pytest.MonkeyPatch.context() as patch:
        examples = ROOT / "bindings/st-py/examples"
        patch.syspath_prepend(str(examples))
        patch.syspath_prepend(str(ROOT / "tools"))
        client = load("gain_verify_client", examples / f"hf_fractional_{family}_study.py")
        summary = load("gain_verify_summary", ROOT / "tools/summarize_wave_gate_long_horizon.py")
        verifier = load("gain_verify", ROOT / "tools/verify_fractional_gain_study.py")
        config = json.loads((examples / f"hf_fractional_pride_{family}.json").read_text())
        config.update(features=8, steps=2, block_size=6, batch_size=2, seeds=[41],
                      checkpoint_every=2, evaluate_every=2, learning_rate=.01)
        torch.manual_seed(197)
        torch.set_num_threads(2)
        model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
            vocab_size=32, n_positions=8, n_embd=8, n_head=2, n_layer=1,
            resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
        )).eval().requires_grad_(False)
        tokens = torch.arange(48).reshape(8, 6) % 32
        driver = client.study
        binding = {"config": config, "base_parameter_sha256": driver.pilot.model_digest(model),
                   "torch": str(torch.__version__), "transformers": transformers.__version__,
                   "data": {"evaluation_block_hashes": {"probe": driver.block_hashes(tokens[:2])}}}
        plan = {**binding, "study_id": driver.identity(binding), "source_revision": "b" * 40,
                "batch_schedules": {"41": driver.pilot.schedule(41, 8, 3, 2)}}
        journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
        study = directory / "study"
        study.mkdir()
        driver.atomic_json(study / "plan.json", plan)
        parent, original = model.transformer.h[0], model.transformer.h[0].mlp
        driver.run_training(model, parent, "mlp", original, tokens, tokens[:2], plan, study,
                            journal, adapter_factory=client.adapter_for)
        driver.run_endpoints(model, parent, "mlp", original, {"probe": tokens[:2]}, plan,
                             study, journal, adapter_factory=client.adapter_for,
                             result_schema=f"spiraltorch.fractional_{family}_study.v1")
        current = verifier.common.completed(study, client)
        plan, journal, result, _ = current
        report = summary.summarize(plan, result, journal, journal["results_sha256"])
        report["input_sha256"] = {key: verifier.digest(study / f"{key}.json")
                                 for key in ("plan", "results", "journal")}
        (directory / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        yield directory, client, summary, verifier, current


def test_saved_tensors_named_adam_and_summary_are_verified_without_scoring(completed_study, monkeypatch):
    directory, client, summary, verifier, _ = completed_study
    def forbidden(*args, **kwargs):
        raise AssertionError("verification must not train or score")
    monkeypatch.setattr(client.study, "run_training", forbidden)
    monkeypatch.setattr(client.study, "run_endpoints", forbidden)
    monkeypatch.setattr(client.study.pilot, "update", forbidden)
    before = {p.name: verifier.digest(p) for p in (directory / "study").iterdir()}
    report = verifier.verify(directory / "study", directory / "summary.json", client, summary)
    assert report["status"] == "passed" and report["summary_rebuilt_byte_identical"]
    assert report["primary_updates"] == report["continuation_only_updates"] == 6
    assert report["planned_runs_verified"] == 3
    if report["schema"] == "spiraltorch.fractional_angle_checkpoint_verification.v1":
        assert report["paired_short_state_status"] == "passed"
        assert set(report["paired_short_states"]) == {"41"}
        for row in report["paired_short_states"]["41"].values():
            assert row["allclose"] is True and row["rtol"] == 3e-6 and row["atol"] == 3e-7
    for key, row in report["runs"].items():
        shape = "history_angle" if "ordinary" in key or "angle" in key else "log_alpha"
        assert set(row["parameters"]) == set(row["named_adam"]) == {"gate", "local_gate", "log_gain", shape}
        assert row["parameters"]["log_gain"]["shape"] == []
        assert row["parameters"]["gate"]["shape"] == [8]
        assert row["final_coordinates"]["gain"] > 0
        assert row["final_coordinates"]["effective_history_gate_l2"] > 0
    assert before == {p.name: verifier.digest(p) for p in (directory / "study").iterdir()}


@pytest.mark.parametrize("arm_index", [0, 1, 2])
@pytest.mark.parametrize("corruption", ["gain", "shape_coordinate", "gate_scale", "recipe", "adam_mapping"])
def test_each_arm_is_bound_to_its_saved_coordinates_and_recipe(completed_study, arm_index, corruption):
    directory, client, _, verifier, current = completed_study
    plan, journal, _, rows = current
    arm = client.ARMS[arm_index]
    key = f"41:{arm}"
    row, entry = copy.deepcopy(rows[key]), copy.deepcopy(journal["runs"][key])
    saved = client.study.load_checkpoint(directory / "study", entry["checkpoint"], plan["study_id"], key)
    if corruption == "gain":
        saved["adapter"]["log_gain"] += .01
    elif corruption == "shape_coordinate":
        saved["adapter"]["history_angle" if arm.startswith("ordinary") or "angle" in arm else "log_alpha"] += .01
    elif corruption == "gate_scale":
        saved["adapter"]["gate"] += .01
    elif corruption == "recipe":
        saved["adapter"]["_extra_state"]["arm"] = "other"
    else:
        ids = saved["optimizer"]["param_groups"][0]["params"]
        ids[0], ids[1] = ids[1], ids[0]
    with pytest.raises(ValueError):
        verifier.inspect_run(client, plan, row, entry, saved)


@pytest.mark.parametrize("corruption", ["dtype", "shape", "nan", "moment_shape", "moment_nan",
    "negative_second_moment", "adam_recipe", "adam_step", "step_dtype", "cursor", "records", "resume"])
def test_invalid_saved_state_is_rejected(completed_study, corruption):
    directory, client, _, verifier, current = completed_study
    plan, journal, _, rows = current
    key = f"41:{client.ARMS[-1]}"
    row, entry = copy.deepcopy(rows[key]), copy.deepcopy(journal["runs"][key])
    saved = client.study.load_checkpoint(directory / "study", entry["checkpoint"], plan["study_id"], key)
    state = saved["optimizer"]["state"][0]
    if corruption == "dtype": saved["adapter"]["log_gain"] = saved["adapter"]["log_gain"].double()
    elif corruption == "shape": saved["adapter"]["log_gain"] = saved["adapter"]["log_gain"].reshape(1)
    elif corruption == "nan": saved["adapter"]["log_gain"].fill_(float("nan"))
    elif corruption == "moment_shape": state["exp_avg"] = torch.zeros(1)
    elif corruption == "moment_nan": state["exp_avg"][0] = float("nan")
    elif corruption == "negative_second_moment": state["exp_avg_sq"][0] = -1.
    elif corruption == "adam_recipe": saved["optimizer"]["param_groups"][0]["lr"] *= 2
    elif corruption == "adam_step": state["step"] += 1
    elif corruption == "step_dtype": state["step"] = state["step"].double()
    elif corruption == "cursor": saved["cursor"] -= 1
    elif corruption == "records": row["records"][0]["loss"] += .1
    else: entry["resume_next_update_equal"] = False
    with pytest.raises(ValueError):
        verifier.inspect_run(client, plan, row, entry, saved)


def test_summary_is_reconstructed_not_trusted(completed_study, tmp_path):
    directory, client, summary, verifier, _ = completed_study
    changed = json.loads((directory / "summary.json").read_bytes())
    changed["primary_updates"] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed, indent=2) + "\n")
    with pytest.raises(ValueError, match="summary artifact"):
        verifier.verify(directory / "study", path, client, summary)


def test_incomplete_study_cannot_be_verified(completed_study, tmp_path):
    _, client, _, verifier, current = completed_study
    plan, journal = copy.deepcopy(current[:2])
    journal["status"] = "training"
    client.study.atomic_json(tmp_path / "plan.json", plan)
    client.study.atomic_json(tmp_path / "journal.json", journal)
    with pytest.raises(ValueError, match="not completed"):
        verifier.common.completed(tmp_path, client)


def test_changed_checkpoint_bytes_are_rejected_before_loading(completed_study, tmp_path):
    directory, client, _, verifier, current = completed_study
    plan, journal, _, _ = current
    key = f"41:{client.ARMS[-1]}"
    receipt = journal["runs"][key]["checkpoint"]
    data = (directory / "study" / receipt["filename"]).read_bytes()
    (tmp_path / receipt["filename"]).write_bytes(data + b"changed")
    with pytest.raises(ValueError, match="checkpoint hash"):
        client.study.load_checkpoint(tmp_path, receipt, plan["study_id"], key)


def test_tensor_receipts_are_bitwise_including_signed_zero(completed_study):
    verifier = completed_study[3]
    assert verifier.tensor_receipt(torch.tensor(0.))["sha256"] != verifier.tensor_receipt(torch.tensor(-0.))["sha256"]


def test_paired_state_comparison_retains_tolerance_failure_and_signed_zero(completed_study):
    verifier = completed_study[3]
    zero = verifier.compare_tensor_maps({"x": torch.tensor(0.)}, {"x": torch.tensor(-0.)})
    assert zero["allclose"] and zero["value_equal"] and not zero["byte_equal"]
    near = verifier.compare_tensor_maps({"x": torch.tensor(1.)}, {"x": torch.tensor(1.000001)})
    assert near["allclose"] and not near["value_equal"]
    far = verifier.compare_tensor_maps({"x": torch.tensor(1.)}, {"x": torch.tensor(1.01)})
    assert far["status"] == "failed" and not far["allclose"]
    with pytest.raises(ValueError):
        verifier.compare_tensor_maps({"x": torch.tensor(1.)}, {"x": torch.tensor(float("nan"))})


@pytest.fixture
def bound_environment(completed_study, tmp_path, monkeypatch):
    import spiraltorch as st
    import spiraltorch.spiraltorch as native
    import spiraltorch.geometry_autograd as geometry

    verifier = completed_study[3]
    angular = completed_study[-1][0]["config"]["schema"] == "spiraltorch.fractional_angle_protocol.v1"
    client_root, package_root = tmp_path / "client", tmp_path / "runtime"
    client_root.mkdir()
    (package_root / "spiraltorch").mkdir(parents=True)
    def module(root, name):
        path = root / name
        path.write_text(f"# fixture {name}\n")
        return SimpleNamespace(__file__=str(path))
    package = package_root / "spiraltorch"
    for target, name in ((st, "__init__.py"), (native, "spiraltorch.so"), (geometry, "geometry_autograd.py")):
        monkeypatch.setattr(target, "__file__", module(package, name).__file__)
    client = module(client_root, "hf_fractional_angle_study.py" if angular else "hf_fractional_gain_study.py")
    if angular:
        client.gain = module(client_root, "hf_fractional_gain_study.py")
    control = client.gain if angular else client
    control.lag = module(client_root, "hf_fractional_lag_study.py")
    client.study = module(client_root, "hf_wave_gate_long_horizon.py")
    client.study.pilot = module(client_root, "hf_wave_gate_conditioning.py")
    client.study.pilot.transformers = transformers
    control.fractional_bridge = module(package, "fractional_autograd.py")
    summary = module(client_root, "summarize_wave_gate_long_horizon.py")
    config = {"recipe": "fixed fixture", "schema": "spiraltorch.fractional_angle_protocol.v1" if angular else "fixture"}
    config_path = client_root / ("hf_fractional_pride_angle.json" if angular else "hf_fractional_pride_gain.json")
    config_path.write_text(json.dumps(config))
    plan = {"config": config, "torch": str(torch.__version__), "transformers": transformers.__version__,
            "source_revision": "b" * 40,
            "adapter_sources_sha256": {"lag_control": verifier.digest(Path(control.lag.__file__)),
                                      "fractional_bridge": verifier.digest(Path(control.fractional_bridge.__file__))}}
    sources = {"angle_study": client, "gain_control": control} if angular else {"gain_study": client}
    plan["adapter_sources_sha256"].update({key: verifier.digest(Path(module.__file__)) for key, module in sources.items()})
    for key, path in (("native_sha256", native.__file__), ("bridge_sha256", geometry.__file__),
                      ("script_sha256", client.study.__file__), ("helper_sha256", client.study.pilot.__file__),
                      ("config_sha256", config_path)):
        plan[key] = verifier.digest(Path(path))
    manifests = []
    for name, root in (("client", client_root), ("runtime", package_root)):
        files = {str(p.relative_to(root)): verifier.digest(p) for p in root.rglob("*") if p.is_file()}
        value = files if name == "client" else {"source_revision": "a" * 40, "files": files}
        path = tmp_path / f"{name}-manifest.json"
        path.write_text(json.dumps(value))
        manifests.append(path)
    return verifier, plan, client, summary, manifests, client_root, package_root


def test_build_and_launch_revisions_are_separate_but_executable_bytes_are_bound(bound_environment):
    verifier, plan, client, summary, manifests, client_root, _ = bound_environment
    cache = client_root / "__pycache__"
    cache.mkdir()
    (cache / "summary.pyc").write_bytes(b"generated")
    report = verifier.verify_environment(plan, client, summary, *manifests)
    angular = plan["config"]["schema"] == "spiraltorch.fractional_angle_protocol.v1"
    assert report["frozen_files_verified"] == {"client": 7 if angular else 6, "runtime": 4}
    assert report["runtime_build_source_revision"] != plan["source_revision"]


@pytest.mark.parametrize("corruption", ["native", "bridge", "helper", "adapter_source", "config",
                                       "framework", "runtime_manifest", "extra", "foreign_helper"])
def test_environment_drift_is_rejected(bound_environment, corruption, tmp_path):
    verifier, plan, client, summary, manifests, client_root, _ = bound_environment
    if corruption in {"native", "bridge", "helper"}:
        plan[f"{corruption}_sha256"] = "0" * 64
    elif corruption == "adapter_source":
        plan["adapter_sources_sha256"]["gain_study"] = "0" * 64
    elif corruption == "config": plan["config"] = {"recipe": "changed"}
    elif corruption == "framework": plan["torch"] = "other"
    elif corruption == "runtime_manifest": manifests[1].write_text(json.dumps({"files": {}}))
    elif corruption == "extra": (client_root / "unexpected.py").write_text("# unexpected\n")
    else:
        source = Path(client.study.pilot.__file__)
        other = tmp_path / source.name
        other.write_bytes(source.read_bytes())
        client.study.pilot.__file__ = str(other)
    with pytest.raises(ValueError):
        verifier.verify_environment(plan, client, summary, *manifests)

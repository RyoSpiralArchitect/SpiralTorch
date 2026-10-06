import copy
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[3]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def completed_pair(tmp_path_factory):
    directory = tmp_path_factory.mktemp("history-factorial-verification")
    with pytest.MonkeyPatch.context() as patch:
        examples = ROOT / "bindings/st-py/examples"
        patch.syspath_prepend(str(examples))
        client = load("factorial_verify_client", examples / "hf_fractional_history_factorial.py")
        old = load("factorial_verify_old", examples / "hf_fractional_two_lag_study.py")
        summary = load("factorial_verify_summary", ROOT / "tools/summarize_wave_gate_long_horizon.py")
        verifier = load("factorial_verify", ROOT / "tools/verify_fractional_history_factorial.py")
        config = json.loads((examples / "hf_fractional_pride_history_factorial.json").read_text())
        config.update(features=8, steps=2, block_size=6, batch_size=2, seeds=[41],
                      checkpoint_every=2, evaluate_every=2, learning_rate=.01)
        tokens = torch.arange(48).reshape(8, 6) % 32
        torch.set_num_threads(2)
        for label, factory in (("current", client), ("previous", old)):
            path = directory / label
            path.mkdir()
            torch.manual_seed(139)
            model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
                vocab_size=32, n_positions=8, n_embd=8, n_head=2, n_layer=1,
                resid_pdrop=0, attn_pdrop=0, embd_pdrop=0, use_cache=False,
            )).eval().requires_grad_(False)
            recipe = copy.deepcopy(config)
            if label == "previous":
                recipe.update(schema="spiraltorch.fractional_two_lag_protocol.v1", arms=old.ARMS,
                              initial_orders=old.INITIAL_ORDERS)
            driver = factory.study
            binding = {"config": recipe, "base_parameter_sha256": driver.pilot.model_digest(model),
                       "model_config_sha256": "a" * 64, "torch": str(torch.__version__),
                       "transformers": transformers.__version__,
                       "data": {"evaluation_block_hashes": {"tail": driver.block_hashes(tokens[:2])}}}
            plan = {**binding, "study_id": driver.identity(binding), "source_revision": "b" * 40,
                    "batch_schedules": {"41": driver.pilot.schedule(41, 8, 3, 2)}}
            journal = {"study_id": plan["study_id"], "status": "training", "runs": {}}
            driver.atomic_json(path / "plan.json", plan)
            parent, original = model.transformer.h[0], model.transformer.h[0].mlp
            driver.run_training(model, parent, "mlp", original, tokens, tokens[:2], plan, path,
                                journal, adapter_factory=factory.adapter_for)
            driver.run_endpoints(model, parent, "mlp", original, {"tail": tokens[:2]}, plan,
                                 path, journal, adapter_factory=factory.adapter_for,
                                 result_schema=("spiraltorch.fractional_history_factorial_study.v1"
                                                if label == "current" else "spiraltorch.fractional_two_lag_study.v1"))
        current = verifier.completed(directory / "current", client)
        plan, journal, result, _ = current
        report = summary.summarize(plan, result, journal, journal["results_sha256"])
        report["input_sha256"] = {key: verifier.digest(directory / "current" / f"{key}.json")
                                 for key in ("plan", "results", "journal")}
        summary_path = directory / "summary.json"
        summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        yield directory, client, summary, verifier, current


def test_real_saved_state_and_exact_historical_replay(completed_pair):
    directory, client, summary, verifier, _ = completed_pair
    report = verifier.verify(directory / "current", directory / "previous", directory / "summary.json",
                             client, summary)
    assert report["status"] == "passed" and report["summary_rebuilt_byte_identical"]
    assert report["primary_updates"] == report["continuation_only_updates"] == 8
    assert report["planned_runs_verified"] == 4
    assert report["historical_raw_full"]["status"] == "exact_replay"
    assert report["historical_raw_full"]["counts_as_additional_independent_seeds"] is False
    assert report["summary_artifact_sha256"] != report["summary_source_sha256"]


@pytest.mark.parametrize("corruption", ["dtype", "shape", "nan", "recipe", "moment_shape", "moment_nan",
                                       "negative_second_moment", "adam_recipe", "adam_step", "adam_mapping",
                                       "cursor", "order", "records", "resume"])
def test_damaged_checkpoint_rejected(completed_pair, corruption):
    directory, client, _, verifier, current = completed_pair
    plan, journal, _, rows = current
    key = "41:history_l2_full"
    row, entry = copy.deepcopy(rows[key]), copy.deepcopy(journal["runs"][key])
    saved = client.study.load_checkpoint(directory / "current", entry["checkpoint"], plan["study_id"], key)
    if corruption == "dtype":
        saved["adapter"]["gate"] = saved["adapter"]["gate"].double()
    elif corruption == "shape":
        saved["adapter"]["gate"] = saved["adapter"]["gate"][:1]
    elif corruption == "nan":
        saved["adapter"]["gate"][0] = float("nan")
    elif corruption == "recipe":
        saved["adapter"]["_extra_state"]["gain"] = 1.
    elif corruption == "moment_shape":
        saved["optimizer"]["state"][0]["exp_avg"] = torch.zeros(1)
    elif corruption == "moment_nan":
        saved["optimizer"]["state"][0]["exp_avg"][0] = float("nan")
    elif corruption == "negative_second_moment":
        saved["optimizer"]["state"][0]["exp_avg_sq"][0] = -1.
    elif corruption == "adam_recipe":
        saved["optimizer"]["param_groups"][0]["lr"] *= 2
    elif corruption == "adam_step":
        saved["optimizer"]["state"][0]["step"] += 1
    elif corruption == "adam_mapping":
        saved["optimizer"]["param_groups"][0]["params"] = [0, 0, 0]
    elif corruption == "cursor":
        saved["cursor"] -= 1
    elif corruption == "order":
        saved["adapter"]["log_alpha"] += .01
    elif corruption == "records":
        row["records"][0]["loss"] += .1
    else:
        entry["resume_next_update_equal"] = False
    with pytest.raises(ValueError):
        verifier.inspect_run(client, plan, row, entry, saved)


def test_changed_summary_cannot_pass_with_matching_claimed_hash(completed_pair, tmp_path):
    directory, client, summary, verifier, _ = completed_pair
    changed = json.loads((directory / "summary.json").read_bytes())
    changed["primary_updates"] += 1
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed, indent=2) + "\n")
    with pytest.raises(ValueError, match="summary artifact"):
        verifier.verify(directory / "current", directory / "previous", path, client, summary)


def test_valid_but_different_replay_is_retained_as_mismatch(completed_pair):
    directory, client, _, verifier, current = completed_pair
    plan, journal, _, rows = current
    adapters, moments = {}, {}
    for key, row in rows.items():
        entry = journal["runs"][key]
        saved = client.study.load_checkpoint(directory / "current", entry["checkpoint"], plan["study_id"], key)
        adapters[key], moments[key] = verifier.inspect_run(client, plan, row, entry, saved)
    previous = verifier.completed(directory / "previous", client)
    # Differing valid endpoint values are an outcome, not grounds to hide the run.
    score = previous[3]["41:history_learned_two"]["scores"]["tail"]
    score["block_losses"][0] += .01
    score["mean"] = sum(score["block_losses"]) / len(score["block_losses"])
    report = verifier.replay(current, (*previous, directory / "previous"), client, adapters, moments)
    assert report["status"] == "mismatch"
    assert report["runs"]["41"]["parity"]["endpoints_equal"] is False
    assert report["runs"]["41"]["parity"]["named_adam_equal"] is True
    assert report["counts_as_additional_independent_seeds"] is False


def test_file_manifest_rejects_changes_extras_and_symlinks(completed_pair, tmp_path):
    verifier = completed_pair[3]
    path = tmp_path / "client.py"
    path.write_text("# frozen\n")
    manifest = {path.name: verifier.digest(path)}
    assert verifier.verify_files(tmp_path, manifest) == 1
    path.write_text("# changed\n")
    with pytest.raises(ValueError, match="frozen file differs"):
        verifier.verify_files(tmp_path, manifest)
    manifest[path.name] = verifier.digest(path)
    extra = tmp_path / "extra.py"
    extra.write_text("# extra\n")
    with pytest.raises(ValueError, match="inventory"):
        verifier.verify_files(tmp_path, manifest)
    extra.unlink()
    link = tmp_path / "linked.py"
    link.symlink_to(path)
    manifest[link.name] = verifier.digest(path)
    with pytest.raises(ValueError, match="file path"):
        verifier.verify_files(tmp_path, manifest)


def test_signed_zero_comparison_is_bitwise(completed_pair):
    equal = completed_pair[3].equal
    assert equal(torch.tensor([0.]), torch.tensor([0.]))
    assert not equal(torch.tensor([0.]), torch.tensor([-0.]))


def test_native_impulse_description_not_python_coefficient_reconstruction(completed_pair):
    _, client, _, verifier, current = completed_pair
    config = current[0]["config"]
    for arm in client.ARMS:
        adapter = client.adapter_for(arm, config, 41)
        initial = verifier.filter_description(adapter)
        assert initial["first_two_past_coefficients"] == [-2., 1.]
        assert initial["lag_3_plus_energy_fraction"] == 0
        assert initial["coefficient_l2"] == pytest.approx(5**.5)
        with torch.no_grad():
            adapter.log_alpha.fill_(torch.tensor(4.).log())
        final = verifier.filter_description(adapter)
        assert final["backend"] == "rust_f32_cpu"
        if arm.startswith("history_l2_"):
            assert final["coefficient_l2"] == pytest.approx(5**.5, rel=2e-7)
        else:
            assert final["coefficient_l2"] == pytest.approx((52 if arm.endswith("_short") else 69)**.5)
        assert final["lag_3_plus_energy_fraction"] == pytest.approx(0 if arm.endswith("_short") else 17/69)

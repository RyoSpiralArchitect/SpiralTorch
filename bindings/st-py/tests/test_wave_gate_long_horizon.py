import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import spiraltorch as st

pytest.importorskip("fcntl")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytestmark = pytest.mark.skipif(
    not hasattr(st, "WaveGateKernel"), reason="native WaveGate kernel required"
)


@pytest.fixture
def driver(monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location(
        "radius_long_horizon_test", examples / "hf_wave_gate_long_horizon.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_model(vocab_size=16):
    torch.manual_seed(23)
    torch.set_num_threads(2)
    return (
        transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=vocab_size,
                n_positions=8,
                n_embd=8,
                n_layer=1,
                n_head=2,
                resid_pdrop=0,
                attn_pdrop=0,
                embd_pdrop=0,
                use_cache=False,
            )
        )
        .eval()
        .requires_grad_(False)
    )


def fixture(driver, directory):
    directory.mkdir()
    model = tiny_model()
    train = (torch.arange(48).reshape(8, 6) % 15) + 1
    config = {
        "features": 8,
        "strength": 0.2,
        "learning_rate": 0.01,
        "steps": 4,
        "batch_size": 2,
        "checkpoint_every": 2,
        "evaluate_every": 2,
        "arms": driver.ARMS,
        "seeds": [41],
    }
    plan = {
        "study_id": "fixture",
        "config": config,
        "base_parameter_sha256": driver.pilot.model_digest(model),
        "batch_schedules": {"41": driver.pilot.schedule(41, 8, 5, 2)},
    }
    journal = {"study_id": "fixture", "status": "training", "runs": {}}
    parent = model.transformer.h[0]
    return model, parent, parent.mlp, train, plan, journal


def test_interrupt_resume_matches_uninterrupted_and_keeps_endpoints_locked(
    driver, tmp_path
):
    full = tmp_path / "full"
    interrupted = tmp_path / "interrupted"
    model, parent, original, train, plan, journal = fixture(driver, full)
    driver.run_training(
        model, parent, "mlp", original, train, train[:2], plan, full, journal
    )
    other, other_parent, other_original, _, other_plan, other_journal = fixture(
        driver, interrupted
    )

    def stop(key, cursor):
        raise RuntimeError(f"intentional stop at {key} {cursor}")

    with pytest.raises(RuntimeError, match="intentional stop"):
        driver.run_training(
            other,
            other_parent,
            "mlp",
            other_original,
            train,
            train[:2],
            other_plan,
            interrupted,
            other_journal,
            after_checkpoint=stop,
        )
    assert other_parent.mlp is other_original
    assert other_journal["runs"]["41:tangent"]["cursor"] == 2
    with pytest.raises(ValueError, match="all planned runs"):
        driver.run_endpoints(
            other,
            other_parent,
            "mlp",
            other_original,
            {"sealed": train[:2]},
            other_plan,
            interrupted,
            other_journal,
        )
    assert not (interrupted / "results.json").exists()
    resumed = tiny_model()
    resumed_parent = resumed.transformer.h[0]
    resumed_original = resumed_parent.mlp
    restored_journal = json.loads((interrupted / "journal.json").read_text())
    driver.run_training(
        resumed,
        resumed_parent,
        "mlp",
        resumed_original,
        train,
        train[:2],
        other_plan,
        interrupted,
        restored_journal,
    )
    for key in journal["runs"]:
        left = driver.load_checkpoint(
            full, journal["runs"][key]["checkpoint"], "fixture", key
        )
        right = driver.load_checkpoint(
            interrupted, restored_journal["runs"][key]["checkpoint"], "fixture", key
        )
        assert driver.pilot.equal_state(left, right)
        assert restored_journal["runs"][key]["resume_next_update_equal"]
    result = driver.run_endpoints(
        resumed,
        resumed_parent,
        "mlp",
        resumed_original,
        {"sealed": train[:2]},
        other_plan,
        interrupted,
        restored_journal,
    )
    assert result["status"] == "completed" and len(result["runs"]) == 3
    assert driver.completed_result(interrupted, restored_journal, other_plan) == result
    (interrupted / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="result hash"):
        driver.completed_result(interrupted, restored_journal, other_plan)


def test_single_writer_hash_identity_and_path_guards(driver, tmp_path):
    with driver.study_lock(tmp_path):
        with pytest.raises(ValueError, match="another writer"):
            with driver.study_lock(tmp_path):
                pass
    payload = {"study_id": "one", "run_key": "41:tangent", "cursor": 2}
    receipt = driver.save_checkpoint(tmp_path, payload)
    assert driver.load_checkpoint(tmp_path, receipt, "one", "41:tangent") == payload
    with pytest.raises(ValueError, match="identity"):
        driver.load_checkpoint(tmp_path, receipt, "two", "41:tangent")
    corrupted = copy.deepcopy(receipt)
    corrupted["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        driver.load_checkpoint(tmp_path, corrupted, "one", "41:tangent")
    corrupted["filename"] = "../checkpoint-elsewhere.pt"
    with pytest.raises(ValueError, match="path"):
        driver.load_checkpoint(tmp_path, corrupted, "one", "41:tangent")
    assert driver.completed_result(tmp_path, {"status": "training"}, {}) is None


def test_mutated_base_cannot_publish_a_resumable_checkpoint(
    driver, tmp_path, monkeypatch
):
    directory = tmp_path / "changed-base"
    model, parent, original, train, plan, journal = fixture(driver, directory)
    update = driver.pilot.update

    def corrupt(*args):
        record = update(*args)
        with torch.no_grad():
            next(original.parameters()).add_(0.01)
        return record

    monkeypatch.setattr(driver.pilot, "update", corrupt)
    with pytest.raises(ValueError, match="base changed before checkpoint"):
        driver.run_training(
            model, parent, "mlp", original, train, train[:2], plan, directory, journal
        )
    assert not list(directory.glob("checkpoint-*.pt"))
    assert not journal["runs"]
    assert parent.mlp is original


def test_data_partition_and_per_block_causal_loss(driver):
    used = driver.pilot.spaced_indices(136, 16)
    remaining = driver.remaining_indices(136, used)
    assert len(remaining) == 120 and not set(remaining).intersection(used)
    left = torch.arange(12).reshape(3, 4)
    right = torch.arange(12, 24).reshape(3, 4)
    driver.assert_disjoint(left, right)
    with pytest.raises(ValueError, match="overlap"):
        driver.assert_disjoint(left, right, left[:1])
    model = tiny_model()
    tokens = (torch.arange(24).reshape(4, 6) % 15) + 1
    result = driver.per_block_loss(model, tokens, 2)
    # Keep batch shapes equal; changing them also changes floating-point matmuls.
    expected = driver.pilot.evaluate(model, tokens, 2)
    assert result["mean"] == pytest.approx(expected, abs=3e-7)
    assert len(result["block_losses"]) == 4


def test_cli_resume_binds_protocol_and_evaluates_only_after_all_runs(
    driver, tmp_path, monkeypatch
):
    config = json.loads(
        Path(driver.__file__)
        .with_name("hf_wave_gate_pride_long_horizon.json")
        .read_text()
    )
    train_body = "\n\n".join(
        "".join(chr(33 + i * 10 + j) for j in range(10)) for i in range(8)
    )
    transfer_body = "".join(chr(160 + i) for i in range(80))
    train_raw = (train_body + "<TRAIN-END>").encode()
    transfer_raw = (transfer_body + "<TRANSFER-END>").encode()
    config.update(
        model_snapshot="fixture",
        features=8,
        steps=4,
        batch_size=2,
        block_size=6,
        checkpoint_every=2,
        evaluate_every=2,
        seeds=[41],
        development_blocks=1,
        transfer_blocks=2,
        corpus_sha256=driver.pilot.digest(train_raw),
        corpus_end_marker="<TRAIN-END>",
        transfer_sha256=driver.pilot.digest(transfer_raw),
        transfer_end_marker="<TRANSFER-END>",
    )
    source = tmp_path / "config.json"
    original_config = json.dumps(config)
    source.write_text(original_config)
    corpus, transfer = tmp_path / "train.txt", tmp_path / "transfer.txt"
    corpus.write_bytes(train_raw)
    transfer.write_bytes(transfer_raw)
    model_dir = tmp_path / "fixture"
    model_dir.mkdir()
    output = tmp_path / "study"
    monkeypatch.setattr(
        driver.transformers.AutoTokenizer,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(
            encode=lambda text, **kw: [ord(c) for c in text]
        ),
    )
    monkeypatch.setattr(
        driver.transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *a, **k: tiny_model(256),
    )
    arguments = [
        "long-study",
        "--config",
        str(source),
        "--model-dir",
        str(model_dir),
        "--corpus",
        str(corpus),
        "--transfer-corpus",
        str(transfer),
        "--output-dir",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    train_function = driver.run_training
    evaluations = []
    evaluate = driver.per_block_loss

    def endpoint(*args, **kwargs):
        journal = json.loads((output / "journal.json").read_text())
        assert len(journal["runs"]) == 3
        assert all(run["status"] == "completed" for run in journal["runs"].values())
        evaluations.append(True)
        return evaluate(*args, **kwargs)

    def interrupt(*args, **kwargs):
        def stop(*_):
            raise RuntimeError("controlled interruption")

        return train_function(*args, **kwargs, after_checkpoint=stop)

    monkeypatch.setattr(driver, "per_block_loss", endpoint)
    monkeypatch.setattr(driver, "run_training", interrupt)
    with pytest.raises(RuntimeError, match="controlled interruption"):
        driver.main()
    assert not evaluations
    monkeypatch.setattr(driver, "run_training", train_function)
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    config["learning_rate"] *= 2
    source.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="identity differs"):
        driver.main()
    assert not evaluations
    source.write_text(original_config)
    driver.main()
    assert len(evaluations) == 8
    driver.main()
    assert len(evaluations) == 8

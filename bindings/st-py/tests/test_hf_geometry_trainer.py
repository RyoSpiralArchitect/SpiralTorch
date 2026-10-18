"""Real HF loop, frozen base and geometry-only checkpoints; no network downloads."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import spiraltorch as st

torch = pytest.importorskip("torch")
hf = pytest.importorskip("transformers")
pytest.importorskip("accelerate")


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


class TinyData(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        tokens = (torch.arange(6) * 3 + index) % 32
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens), "labels": tokens.clone()}


class CheckClipping(hf.TrainerCallback):
    def __init__(self):
        self.norms = []

    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        grads = [p.grad for p in model.geometry.parameters() if p.grad is not None]
        assert len(grads) == len(list(model.geometry.parameters()))
        assert all(torch.isfinite(g).all() for g in grads)
        norm = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).item()
        assert 0 < norm <= args.max_grad_norm * 1.001
        assert all(p.grad is None for p in model.base.parameters())
        self.norms.append(norm)


class StopAtTwo(hf.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_save = True
            control.should_training_stop = True


def make_trainer(output, architecture="gpt2", callbacks=(), **options):
    torch.set_num_threads(2)
    torch.manual_seed(123)
    if architecture == "gpt2":
        model = hf.GPT2LMHeadModel(hf.GPT2Config(
            vocab_size=32, n_positions=8, n_embd=8, n_layer=4, n_head=2,
            resid_pdrop=.1, embd_pdrop=.1, attn_pdrop=.1, use_cache=False))
        paths = [f"transformer.h.{i}.mlp" for i in range(4)]
    else:
        model = hf.LlamaForCausalLM(hf.LlamaConfig(
            vocab_size=32, hidden_size=8, intermediate_size=16, num_hidden_layers=4,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=16,
            attention_dropout=.1, use_cache=False))
        paths = [f"model.layers.{i}.mlp" for i in range(4)]
    model.requires_grad_(False)
    digest = hashlib.sha256(model.config.to_json_string().encode())
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.numpy().tobytes())
    base = "sha256:" + digest.hexdigest()
    torch.manual_seed(124)
    geometry = st.GeometryAdapterStack(dict(zip(paths, (
        st.WaveGateAdapter(8, log_radius=0., learnable_radius=True),
        st.ToposResonatorAdapter(8), st.EllipticAnchoredResidualAdapter(8),
        st.FractionalAngleGainHistoryAdapter(8, kernel_len=5, initial_angle=.2),
    ))))
    config = dict(output_dir=str(output), use_cpu=True, report_to=[], disable_tqdm=True,
                  max_steps=4, per_device_train_batch_size=2, gradient_accumulation_steps=2,
                  per_device_eval_batch_size=2, learning_rate=.01, optim="adamw_torch",
                  max_grad_norm=.0001, save_steps=2, logging_steps=1, seed=77, data_seed=78)
    config.update(options)
    return st.hf_geometry_trainer_class()(
        model=model, geometry=geometry, base_model_identity=base,
        args=hf.TrainingArguments(**config), train_dataset=TinyData(), eval_dataset=TinyData(),
        data_collator=hf.default_data_collator, callbacks=list(callbacks))


@pytest.mark.parametrize("architecture", ["gpt2", "llama"])
def test_actual_hf_loop_clips_updates_and_exactly_resumes(tmp_path, architecture):
    clipped = CheckClipping()
    full = make_trainer(tmp_path / "full", architecture, [clipped])
    base_before = copy.deepcopy(full.base_model.state_dict())
    geometry_before = copy.deepcopy(full.geometry.state_dict())
    full.train()
    assert full.state.global_step == 4 and len(clipped.norms) == 4
    assert not equal(geometry_before, full.geometry.state_dict())
    for index in range(4):
        names = [name for name, _ in full.geometry.named_parameters() if name.startswith(f"adapters.{index}.")]
        assert any(not torch.equal(geometry_before[name], full.geometry.state_dict()[name]) for name in names)
    assert equal(base_before, full.base_model.state_dict())
    # The threshold was actually reached, not just an inactive clipping branch.
    assert all(n > full.args.max_grad_norm * .99 for n in clipped.norms)
    expected_rng = torch.get_rng_state().clone()
    expected_optimizer = copy.deepcopy(full.optimizer.state_dict())
    expected_scheduler = copy.deepcopy(full.lr_scheduler.state_dict())
    expected_predictions = full.predict(TinyData()).predictions

    partial = make_trainer(tmp_path / "partial", architecture, [StopAtTwo()])
    partial.train()
    assert partial.state.global_step == 2
    checkpoint = tmp_path / "partial/checkpoint-2"
    assert {"spiraltorch-geometry.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
            "trainer_state.json"}.issubset({p.name for p in checkpoint.iterdir()})
    assert not (checkpoint / "model.safetensors").exists()
    assert not (checkpoint / "pytorch_model.bin").exists()
    pin = partial.last_geometry_artifact_id
    resumed = make_trainer(tmp_path / "resumed", architecture)
    resumed.train(resume_from_checkpoint=str(checkpoint), geometry_checkpoint_id=pin)
    assert resumed.state.global_step == 4
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert equal(full.geometry.state_dict(), resumed.geometry.state_dict())
    assert equal(expected_optimizer, resumed.optimizer.state_dict())
    assert equal(expected_scheduler, resumed.lr_scheduler.state_dict())
    assert equal(base_before, resumed.base_model.state_dict())
    assert (expected_predictions == resumed.predict(TinyData()).predictions).all()
    assert resumed.geometry._active_token is None


def test_resume_in_fresh_isolated_process(tmp_path):
    full = make_trainer(tmp_path / "full")
    full.train()
    expected = full.geometry.export_artifact(base_model_identity=full._geometry_base_id).artifact_id
    partial = make_trainer(tmp_path / "partial", callbacks=[StopAtTwo()])
    partial.train()
    payload = {"module": str(Path(__file__).resolve()), "output": str(tmp_path / "child"),
               "checkpoint": str(tmp_path / "partial/checkpoint-2"), "id": partial.last_geometry_artifact_id}
    script = """
import importlib.util, json, sys
p = json.load(sys.stdin)
spec = importlib.util.spec_from_file_location('geometry_training_test', p['module'])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
t = m.make_trainer(p['output'])
t.train(resume_from_checkpoint=p['checkpoint'], geometry_checkpoint_id=p['id'])
print('FINAL_ID=' + t.geometry.export_artifact(base_model_identity=t._geometry_base_id).artifact_id)
"""
    run = subprocess.run([sys.executable, "-I", "-B", "-c", script], input=json.dumps(payload),
                         text=True, capture_output=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert "FINAL_ID=" + expected in run.stdout


@pytest.fixture
def checkpoint(tmp_path):
    trainer = make_trainer(tmp_path / "partial", callbacks=[StopAtTwo()])
    trainer.train()
    return tmp_path / "partial/checkpoint-2", trainer.last_geometry_artifact_id


@pytest.mark.parametrize("missing", ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"])
def test_missing_companions_rejected_before_live_weights_change(tmp_path, checkpoint, missing):
    path, pin = checkpoint
    (path / missing).unlink()
    trainer = make_trainer(tmp_path / "other")
    before = copy.deepcopy(trainer.geometry.state_dict())
    with pytest.raises(ValueError, match="incomplete HF checkpoint"):
        trainer.train(resume_from_checkpoint=str(path), geometry_checkpoint_id=pin)
    assert equal(before, trainer.geometry.state_dict())


@pytest.mark.parametrize("change", ["missing_pin", "wrong_pin", "layout", "base"])
def test_wrong_identity_or_recipe_rejected_before_mutation(tmp_path, checkpoint, change):
    path, pin = checkpoint
    trainer = make_trainer(tmp_path / "other")
    if change == "layout":
        trainer.geometry.adapters[0].strength = .7
    elif change == "base":
        trainer._geometry_base_id = "sha256:" + "a" * 64
    before = copy.deepcopy(trainer.geometry.state_dict())
    with pytest.raises(ValueError):
        trainer.train(resume_from_checkpoint=str(path),
                      geometry_checkpoint_id=None if change == "missing_pin" else "sha256:" + "f" * 64 if change == "wrong_pin" else pin)
    assert equal(before, trainer.geometry.state_dict())


@pytest.mark.parametrize("options", [{"gradient_checkpointing": True}, {"torch_compile": True},
                                    {"save_only_model": True}, {"neftune_noise_alpha": 1.},
                                    {"push_to_hub": True}, {"group_by_length": True}])
def test_unsupported_modes_reject_without_training(tmp_path, options, monkeypatch):
    # TrainingArguments(torch_compile=True) mutates Accelerate's process env
    # before our constructor can reject it. Do not leak it into later tests.
    monkeypatch.setenv("ACCELERATE_DYNAMO_BACKEND", "NO")
    monkeypatch.setenv("ACCELERATE_DYNAMO_MODE", "default")
    with pytest.raises(ValueError):
        make_trainer(tmp_path, **options)


def test_padded_batch_and_late_base_unfreeze_reject(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = hf.default_data_collator([TinyData()[0]])
    batch["attention_mask"][0, -1] = 0
    with pytest.raises(ValueError, match="unpadded"):
        trainer.training_step(trainer.model, batch)
    assert trainer.geometry._active_token is None
    next(trainer.base_model.parameters()).requires_grad_()
    with pytest.raises(ValueError, match="remain frozen"):
        trainer.train()


def test_geometry_optimizer_excludes_frozen_base_and_missing_params(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.create_optimizer()
    trainer._validate_optimizer()
    trainer.optimizer = torch.optim.AdamW([next(trainer.geometry.parameters())])
    with pytest.raises(ValueError, match="exactly"):
        trainer.train()


def test_causal_label_smoothing_uses_base_model_name(tmp_path):
    trainer = make_trainer(tmp_path, label_smoothing_factor=.1)
    batch = hf.default_data_collator([TinyData()[0], TinyData()[1]])
    trainer.model.eval()
    with torch.no_grad(), trainer.geometry.attach(trainer.base_model):
        expected = trainer.label_smoother(trainer.base_model(**batch), batch["labels"], shift_labels=True)
        actual = trainer.compute_loss(trainer.model, dict(batch))
    assert torch.equal(expected, actual)


def test_evaluation_actually_uses_geometry_and_detaches_after_failure(tmp_path):
    trainer = make_trainer(tmp_path)
    with torch.no_grad():
        trainer.geometry.adapters[0].gate.copy_(torch.linspace(-.4, .5, 8))
        trainer.geometry.adapters[0].bias.copy_(torch.linspace(.1, -.2, 8))
    batch = hf.default_data_collator([TinyData()[0], TinyData()[1]])
    trainer.model.eval()
    with torch.no_grad():
        bare = trainer.base_model(**batch).loss
        with trainer.geometry.attach(trainer.base_model):
            expected = trainer.base_model(**batch).loss
    actual, _, _ = trainer.prediction_step(trainer.model, batch, prediction_loss_only=True)
    assert torch.equal(expected, actual) and not torch.equal(bare, actual)
    with pytest.raises(RuntimeError, match="use Trainer"):
        trainer.model(**batch)
    broken = dict(batch, use_cache=True)
    with pytest.raises(ValueError, match="cached"):
        trainer.prediction_step(trainer.model, broken, prediction_loss_only=True)
    assert trainer.geometry._active_token is None
    actual, _, _ = trainer.prediction_step(trainer.model, batch, prediction_loss_only=True)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("smoothing", [0., .1])
@pytest.mark.parametrize("masked", [False, True])
def test_accumulation_matches_full_batch_gradients(tmp_path, smoothing, masked):
    class Capture(hf.TrainerCallback):
        def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
            self.gradients = [p.grad.detach().clone() for p in model.geometry.parameters()]

    class MaskedData(TinyData):
        def __getitem__(self, index):
            batch = super().__getitem__(index)
            batch["labels"][: 5 - index % 5] = -100
            return batch

    captures = []
    for batch_size, accumulation in [(4, 1), (2, 2)]:
        capture = Capture()
        trainer = make_trainer(tmp_path / str(batch_size), callbacks=[capture],
                               max_steps=1, per_device_train_batch_size=batch_size,
                               gradient_accumulation_steps=accumulation, max_grad_norm=0.,
                               label_smoothing_factor=smoothing)
        if masked:
            trainer.train_dataset = MaskedData()
        for module in trainer.base_model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        trainer.base_model.config.attn_pdrop = 0.
        trainer.train()
        captures.append(capture.gradients)
    assert len(captures[0]) == len(captures[1])
    for a, b in zip(*captures):
        torch.testing.assert_close(a, b, atol=1e-7, rtol=2e-5)


def test_causal_count_excludes_first_and_ignored_labels(tmp_path):
    trainer = make_trainer(tmp_path)
    batch = hf.default_data_collator([TinyData()[0], TinyData()[1]])
    batch["labels"][0, 2] = -100
    assert trainer._get_num_items_in_batch([batch], torch.device("cpu")).item() == 9


def test_accelerate_environment_cannot_silently_enable_compilation(tmp_path, monkeypatch):
    monkeypatch.setenv("ACCELERATE_DYNAMO_BACKEND", "INDUCTOR")
    with pytest.raises(ValueError, match="compiled"):
        make_trainer(tmp_path)


@pytest.mark.parametrize("training", [True, False])
def test_existing_but_unexecuted_placement_fails_closed(tmp_path, training):
    template = make_trainer(tmp_path)
    template.base_model.add_module("unused_geometry_target", torch.nn.Linear(8, 8).requires_grad_(False))
    geometry = st.GeometryAdapterStack({
        "transformer.h.0.mlp": st.WaveGateAdapter(8),
        "unused_geometry_target": st.WaveGateAdapter(8),
    })
    trainer = st.hf_geometry_trainer_class()(
        model=template.base_model, geometry=geometry, base_model_identity=template._geometry_base_id,
        args=template.args, train_dataset=TinyData(), data_collator=hf.default_data_collator)
    before = copy.deepcopy(geometry.state_dict())
    with pytest.raises(RuntimeError, match="placements were not executed.*unused_geometry_target"):
        if training:
            trainer.train()
        else:
            trainer.predict(TinyData())
    assert equal(before, geometry.state_dict())
    assert geometry._active_token is None
    assert not template.base_model.unused_geometry_target._forward_hooks


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="actual MPS device required")
def test_mps_rng_companion_and_exact_dropout_resume(tmp_path):
    options = dict(use_cpu=False, max_steps=3, save_steps=1)
    full = make_trainer(tmp_path / "full", **options)
    assert full.args.device.type == "mps"
    full.train()
    expected_rng = torch.mps.get_rng_state().clone()
    expected = copy.deepcopy(full.geometry.state_dict())
    partial = make_trainer(tmp_path / "partial", callbacks=[StopAtTwo()], **options)
    partial.train()
    checkpoint = tmp_path / "partial/checkpoint-2"
    assert (checkpoint / "spiraltorch-mps-rng.pt").is_file()
    resumed = make_trainer(tmp_path / "resumed", **options)
    resumed.train(resume_from_checkpoint=str(checkpoint), geometry_checkpoint_id=partial.last_geometry_artifact_id)
    assert torch.equal(expected_rng, torch.mps.get_rng_state())
    assert equal(expected, resumed.geometry.state_dict())
    assert equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    (checkpoint / "spiraltorch-mps-rng.pt").unlink()
    missing = make_trainer(tmp_path / "missing", **options)
    with pytest.raises(ValueError, match="incomplete.*spiraltorch-mps-rng"):
        missing.train(resume_from_checkpoint=str(checkpoint), geometry_checkpoint_id=partial.last_geometry_artifact_id)


@pytest.mark.parametrize("stop", [7, 8, 15])
def test_resume_inside_a_later_epoch_preserves_actual_training_order(tmp_path, stop):
    class FourteenRows(TinyData):
        def __len__(self):
            return 14

    class StopAtBoundary(hf.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == stop:
                control.should_save = control.should_training_stop = True

    def build(name, callbacks=()):
        trainer = make_trainer(tmp_path / name, callbacks=callbacks, max_steps=17,
                               save_steps=stop, gradient_accumulation_steps=1)
        trainer.train_dataset = FourteenRows()
        return trainer

    full = build("full")
    full.train()
    partial = build("partial", [StopAtBoundary()])
    partial.train()
    assert partial.state.epoch >= 1
    resumed = build("resumed")
    resumed.train(resume_from_checkpoint=str(tmp_path / f"partial/checkpoint-{stop}"),
                  geometry_checkpoint_id=partial.last_geometry_artifact_id)
    assert equal(full.geometry.state_dict(), resumed.geometry.state_dict())
    assert equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    assert equal(full.lr_scheduler.state_dict(), resumed.lr_scheduler.state_dict())


def test_epoch_sampler_matches_hf_order_without_using_global_rng(tmp_path):
    from accelerate.data_loader import SeedableRandomSampler, skip_first_batches

    trainer = make_trainer(tmp_path)
    trainer.state.epoch = 2.25
    sampler = trainer._get_train_sampler()
    rng = torch.get_rng_state().clone()
    expected = SeedableRandomSampler(trainer.train_dataset, data_seed=trainer.args.data_seed)
    expected.set_epoch(2)
    assert list(sampler) == list(expected)
    assert torch.equal(rng, torch.get_rng_state())
    loader = trainer.get_train_dataloader()
    loader.set_epoch(2)
    skipped = skip_first_batches(loader, 2)
    actual = [int(row) for batch in skipped for row in batch["input_ids"][:, 0]]
    expected.set_epoch(2)
    assert actual == list(expected)[4:]


def test_streaming_training_rejects_instead_of_claiming_cursor_replay(tmp_path):
    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            yield TinyData()[0]

    trainer = make_trainer(tmp_path)
    trainer.train_dataset = Stream()
    with pytest.raises(ValueError, match="finite map-style"):
        trainer.train()


def test_early_epoch_callback_rejects_before_checkpoint_save(tmp_path):
    class SkipRestOfEpoch(hf.TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 2:
                control.should_epoch_stop = control.should_save = True

    trainer = make_trainer(tmp_path, callbacks=[SkipRestOfEpoch()], max_steps=9,
                           gradient_accumulation_steps=1, save_steps=99)
    with pytest.raises(ValueError, match="early epoch stopping"):
        trainer.train()
    assert trainer.state.global_step == 2
    assert not list(tmp_path.glob("checkpoint-*"))
    assert trainer.geometry._active_token is None


@pytest.mark.parametrize("event", ["log", "evaluate"])
def test_late_epoch_stop_callback_rejects_before_checkpoint_write(tmp_path, event):
    class SkipAfterReporting(hf.TrainerCallback):
        def on_log(self, args, state, control, **kwargs):
            if event == "log" and state.global_step == 2:
                control.should_epoch_stop = control.should_save = True

        def on_evaluate(self, args, state, control, **kwargs):
            if event == "evaluate" and state.global_step == 2:
                control.should_epoch_stop = control.should_save = True

    trainer = make_trainer(tmp_path, callbacks=[SkipAfterReporting()], max_steps=9,
                           gradient_accumulation_steps=1, eval_strategy="steps", eval_steps=2,
                           save_steps=2)
    with pytest.raises(ValueError, match="early epoch stopping"):
        trainer.train()
    assert trainer.state.global_step == 2
    assert not list(tmp_path.glob("checkpoint-*"))

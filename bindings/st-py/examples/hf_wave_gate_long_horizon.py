"""Offline long-horizon learning with restartable cursors and endpoint-only tests.

Python owns experimental orchestration, not the geometric forward or derivative.
The output directory has one writer. Checkpoints are immutable; an atomic journal
publishes their hashes only after they have been flushed successfully.
"""

import argparse
import copy
import fcntl
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch
import transformers

import hf_wave_gate_conditioning as pilot


ARMS = ["tangent", "wave_gate_radius4", "wave_gate_learnable_radius4"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(payload):
    return pilot.digest(json.dumps(payload, sort_keys=True, allow_nan=False).encode())


class TerminalStudyError(ValueError):
    """A declared protocol failure, not an interruption eligible for replay."""

    def __init__(self, message, *, details):
        super().__init__(message)
        identity(details)
        self.details = copy.deepcopy(details)


def atomic_json(path, value):
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".journal-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def study_lock(directory):
    with (directory / ".writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another writer holds this study directory") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def save_checkpoint(directory, payload):
    fd, temporary = tempfile.mkstemp(dir=directory, prefix="checkpoint-", suffix=".pt")
    path = Path(temporary)
    with os.fdopen(fd, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return {"filename": path.name, "sha256": pilot.digest(path.read_bytes())}


def load_checkpoint(directory, receipt, study_id, run_key):
    name = receipt["filename"]
    require(
        Path(name).name == name and name.startswith("checkpoint-"),
        "invalid checkpoint path",
    )
    path = directory / name
    require(not path.is_symlink(), "checkpoint must not be a symlink")
    require(
        pilot.digest(path.read_bytes()) == receipt["sha256"], "checkpoint hash mismatch"
    )
    saved = torch.load(path, weights_only=True, map_location="cpu")
    require(
        saved["study_id"] == study_id and saved["run_key"] == run_key,
        "checkpoint identity mismatch",
    )
    return saved


def remaining_indices(count, used):
    used = set(used)
    require(all(0 <= index < count for index in used), "invalid development index")
    return [i for i in range(count) if i not in used]


def block_hashes(blocks):
    return [pilot.digest(row.numpy().tobytes()) for row in blocks]


def assert_disjoint(train, *heldout):
    seen = set(block_hashes(train))
    for blocks in heldout:
        current = set(block_hashes(blocks))
        require(not seen.intersection(current), "training/evaluation blocks overlap")
        seen.update(current)


def per_block_loss(model, blocks, batch_size):
    losses = []
    with torch.no_grad():
        for start in range(0, len(blocks), batch_size):
            batch = blocks[start : start + batch_size]
            logits = model(batch).logits[:, :-1, :].contiguous()
            loss = (
                torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch[:, 1:].contiguous().reshape(-1),
                    reduction="none",
                )
                .reshape(batch.shape[0], -1)
                .mean(dim=1)
            )
            require(bool(torch.isfinite(loss).all()), "nonfinite endpoint loss")
            losses.extend(loss.tolist())
    require(bool(losses), "empty endpoint set")
    return {"mean": sum(losses) / len(losses), "block_losses": losses}


def endpoint_gate(journal, plan, directory):
    require(journal.get("status") != "terminal_failure" and "terminal_failure" not in journal,
            "terminal protocol failure locks endpoint evaluation")
    expected = [
        f"{seed}:{arm}"
        for seed in plan["config"]["seeds"]
        for arm in plan["config"]["arms"]
    ]
    require(journal["study_id"] == plan["study_id"], "journal identity mismatch")
    require(
        set(journal["runs"]) == set(expected),
        "all planned runs must finish before endpoint evaluation",
    )
    for key in expected:
        entry = journal["runs"][key]
        require(
            entry["status"] == "completed",
            "endpoint evaluation is locked until training completes",
        )
        saved = load_checkpoint(directory, entry["checkpoint"], plan["study_id"], key)
        require(
            saved.get("frozen_base_verified") is True,
            "checkpoint has no frozen-base verification",
        )
        require(
            saved["cursor"] == plan["config"]["steps"], "incomplete endpoint cursor"
        )
        require(
            entry["resume_next_update_equal"] and entry["frozen_base_unchanged"],
            "unverified endpoint",
        )
    return expected


def make_adapter(arm, config, seed, factory):
    return (
        pilot.adapter_for(arm, config)
        if factory is None
        else factory(arm, config, seed)
    )


def make_optimizer(adapter, arm, config, factory):
    return (
        torch.optim.Adam(adapter.parameters(), lr=config["learning_rate"])
        if factory is None else factory(adapter, arm, config)
    )


def run_training(
    model,
    parent,
    child,
    original,
    train,
    dev_probe,
    plan,
    directory,
    journal,
    after_checkpoint=None,
    *,
    adapter_factory=None,
    optimizer_factory=None,
):
    require(journal.get("status") != "terminal_failure" and "terminal_failure" not in journal,
            "terminal protocol failure cannot resume")
    config, study_id = plan["config"], plan["study_id"]
    base_hash = pilot.model_digest(model)
    require(base_hash == plan["base_parameter_sha256"], "base model differs from plan")
    for seed in config["seeds"]:
        batches = plan["batch_schedules"][str(seed)]
        for arm in config["arms"]:
            key = f"{seed}:{arm}"
            entry = journal["runs"].get(key)
            saved = (
                load_checkpoint(directory, entry["checkpoint"], study_id, key)
                if entry
                else None
            )
            if saved:
                require(
                    saved.get("frozen_base_verified") is True,
                    "checkpoint has no frozen-base verification",
                )
                require(
                    entry["cursor"] == saved["cursor"],
                    "journal cursor differs from checkpoint",
                )
            if entry and entry["status"] == "completed":
                require(
                    saved["cursor"] == config["steps"], "incomplete completed cursor"
                )
                continue
            adapter = make_adapter(arm, config, seed, adapter_factory)
            initial_hash = pilot.model_digest(adapter)
            projection_hash = (
                pilot.model_digest(adapter, exclude={"raw_mix"})
                if hasattr(adapter, "raw_mix") else None
            )
            optimizer = make_optimizer(adapter, arm, config, optimizer_factory)
            cursor, records, development = 0, [], []
            if saved:
                require(
                    saved.get("initial_parameter_sha256") == initial_hash,
                    "resume initialization differs",
                )
                if projection_hash is not None:
                    require(
                        saved.get("initial_projection_sha256") == projection_hash,
                        "resume projection initialization differs",
                    )
                adapter.load_state_dict(saved["adapter"])
                optimizer.load_state_dict(saved["optimizer"])
                cursor, records, development = (
                    saved["cursor"],
                    saved["records"],
                    saved["development"],
                )
                require(
                    0 <= cursor <= config["steps"] and len(records) == cursor,
                    "invalid resume cursor",
                )
                require(
                    all(
                        row["step"] == i + 1 and row["batch_indices"] == batches[i]
                        for i, row in enumerate(records)
                    ),
                    "checkpoint batch history mismatch",
                )
            parent.add_module(child, torch.nn.Sequential(original, adapter))
            phase, attempted_step = "initial_identity", 0
            try:
                if cursor == 0:
                    with torch.no_grad():
                        observed = model(train[:1]).logits.detach()
                    parent.add_module(child, original)
                    with torch.no_grad():
                        baseline = model(train[:1]).logits.detach()
                    parent.add_module(child, torch.nn.Sequential(original, adapter))
                    require(
                        torch.equal(observed, baseline),
                        "adapter is not identity initialized",
                    )
                for index in range(cursor, config["steps"]):
                    phase, attempted_step = "primary_update", index + 1
                    row = pilot.update(model, adapter, optimizer, train[batches[index]])
                    row.update(step=index + 1, batch_indices=batches[index])
                    records.append(row)
                    if (index + 1) % config[
                        "evaluate_every"
                    ] == 0 or index + 1 == config["steps"]:
                        phase = "development"
                        development.append(
                            {
                                "step": index + 1,
                                "loss": pilot.evaluate(
                                    model, dev_probe, config["batch_size"]
                                ),
                            }
                        )
                        print(
                            f"{key} step={index + 1} development={development[-1]['loss']:.6f}",
                            flush=True,
                        )
                    if (index + 1) % config[
                        "checkpoint_every"
                    ] == 0 or index + 1 == config["steps"]:
                        # A failed base invariant must not publish a resumable checkpoint.
                        parent.add_module(child, original)
                        try:
                            require(
                                pilot.model_digest(model) == base_hash,
                                "frozen base changed before checkpoint",
                            )
                            require(
                                all(p.grad is None for p in model.parameters()),
                                "base gradients changed before checkpoint",
                            )
                        finally:
                            parent.add_module(
                                child, torch.nn.Sequential(original, adapter)
                            )
                        payload = {
                            "study_id": study_id,
                            "run_key": key,
                            "cursor": index + 1,
                            "adapter": adapter.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "records": records,
                            "development": development,
                            "frozen_base_verified": True,
                            "initial_parameter_sha256": initial_hash,
                        }
                        if projection_hash is not None:
                            payload["initial_projection_sha256"] = projection_hash
                        receipt = save_checkpoint(directory, payload)
                        journal["runs"][key] = {
                            "status": "training",
                            "cursor": index + 1,
                            "checkpoint": receipt,
                        }
                        atomic_json(directory / "journal.json", journal)
                        if after_checkpoint:
                            after_checkpoint(key, index + 1)
                entry = journal["runs"][key]
                endpoint = load_checkpoint(
                    directory, entry["checkpoint"], study_id, key
                )
                require(
                    endpoint["cursor"] == config["steps"], "missing final checkpoint"
                )
                # These two updates verify continuation, never replace the frozen endpoint.
                phase, attempted_step = "continuation_reference", config["steps"] + 1
                pilot.update(model, adapter, optimizer, train[batches[-1]])
                restored = make_adapter(arm, config, seed, adapter_factory)
                restored.load_state_dict(endpoint["adapter"])
                resumed = make_optimizer(restored, arm, config, optimizer_factory)
                resumed.load_state_dict(copy.deepcopy(endpoint["optimizer"]))
                parent.add_module(child, torch.nn.Sequential(original, restored))
                phase = "continuation_replay"
                pilot.update(model, restored, resumed, train[batches[-1]])
                require(
                    pilot.equal_state(adapter.state_dict(), restored.state_dict()),
                    "next parameter update differs",
                )
                require(
                    pilot.equal_state(optimizer.state_dict(), resumed.state_dict()),
                    "next optimizer state differs",
                )
            except TerminalStudyError as error:
                journal["status"] = "terminal_failure"
                journal["terminal_failure"] = {
                    "schema": "spiraltorch.terminal_study_failure.v1",
                    "study_id": study_id, "run_key": key, "phase": phase,
                    "attempted_step": attempted_step,
                    "completed_primary_updates_in_run": len(records),
                    "failed_update_resumable": False,
                    "optimizer_step_may_have_executed": phase in {
                        "primary_update", "continuation_reference", "continuation_replay"},
                    "details": error.details,
                }
                atomic_json(directory / "journal.json", journal)
                raise
            finally:
                parent.add_module(child, original)
            require(pilot.model_digest(model) == base_hash, "frozen base changed")
            require(
                all(p.grad is None for p in model.parameters()),
                "base gradients changed",
            )
            entry.update(
                status="completed",
                resume_next_update_equal=True,
                frozen_base_unchanged=True,
                extra_validation_updates=2,
            )
            atomic_json(directory / "journal.json", journal)
    journal["status"] = "training_completed"
    atomic_json(directory / "journal.json", journal)


def run_endpoints(
    model,
    parent,
    child,
    original,
    evaluation,
    plan,
    directory,
    journal,
    *,
    adapter_factory=None,
    result_schema="spiraltorch.wave_gate_long_horizon.v1",
):
    keys = endpoint_gate(journal, plan, directory)
    require(
        pilot.model_digest(model) == plan["base_parameter_sha256"],
        "base differs before endpoint evaluation",
    )
    config = plan["config"]
    report = {
        "schema": result_schema,
        "status": "evaluating",
        "study_id": plan["study_id"],
        "baseline": {},
        "runs": [],
    }
    for label, blocks in evaluation.items():
        report["baseline"][label] = per_block_loss(model, blocks, config["batch_size"])
    for key in keys:
        entry = journal["runs"][key]
        saved = load_checkpoint(directory, entry["checkpoint"], plan["study_id"], key)
        seed, arm = key.split(":", 1)
        adapter = make_adapter(arm, config, int(seed), adapter_factory)
        require(
            saved.get("initial_parameter_sha256") == pilot.model_digest(adapter),
            "endpoint initialization differs",
        )
        gated_metadata = {}
        if hasattr(adapter, "raw_mix"):
            require(
                saved.get("initial_projection_sha256")
                == pilot.model_digest(adapter, exclude={"raw_mix"}),
                "endpoint projection initialization differs",
            )
            gated_metadata["initial_projection_sha256"] = saved["initial_projection_sha256"]
        adapter.load_state_dict(saved["adapter"])
        gated_metadata.update({f"final_{key}": value
                               for key, value in pilot.gain_snapshot(adapter).items()})
        gated_metadata.update({f"final_{key}": value
                               for key, value in pilot.angle_snapshot(adapter).items()})
        if hasattr(adapter, "raw_mix"):
            gated_metadata["final_raw_mix"] = float(adapter.raw_mix.detach())
        if hasattr(adapter, "log_alpha"):
            gated_metadata["final_log_alpha"] = float(adapter.log_alpha.detach())
            gated_metadata["final_alpha"] = float(adapter.log_alpha.detach().exp())
        if hasattr(adapter, "logit_decay"):
            gated_metadata["final_logit_decay"] = float(adapter.logit_decay.detach())
            gated_metadata["final_decay"] = float(adapter.logit_decay.detach().sigmoid())
        radius = getattr(adapter, "log_radius", None)
        parent.add_module(child, torch.nn.Sequential(original, adapter))
        try:
            scores = {
                label: per_block_loss(model, blocks, config["batch_size"])
                for label, blocks in evaluation.items()
            }
        finally:
            parent.add_module(child, original)
        report["runs"].append(
            {
                "run_key": key,
                "checkpoint": entry["checkpoint"],
                "scores": scores,
                "records": saved["records"],
                "development": saved["development"],
                "parameter_count": sum(p.numel() for p in adapter.parameters()),
                "trainable_parameter_count": sum(
                    p.numel() for p in adapter.parameters() if p.requires_grad
                ),
                "final_log_radius": None if radius is None else float(radius.detach()),
                "initial_parameter_sha256": saved.get("initial_parameter_sha256"),
                "resume_next_update_equal": entry["resume_next_update_equal"],
                **gated_metadata,
            }
        )
        atomic_json(directory / "results.json", report)
    require(
        pilot.model_digest(model) == plan["base_parameter_sha256"],
        "base changed during evaluation",
    )
    report["status"] = "completed"
    atomic_json(directory / "results.json", report)
    journal["status"] = "completed"
    journal["results_sha256"] = pilot.digest((directory / "results.json").read_bytes())
    atomic_json(directory / "journal.json", journal)
    return report


def completed_result(directory, journal, plan):
    if journal["status"] != "completed":
        return None
    endpoint_gate(journal, plan, directory)
    raw = (directory / "results.json").read_bytes()
    require(
        pilot.digest(raw) == journal["results_sha256"], "completed result hash mismatch"
    )
    result = json.loads(raw)
    require(
        result["status"] == "completed" and result["study_id"] == plan["study_id"],
        "completed result identity mismatch",
    )
    return result


def prepare_data(tokenizer, corpus_bytes, transfer_bytes, config):
    require(
        pilot.digest(corpus_bytes) == config["corpus_sha256"],
        "training corpus hash mismatch",
    )
    require(
        pilot.digest(transfer_bytes) == config["transfer_sha256"],
        "transfer corpus hash mismatch",
    )
    body, _ = pilot.corpus_body(corpus_bytes.decode(), config["corpus_end_marker"])
    other_body, _ = pilot.corpus_body(
        transfer_bytes.decode(), config["transfer_end_marker"]
    )
    train_text, dev_text, cut = pilot.split_text(body)

    def pack(text):
        return pilot.pack_tokens(
            tokenizer.encode(text, add_special_tokens=False), config["block_size"]
        )

    train, dev, transfer = pack(train_text), pack(dev_text), pack(other_body)
    development = pilot.spaced_indices(len(dev), config["development_blocks"])
    endpoint = remaining_indices(len(dev), development)
    transfer_indices = pilot.spaced_indices(len(transfer), config["transfer_blocks"])
    require(bool(endpoint), "no unused within-book evaluation blocks")
    evaluation = {
        "pride_unused_tail": dev[endpoint],
        "alice_transfer": transfer[transfer_indices],
    }
    assert_disjoint(train, dev[development], *evaluation.values())
    metadata = {
        "split_character_offset": cut,
        "train_blocks": len(train),
        "development_blocks": len(dev),
        "development_indices": development,
        "endpoint_indices": endpoint,
        "transfer_indices": transfer_indices,
        "train_tokens_sha256": pilot.digest(train.numpy().tobytes()),
        "development_tokens_sha256": pilot.digest(dev.numpy().tobytes()),
        "transfer_tokens_sha256": pilot.digest(transfer.numpy().tobytes()),
        "evaluation_block_hashes": {k: block_hashes(v) for k, v in evaluation.items()},
    }
    return train, dev[development], evaluation, metadata


def main(
    *,
    arms=None,
    adapter_factory=None,
    adapter_sources=None,
    optimizer_factory=None,
    result_schema="spiraltorch.wave_gate_long_horizon.v1",
):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "model-dir", "corpus", "transfer-corpus", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    require(optimizer_factory is None or bool(adapter_sources), "custom optimizer sources must be bound")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    require(config["arms"] == (ARMS if arms is None else arms), "unrecognized arms")
    require(
        args.model_dir.is_dir() and args.model_dir.name == config["model_snapshot"],
        "model snapshot mismatch",
    )
    for key in (
        "steps",
        "batch_size",
        "block_size",
        "checkpoint_every",
        "evaluate_every",
        "threads",
    ):
        require(type(config[key]) is int and config[key] > 0, f"invalid {key}")
    require(config["steps"] > 1, "at least two updates required")
    require(
        len(set(config["seeds"])) == len(config["seeds"]) > 0,
        "duplicate or empty seeds",
    )
    require(all(type(seed) is int for seed in config["seeds"]), "invalid seed")
    if args.resume:
        require(args.output_dir.is_dir(), "resume requires an existing study directory")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    with study_lock(args.output_dir):
        torch.set_num_threads(config["threads"])
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_dir, local_files_only=True
        )
        tokenizer.model_max_length = 10**9
        train, development, evaluation, data = prepare_data(
            tokenizer,
            args.corpus.read_bytes(),
            args.transfer_corpus.read_bytes(),
            config,
        )
        model = (
            transformers.AutoModelForCausalLM.from_pretrained(
                args.model_dir, local_files_only=True, torch_dtype=torch.float32
            )
            .cpu()
            .eval()
            .requires_grad_(False)
        )
        model.config.use_cache = False
        require(
            config["block_size"] <= model.config.max_position_embeddings,
            "context exceeded",
        )
        import spiraltorch.spiraltorch as native
        import spiraltorch.geometry_autograd as bridge

        source = Path(__file__)
        binding = {
            "config_sha256": pilot.digest(config_bytes),
            "config": config,
            "data": data,
            "script_sha256": pilot.digest(source.read_bytes()),
            "helper_sha256": pilot.digest(Path(pilot.__file__).read_bytes()),
            "native_sha256": pilot.digest(Path(native.__file__).read_bytes()),
            "bridge_sha256": pilot.digest(Path(bridge.__file__).read_bytes()),
            "base_parameter_sha256": pilot.model_digest(model),
            "model_config_sha256": identity(model.config.to_dict()),
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "result_schema": result_schema,
        }
        if adapter_factory is not None:
            require(bool(adapter_sources), "custom adapter sources must be bound")
            binding["adapter_sources_sha256"] = {
                name: pilot.digest(Path(path).read_bytes())
                for name, path in adapter_sources.items()
            }
        study_id = identity(binding)
        plan = {
            **binding,
            "study_id": study_id,
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "batch_schedules": {
                str(seed): pilot.schedule(
                    seed, len(train), config["steps"] + 1, config["batch_size"]
                )
                for seed in config["seeds"]
            },
        }
        if args.resume:
            existing = json.loads((args.output_dir / "plan.json").read_text())
            require(
                existing["study_id"] == study_id
                and identity({key: existing.get(key) for key in binding}) == study_id
                and existing["batch_schedules"] == plan["batch_schedules"],
                "resume protocol/runtime/data identity differs",
            )
            plan = existing
            journal = json.loads((args.output_dir / "journal.json").read_text())
            require(journal["study_id"] == study_id, "journal identity mismatch")
            if completed_result(args.output_dir, journal, plan) is not None:
                print("Study already completed; endpoints were not rerun.", flush=True)
                return
        else:
            journal = {"study_id": study_id, "status": "training", "runs": {}}
            atomic_json(args.output_dir / "plan.json", plan)
            atomic_json(args.output_dir / "journal.json", journal)
        parent_name, child = config["block"].rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        original = parent.get_submodule(child)
        run_training(
            model,
            parent,
            child,
            original,
            train,
            development,
            plan,
            args.output_dir,
            journal,
            adapter_factory=adapter_factory,
            optimizer_factory=optimizer_factory,
        )
        run_endpoints(
            model,
            parent,
            child,
            original,
            evaluation,
            plan,
            args.output_dir,
            journal,
            adapter_factory=adapter_factory,
            result_schema=result_schema,
        )


if __name__ == "__main__":
    main()

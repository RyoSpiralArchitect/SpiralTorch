"""Offline paired WaveGate learning on a local corpus, with Rust conditioning.

The config is fixed before training. Seed affects minibatch order, not the
zero-initialized adapter. This is a bounded study, not a speed benchmark.
"""

import argparse
import copy
import hashlib
import json
import math
import random
import subprocess
from pathlib import Path

import torch
import transformers
import spiraltorch as st


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def model_digest(model, *, exclude=()):
    result = hashlib.sha256()
    for name, value in model.named_parameters():
        if name in exclude:
            continue
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def split_text(text):
    cut = text.rfind("\n\n", 0, int(len(text) * 0.9))
    if cut <= 0 or cut >= len(text):
        raise ValueError("corpus needs a paragraph boundary before its 90% split")
    return text[:cut], text[cut:], cut


def corpus_body(text, end_marker):
    if text.count(end_marker) != 1:
        raise ValueError("expected exactly one corpus end marker")
    offset = text.index(end_marker)
    return text[:offset], offset


def pack_tokens(tokens, block_size):
    if block_size < 2 or len(tokens) < block_size:
        raise ValueError("at least one complete block of two or more tokens required")
    count = len(tokens) // block_size
    return torch.tensor(tokens[: count * block_size], dtype=torch.long).reshape(
        count, block_size
    )


def spaced_indices(count, selected):
    if selected < 1 or count < selected:
        raise ValueError("requested probe is larger than available blocks")
    if selected == 1:
        return [0]
    return [i * (count - 1) // (selected - 1) for i in range(selected)]


def schedule(seed, count, steps, batch_size):
    if count < batch_size or steps < 1 or batch_size < 1:
        raise ValueError("invalid schedule dimensions")
    rng = random.Random(seed)
    indices = []
    while len(indices) < steps * batch_size:
        epoch = list(range(count))
        rng.shuffle(epoch)
        indices.extend(epoch)
    return [indices[i * batch_size : (i + 1) * batch_size] for i in range(steps)]


class ObservedWaveGate(st.WaveGateAdapter):
    last_conditioning = None

    def forward(self, value):
        output, self.last_conditioning = self.forward_with_conditioning(value)
        return output


class TangentControl(st.WaveGateAdapter):
    last_conditioning = None

    def forward(self, value):
        # The protocol fixes curvature=-1, so the initial projection slope is 1.
        if self.get_extra_state()["kernel"]["curvature"] != -1.0:
            raise ValueError("this tangent control requires curvature=-1")
        return value + self.strength * (value * self.gate + self.bias)


def adapter_for(arm, config):
    constructor = TangentControl if arm == "tangent" else ObservedWaveGate
    radius_options = {}
    if arm in RADIUS_ARMS[2:]:
        radius_options = {
            "log_radius": math.log(4.0) if arm.endswith("radius4") else 0.0,
            "learnable_radius": "learnable" in arm,
        }
    return constructor(
        config["features"],
        strength=0.0 if arm == "off" else config["strength"],
        curvature=-1.0,
        saturation=10000.0 if arm == "wave_gate_wide_saturation" else 1.0,
        porosity=0.05,
        **radius_options,
    )


LEGACY_ARMS = ["off", "tangent", "wave_gate", "wave_gate_wide_saturation"]
RADIUS_ARMS = [
    "off",
    "tangent",
    "wave_gate_radius1",
    "wave_gate_radius4",
    "wave_gate_learnable_radius1",
    "wave_gate_learnable_radius4",
]


def evaluate(model, blocks, batch_size):
    total, targets = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(blocks), batch_size):
            batch = blocks[start : start + batch_size]
            loss = model(batch, labels=batch).loss
            if not torch.isfinite(loss):
                raise ValueError("nonfinite evaluation loss")
            count = batch.shape[0] * (batch.shape[1] - 1)
            total += float(loss) * count
            targets += count
    return total / targets


def gain_snapshot(adapter):
    """Read the adapter's actual gain; native adapters delegate its chart to Rust."""
    if not hasattr(adapter, "log_gain"):
        return {}
    coordinate = adapter.log_gain
    if coordinate.ndim != 0 or coordinate.dtype != torch.float32:
        raise ValueError("gain telemetry requires scalar float32 log_gain")
    log_gain, gain = float(coordinate.detach()), float(adapter.gain)
    if not math.isfinite(log_gain) or not math.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be positive finite f32")
    gate = adapter.gate.detach().double().tanh() * adapter.strength * gain
    scale = float(gate.norm())
    if not math.isfinite(scale):
        raise ValueError("nonfinite effective history gate scale")
    return {"log_gain": log_gain, "gain": gain, "effective_history_gate_l2": scale}


def angle_snapshot(adapter):
    if not hasattr(adapter, "history_angle"):
        return {}
    coordinate = adapter.history_angle
    if coordinate.ndim != 0 or coordinate.dtype != torch.float32:
        raise ValueError("angle telemetry requires scalar float32")
    result = {"history_angle": float(coordinate.detach())}
    alpha = getattr(adapter, "alpha", None)
    if alpha is not None:
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("angle chart must return a positive finite order")
        result["alpha"] = float(alpha)
    return result


def update(model, adapter, optimizer, batch):
    optimizer.zero_grad()
    loss = model(batch, labels=batch).loss
    if not torch.isfinite(loss):
        raise ValueError("nonfinite training loss")
    loss.backward()
    record = {
        "loss": float(loss.detach()),
        "conditioning": getattr(adapter, "last_conditioning", None),
    }
    record.update({f"{key}_before_update": value for key, value in gain_snapshot(adapter).items()})
    record.update({f"{key}_before_update": value for key, value in angle_snapshot(adapter).items()})
    for name, parameter in adapter.named_parameters():
        if name == "log_gain":
            record["log_gain_trainable"] = parameter.requires_grad
            record["log_gain_gradient"] = None
        if name == "log_alpha":
            record["log_alpha_trainable"] = parameter.requires_grad
            record["log_alpha_before_update"] = float(parameter.detach())
            record["alpha_before_update"] = float(parameter.detach().exp())
            record["log_alpha_gradient"] = None
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise ValueError(f"frozen {name} has a stale gradient")
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise ValueError(f"invalid {name} gradient")
        record[f"{name}_gradient_l2"] = float(parameter.grad.norm())
        record[f"{name}_before_update_l2"] = float(parameter.detach().norm())
        if name in {"log_radius", "raw_mix", "log_alpha", "logit_decay", "log_gain", "history_angle"}:
            record[f"{name}_gradient"] = float(parameter.grad)
            record[f"{name}_before_update"] = float(parameter.detach())
    optimizer.step()
    if hasattr(optimizer, "last_step_diagnostics"):
        record["optimizer_step"] = optimizer.last_step_diagnostics
    angle_after_update = angle_snapshot(adapter)
    if not all(torch.isfinite(p).all() for p in adapter.parameters()):
        raise ValueError("nonfinite adapter update")
    record.update({f"{key}_after_update": value for key, value in gain_snapshot(adapter).items()})
    record.update({f"{key}_after_update": value for key, value in angle_after_update.items()})
    if hasattr(adapter, "raw_mix"):
        record["raw_mix_after_update"] = float(adapter.raw_mix.detach())
    if hasattr(adapter, "log_alpha"):
        record["log_alpha_after_update"] = float(adapter.log_alpha.detach())
        alpha = adapter.log_alpha.detach().exp()
        if not bool(torch.isfinite(alpha)) or not bool(alpha > 0):
            raise ValueError("updated alpha is not representable as positive finite f32")
        record["alpha_after_update"] = float(alpha)
    if hasattr(adapter, "logit_decay"):
        record["logit_decay_after_update"] = float(adapter.logit_decay.detach())
        decay = adapter.logit_decay.detach().sigmoid()
        if not bool(torch.isfinite(decay)) or not bool((decay > 0) & (decay < 1)):
            raise ValueError("updated decay is not representable in (0,1)")
        record["decay_after_update"] = float(decay)
    return record


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(equal_state(left[k], right[k]) for k in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(equal_state(a, b) for a, b in zip(left, right))
        )
    return left == right


def native_fixture():
    kernel = st.WaveGateKernel(curvature=-0.7, saturation=1.0, porosity=0.2)
    x, gate, bias, upstream = (
        [0.2, -0.3, 1.5, 0.5],
        [1.4, -1.1],
        [0.15, -0.05],
        [0.35, -0.2, -0.1, 0.3],
    )
    batch = kernel.forward(x, gate, bias, 2, 2)
    dx, dg, db = batch.vjp(upstream)
    radius_batch = kernel.forward_with_log_radius(x, gate, bias, 2, 2, 0.5)
    rdx, rdg, rdb, rdr = radius_batch.vjp_with_log_radius(upstream)
    return {
        "input": x,
        "gate": gate,
        "bias": bias,
        "upstream": upstream,
        "output": batch.output,
        "grad_input": dx,
        "grad_gate": dg,
        "grad_bias": db,
        "conditioning": json.loads(batch.conditioning_json()),
        "radius": {
            "log_radius": 0.5,
            "output": radius_batch.output,
            "grad_input": rdx,
            "grad_gate": rdg,
            "grad_bias": rdb,
            "grad_log_radius": rdr,
            "conditioning": json.loads(radius_batch.conditioning_json()),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    corpus_bytes = args.corpus.read_bytes()
    if digest(corpus_bytes) != config["corpus_sha256"]:
        raise ValueError("corpus hash differs from fixed protocol")
    if not args.model_dir.is_dir() or args.model_dir.name != config["model_snapshot"]:
        raise ValueError("cached model snapshot differs from protocol")
    if config["arms"] not in (LEGACY_ARMS, RADIUS_ARMS):
        raise ValueError("unrecognized arms")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(config["threads"])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True
    )
    body, body_end = corpus_body(
        corpus_bytes.decode("utf-8"), config["corpus_end_marker"]
    )
    train_text, dev_text, cut = split_text(body)
    tokenizer.model_max_length = 10**9
    train = pack_tokens(
        tokenizer.encode(train_text, add_special_tokens=False), config["block_size"]
    )
    dev = pack_tokens(
        tokenizer.encode(dev_text, add_special_tokens=False), config["block_size"]
    )
    dev_indices = spaced_indices(len(dev), config["development_blocks"])
    probe_indices = spaced_indices(len(train), config["train_probe_blocks"])
    dev_probe, train_probe = dev[dev_indices], train[probe_indices]
    schedules = {
        str(seed): schedule(seed, len(train), config["steps"] + 1, config["batch_size"])
        for seed in config["seeds"]
    }
    import spiraltorch.spiraltorch as native

    report = {
        "schema": "spiraltorch.wave_gate_conditioning_study.v1",
        "status": "running",
        "config": config,
        "config_sha256": digest(config_bytes),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "native_sha256": digest(Path(native.__file__).read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "corpus": {
            "sha256": digest(corpus_bytes),
            "bytes": len(corpus_bytes),
            "body_sha256": digest(body.encode()),
            "body_end_character_offset": body_end,
            "split_character_offset": cut,
            "train_text_sha256": digest(train_text.encode()),
            "development_text_sha256": digest(dev_text.encode()),
            "train_blocks": len(train),
            "development_blocks": len(dev),
            "train_probe_indices": probe_indices,
            "development_probe_indices": dev_indices,
            "train_tokens_sha256": digest(train.numpy().tobytes()),
            "development_tokens_sha256": digest(dev.numpy().tobytes()),
            "token_dtype": "int64_little_endian",
            "targets_per_training_arm": config["steps"]
            * config["batch_size"]
            * (config["block_size"] - 1),
        },
        "batch_schedules": schedules,
        "native_fixture": native_fixture(),
        "runs": [],
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    save_json(args.output_dir / "plan.json", report)
    model = (
        transformers.AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        .cpu()
        .eval()
        .requires_grad_(False)
    )
    model.config.use_cache = False
    if config["block_size"] > model.config.max_position_embeddings:
        raise ValueError("block size exceeds model context")
    parent_path, child_name = config["block"].rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    original = parent.get_submodule(child_name)
    baseline_hash = model_digest(model)
    baseline_dev = evaluate(model, dev_probe, config["batch_size"])
    baseline_train = evaluate(model, train_probe, config["batch_size"])
    with torch.no_grad():
        baseline_logits = model(train_probe[:1]).logits.detach()
    report.update(
        base_parameter_sha256=baseline_hash,
        baseline_development_loss=baseline_dev,
        baseline_train_probe_loss=baseline_train,
    )
    for seed in config["seeds"]:
        for arm in config["arms"]:
            adapter = adapter_for(arm, config)
            parent.add_module(child_name, torch.nn.Sequential(original, adapter))
            optimizer = torch.optim.Adam(
                adapter.parameters(), lr=config["learning_rate"]
            )
            run = {
                "seed": seed,
                "arm": arm,
                "steps": [],
                "parameter_count": sum(p.numel() for p in adapter.parameters()),
                "development_losses": [{"step": 0, "loss": baseline_dev}],
            }
            try:
                with torch.no_grad():
                    assert torch.equal(model(train_probe[:1]).logits, baseline_logits)
                for step, indices in enumerate(
                    schedules[str(seed)][: config["steps"]], start=1
                ):
                    batch = train[indices]
                    if arm == "off":
                        record = {
                            "loss": evaluate(model, batch, config["batch_size"]),
                            "conditioning": None,
                        }
                    else:
                        record = update(model, adapter, optimizer, batch)
                    record.update(step=step, batch_indices=indices)
                    run["steps"].append(record)
                    if step % config["evaluate_every"] == 0 or step == config["steps"]:
                        value = (
                            baseline_dev
                            if arm == "off"
                            else evaluate(model, dev_probe, config["batch_size"])
                        )
                        run["development_losses"].append({"step": step, "loss": value})
                        print(
                            f"{seed} {arm} step={step} development={value:.6f}",
                            flush=True,
                        )
                run["final_train_probe_loss"] = evaluate(
                    model, train_probe, config["batch_size"]
                )
                checkpoint = args.output_dir / f"{seed}-{arm}.pt"
                torch.save(
                    {
                        "adapter": adapter.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "base_parameter_sha256": baseline_hash,
                        "steps": config["steps"],
                        "next_batch_indices": schedules[str(seed)][-1],
                    },
                    checkpoint,
                )
                run["checkpoint"] = {
                    "filename": checkpoint.name,
                    "sha256": digest(checkpoint.read_bytes()),
                }
                if arm != "off":
                    restored = adapter_for(arm, config)
                    saved = torch.load(checkpoint, weights_only=True)
                    restored.load_state_dict(saved["adapter"])
                    resumed_optimizer = torch.optim.Adam(
                        restored.parameters(), lr=config["learning_rate"]
                    )
                    resumed_optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
                    next_batch = train[saved["next_batch_indices"]]
                    update(model, adapter, optimizer, next_batch)
                    parent.add_module(
                        child_name, torch.nn.Sequential(original, restored)
                    )
                    update(model, restored, resumed_optimizer, next_batch)
                    assert equal_state(adapter.state_dict(), restored.state_dict())
                    assert equal_state(
                        optimizer.state_dict(), resumed_optimizer.state_dict()
                    )
                    run["resume_next_update_equal"] = True
                    run["extra_validation_updates"] = 2
                    assert any(row["gate_gradient_l2"] > 0 for row in run["steps"])
                    assert any(row["bias_gradient_l2"] > 0 for row in run["steps"])
                    if "learnable" in arm:
                        assert run["steps"][0]["log_radius_gradient"] == 0.0
                        assert any(
                            row["log_radius_gradient_l2"] > 0
                            for row in run["steps"][1:]
                        )
            finally:
                parent.add_module(child_name, original)
            assert model_digest(model) == baseline_hash
            assert all(p.grad is None for p in model.parameters())
            run["frozen_base_unchanged"] = True
            report["runs"].append(run)
            save_json(args.output_dir / "results.json", report)
    report["status"] = "completed"
    save_json(args.output_dir / "results.json", report)


if __name__ == "__main__":
    main()

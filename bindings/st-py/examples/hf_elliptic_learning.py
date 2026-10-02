"""Local-only pretrained learning-path probe, not a language-quality benchmark.

Explicitly select a tensor-valued block. Uses a tiny authored corpus, frozen base,
and off / parameter-matched tangent-linear / geometric controls. No downloads.
The legacy filename also supports WaveGate via --geometry wave_gate.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import transformers
import spiraltorch as st


class TangentControl(st.EllipticResidualAdapter):
    def __init__(self, features, **options):
        super().__init__(features, **options)
        warp = st.EllipticWarp(**self.get_extra_state()["warp"])
        anchor = warp.map_orientations_batch([1.0, 0.0, 0.0])
        self.register_buffer("anchor", torch.tensor(anchor.features))
        rows = []
        for i in range(9):
            seed = [float(j == i) for j in range(9)]
            rows.append(anchor.vjp(seed)[1:])
        self.register_buffer("tangent", torch.tensor(rows))

    def forward(self, value):
        coordinates = self.orientation(value)
        features = self.anchor + coordinates @ self.tangent.T
        return value + self.strength * self.readout(features)


class WaveGateTangentControl(st.WaveGateAdapter):
    """First-order map at the zero gate/bias initialization, with the same 2F parameters."""

    def forward(self, value):
        curvature = self.get_extra_state()["kernel"]["curvature"]
        return value + self.strength * (value * self.gate + self.bias) / math.sqrt(
            -curvature
        )


TEXT = {
    "train": [
        "A geometric adapter changes hidden representations while the language model stays frozen.",
        "Rust computes the feature map and its derivative. Python connects that derivative to the training loss.",
    ],
    "development": [
        "Check gradients before comparing a geometric model with a simpler control.",
        "A checkpoint must preserve the learned parameters and the configuration of the computation.",
    ],
}


def weight_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--geometry", choices=("elliptic", "wave_gate"), default="elliptic"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    if not args.model_dir.is_dir() or args.steps < 2:
        parser.error("local model directory and at least two steps required")
    torch.set_num_threads(2)
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = {}
    for key, texts in TEXT.items():
        batch = tokenizer(
            texts, padding=True, return_tensors="pt", truncation=True, max_length=64
        )
        batch["labels"] = batch["input_ids"].masked_fill(
            batch["attention_mask"] == 0, -100
        )
        batches[key] = batch
    parent_path, child_name = args.block.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    original = parent.get_submodule(child_name)
    baseline_hash = weight_hash(model)
    with torch.no_grad():
        baseline_logits = model(**batches["train"]).logits
    report = {
        "schema": f"spiraltorch.hf_{args.geometry}_learning.v1",
        "scope": "bounded pretrained wiring/control experiment; tiny authored corpus",
        "model_type": model.config.model_type,
        "base_parameter_sha256": baseline_hash,
        "model_config_sha256": hashlib.sha256(
            (args.model_dir / "config.json").read_bytes()
        ).hexdigest(),
        "block": args.block,
        "features": args.features,
        "steps": args.steps,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "device": "cpu",
        "geometry_execution": "rust_f32_cpu",
        "threads": 2,
        "geometry": args.geometry,
        "seed_scope": (
            "deterministic zero initialization; seeds do not create independent data or parameter draws"
            if args.geometry == "wave_gate"
            else "trainable projection initialization"
        ),
        "texts": TEXT,
        "inputs": {
            key: {k: v.tolist() for k, v in b.items()} for key, b in batches.items()
        },
        "runs": [],
    }
    for seed in args.seeds:
        for arm in ("off", "tangent", args.geometry):
            torch.manual_seed(seed)
            if args.geometry == "wave_gate":
                constructor = (
                    WaveGateTangentControl if arm == "tangent" else st.WaveGateAdapter
                )
                gradient_fields = ("gate_grad_l1", "bias_grad_l1")
            else:
                constructor = (
                    TangentControl if arm == "tangent" else st.EllipticResidualAdapter
                )
                gradient_fields = ("orientation_grad_l1", "readout_grad_l1")
            adapter = constructor(args.features, strength=0.0 if arm == "off" else 0.1)
            parent.add_module(child_name, torch.nn.Sequential(original, adapter))
            with torch.no_grad():
                assert torch.equal(model(**batches["train"]).logits, baseline_logits)
            opt = torch.optim.Adam(adapter.parameters(), lr=1e-3)
            run = {
                "seed": seed,
                "arm": arm,
                "parameters": sum(p.numel() for p in adapter.parameters()),
                "train_loss": [],
                "development_loss": [],
                gradient_fields[0]: [],
                gradient_fields[1]: [],
            }
            try:
                for step in range(args.steps + 1):
                    with torch.no_grad():
                        run["development_loss"].append(
                            float(model(**batches["development"]).loss)
                        )
                    loss = model(**batches["train"]).loss
                    assert torch.isfinite(loss)
                    run["train_loss"].append(float(loss.detach()))
                    if step == args.steps:
                        break
                    opt.zero_grad()
                    if arm == "off":
                        for label in gradient_fields:
                            run[label].append(0.0)
                        continue
                    loss.backward()
                    parameters = (
                        (adapter.gate, adapter.bias)
                        if args.geometry == "wave_gate"
                        else (adapter.orientation.weight, adapter.readout.weight)
                    )
                    for label, parameter in zip(gradient_fields, parameters):
                        assert (
                            parameter.grad is not None
                            and torch.isfinite(parameter.grad).all()
                        )
                        run[label].append(float(parameter.grad.abs().sum()))
                    opt.step()
                with torch.no_grad():
                    run["max_logit_change"] = float(
                        (model(**batches["train"]).logits - baseline_logits).abs().max()
                    )
                if arm != "off":
                    assert all(sum(run[label]) > 0 for label in gradient_fields)
                assert all(
                    p.grad is None for p in model.parameters() if not p.requires_grad
                )
            finally:
                parent.add_module(child_name, original)
            assert weight_hash(model) == baseline_hash
            run["frozen_base_unchanged"] = True
            if args.checkpoint_dir is not None:
                args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                path = args.checkpoint_dir / f"{seed}-{arm}.pt"
                if path.exists():
                    raise FileExistsError(f"checkpoint already exists: {path.name}")
                torch.save(
                    {
                        "schema": f"spiraltorch.hf_{args.geometry}_checkpoint.v1",
                        "adapter": adapter.state_dict(),
                        "optimizer": opt.state_dict(),
                        "base_parameter_sha256": baseline_hash,
                        "block": args.block,
                        "seed": seed,
                        "arm": arm,
                        "steps": args.steps,
                    },
                    path,
                )
                run["checkpoint"] = {
                    "filename": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            report["runs"].append(run)
    if args.geometry == "wave_gate":
        kernel = st.WaveGateKernel(curvature=-0.7, saturation=1.0, porosity=0.2)
        x, gate, bias, seed = (
            [0.2, -0.3, 1.5, 0.5],
            [1.4, -1.1],
            [0.15, -0.05],
            [0.35, -0.2, -0.1, 0.3],
        )
        batch = kernel.forward(x, gate, bias, 2, 2)
        gradients = batch.vjp(seed)
        report["native_fixture"] = {
            "input": x,
            "gate": gate,
            "bias": bias,
            "upstream": seed,
            "output": batch.output,
            "grad_input": gradients[0],
            "grad_gate": gradients[1],
            "grad_bias": gradients[2],
            "conditioning": json.loads(batch.conditioning_json()),
        }
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    warp = st.EllipticWarp(1.0, 4, 2)
    x = [0.3, 0.4, 0.8, 1e-4, 2e-4, 1.0, 1e20, 2e20, 3e20]
    batch = warp.map_orientations_batch(x)
    seed = [0.4, -0.3, 0.2, 0.1, -0.1, 0.2, 0.3, -0.4, 0.5] * 3
    report["native_fixture"] = {
        "input": x,
        "upstream": seed,
        "features": batch.features,
        "vjp": batch.vjp(seed),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

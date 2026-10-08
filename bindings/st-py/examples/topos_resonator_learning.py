"""Offline wiring experiment, not a pretrained FT or quality benchmark.

Run against a built SpiralTorch package with torch and transformers installed.
All arms share a frozen, randomly initialized tiny HF language model and data.
Only the pointwise gate learns. No model or dataset is downloaded.
"""

import argparse
import hashlib
import json

import torch
import transformers
import spiraltorch as st


class LinearGate(torch.nn.Module):
    """Same parameter count and initial local gain, without the Topos rewrite."""

    def __init__(self, features, strength, gain):
        super().__init__()
        self.gate = torch.nn.Parameter(torch.zeros(features))
        self.strength = strength
        self.gain = gain

    def forward(self, x):
        return x + self.strength * self.gain * x * self.gate


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(seed, arm, steps):
    torch.manual_seed(seed)
    model = (
        transformers.GPT2LMHeadModel(
            transformers.GPT2Config(
                vocab_size=16,
                n_positions=16,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0.0,
                embd_pdrop=0.0,
                attn_pdrop=0.0,
            )
        )
        .eval()
        .requires_grad_(False)
    )
    train = torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2], [4, 5, 6, 4, 5, 6, 4, 5]])
    heldout = torch.tensor([[2, 3, 1, 2, 3, 1, 2, 3], [5, 6, 4, 5, 6, 4, 5, 6]])
    strength, coupling, iterations = 0.5, 0.35, 6
    gain = sum(coupling**i for i in range(iterations))
    adapter = (
        LinearGate(16, strength, gain)
        if arm == "linear_gate"
        else st.ToposResonatorAdapter(
            16,
            strength=0.0 if arm == "off" else strength,
            coupling=coupling,
            iterations=iterations,
            saturation=0.001,
            porosity=0.2,
        )
    )
    with torch.no_grad():
        original = model(train).logits.clone()
    model.transformer.h[0].mlp = torch.nn.Sequential(
        model.transformer.h[0].mlp, adapter
    )
    assert torch.equal(original, model(train).logits.detach())
    before = frozen_hash(model)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.05)
    train_losses, eval_losses, gradient_l1 = [], [], []
    for step in range(steps + 1):
        with torch.no_grad():
            eval_losses.append(float(model(heldout, labels=heldout).loss))
        loss = model(train, labels=train).loss
        assert torch.isfinite(loss)
        train_losses.append(float(loss.detach()))
        if step == steps:
            break
        optimizer.zero_grad()
        if arm != "off":
            loss.backward()
            assert torch.isfinite(adapter.gate.grad).all()
            gradient_l1.append(float(adapter.gate.grad.abs().sum()))
            optimizer.step()
        else:
            gradient_l1.append(0.0)
    captured = []
    hook = adapter.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach())
    )
    with torch.no_grad():
        changed = float((model(train).logits - original).abs().max())
    hook.remove()
    with torch.no_grad():
        probe = captured[0]
        nonlinear_delta = (
            adapter(probe) - (probe + strength * gain * probe * adapter.gate)
        ).abs()
    assert frozen_hash(model) == before
    if arm != "off":
        assert sum(gradient_l1) > 0 and changed > 0
    return {
        "seed": seed,
        "arm": arm,
        "base_sha256": before,
        "train_loss": train_losses,
        "development_loss": eval_losses,
        "gate_gradient_l1": gradient_l1,
        "gate": adapter.gate.detach().tolist(),
        "max_logit_change": changed,
        "max_departure_from_linear_gate": float(nonlinear_delta.max()),
        "nonlinear_values": int((nonlinear_delta > 1e-7).sum()),
        "frozen_base_unchanged": True,
        "recipe": adapter.get_extra_state() if arm != "linear_gate" else {"gain": gain},
    }


def native_fixture():
    kernel = st.ToposResonatorKernel(
        coupling=0.35, iterations=6, saturation=1.0, porosity=0.2
    )
    x, gate, upstream = (
        [0.2, -0.4, 4.0, -3.0],
        [0.3, -0.2, 0.7, 0.5],
        [0.4, -0.7, -0.3, 0.8],
    )
    dx, dg = kernel.backward(x, gate, upstream, 2, 2)
    return {
        "config": json.loads(kernel.configuration_json()),
        "input": x,
        "gate": gate,
        "upstream": upstream,
        "rows": 2,
        "features": 2,
        "output": kernel.forward(x, gate, 2, 2),
        "grad_input": dx,
        "grad_gate": dg,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    report = {
        "schema": "spiraltorch.topos_learning_probe.v1",
        "scope": "random tiny frozen HF model; synthetic periodic tokens; wiring only",
        "execution": "rust_f32_cpu_bridge",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "steps": args.steps,
        "native_fixture": native_fixture(),
        "runs": [
            run(seed, arm, args.steps)
            for seed in (17, 29, 43)
            for arm in ("off", "linear_gate", "topos")
        ],
    }
    for seed in (17, 29, 43):
        assert len({r["base_sha256"] for r in report["runs"] if r["seed"] == seed}) == 1
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

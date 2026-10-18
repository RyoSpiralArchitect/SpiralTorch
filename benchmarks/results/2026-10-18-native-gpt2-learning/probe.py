"""Three bounded offline geometry-SGD steps on an existing GPT-2 snapshot."""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

WINDOWS = [
    ("meaning", "The meaning of life is not the same as", [464, 3616, 286, 1204, 318, 407, 262, 976, 355]),
    ("models", "Machine learning models can be used to predict", [37573, 4673, 4981, 460, 307, 973, 284, 4331]),
    ("spiraltorch", "SpiralTorch is a free, open source", [50, 4063, 282, 15884, 354, 318, 257, 1479, 11, 1280, 2723]),
]
RATE = .02


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {"file": path.name, "sha256": digest(path)}


def save_artifact(path, artifact):
    # Artifact IDs pin exact bytes, not re-indented/re-serialized JSON values.
    with path.open("x") as stream:
        stream.write(artifact.to_json())
    return {"file": path.name, "sha256": digest(path)}


@contextmanager
def attach_reference(stack, model):
    handles, seen = [], set()
    try:
        for path, adapter in zip(stack.paths, stack.adapters):
            def hook(_module, _inputs, output, path=path, adapter=adapter):
                seen.add(path)
                value = output[0] if isinstance(output, tuple) else output
                changed = adapter(value)
                return (changed,) + output[1:] if isinstance(output, tuple) else changed
            handles.append(model.get_submodule(path).register_forward_hook(hook))
        yield
        if seen != set(stack.paths):
            raise ValueError("not all reference geometry placements executed")
    finally:
        for handle in handles:
            handle.remove()


def initial_artifact(st, model):
    width = json.loads(model.config_json())["width"]

    def parameter(name, shape, values):
        return {"name": name, "shape": shape, "values": values, "role": "parameter", "requires_grad": True}

    gate = [((i % 7) - 3) * .002 for i in range(width)]
    sites = [
        ("transformer.h.0.mlp", {
            "schema": "spiraltorch.wave_gate_adapter.v2", "features": width, "strength": .1, "learnable_radius": True,
            "kernel": {"curvature": -1., "saturation": 1., "porosity": .05, "max_values": 65536},
        }, [parameter("gate", [width], gate), parameter("bias", [width], [0.] * width), parameter("log_radius", [], [.2])]),
        ("transformer.h.6.attn", {
            "schema": "spiraltorch.fractional_angle_gain_history_adapter.v1", "features": width, "strength": .1,
            "lag_window": [1, 4], "kernel": {"kernel_len": 5, "step": .8, "max_values": 65536, "max_products": 327680},
        }, [parameter("gate", [width], gate), parameter("local_gate", [width], [.01] * width),
            parameter("history_angle", [], [-.2]), parameter("log_gain", [], [.1])]),
        ("transformer.h.11.mlp", {
            "schema": "spiraltorch.topos_resonator_adapter.v1", "features": width, "strength": .1,
            "kernel": {"coupling": .2, "iterations": 3, "saturation": 1., "porosity": .05, "max_values": 65536},
        }, [parameter("gate", [width], gate)]),
    ]
    return st.GeometryAdapterArtifact.create_json(json.dumps({
        "schema": "spiraltorch.geometry_adapter_artifact.v1", "base_model_identity": model.base_model_identity,
        "policy": "full_prefix_f32_no_cache.v1", "training": False,
        "placements": [{"path": path, "recipe": recipe, "tensors": tensors, "training": False} for path, recipe, tensors in sites],
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["native", "reference"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-report", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = {name: digest(args.model_dir / name) for name in ("config.json", "model.safetensors", "tokenizer.json")}
    rows = []
    if args.mode == "native":
        class NoForeignML(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"torch", "numpy", "transformers", "accelerate", "safetensors"}:
                    raise AssertionError("native learning attempted foreign ML import: " + fullname)
        sys.meta_path.insert(0, NoForeignML())
        import spiraltorch as st
        archive = st.SafeTensorArchive(args.model_dir / "model.safetensors")
        base = st.NativeCausalLm.from_gpt2((args.model_dir / "config.json").read_text(), archive)
        del archive
        initial = initial_artifact(st, base)
        save_artifact(args.output / "initial.json", initial)
        model = base.with_geometry(initial)
        ids = WINDOWS[0][2]
        before = bytes(base.forward(ids, [1, len(ids)], last_only=True).values_buffer())
        for step, (name, text, ids) in enumerate(WINDOWS):
            gradient = model.geometry_loss_gradients(ids, [1, len(ids)])
            parameters = save(args.output / f"gradient-{step}.json", gradient.parameter_gradients())
            model = model.step_geometry_sgd(gradient, RATE)
            state = save_artifact(args.output / f"state-{step}.json", model.geometry_artifact())
            rows.append({"step": step, "case": name, "text": text, "ids": ids, "pre_nll": gradient.loss,
                         "post_nll": model.next_token_loss(ids, [1, len(ids)]), "gradient": parameters, "state": state})
            print(json.dumps({"mode": args.mode, "step": step, "pre_nll": rows[-1]["pre_nll"], "post_nll": rows[-1]["post_nll"]}), flush=True)
        saved = model.geometry_artifact()
        restored = base.with_geometry(st.GeometryAdapterArtifact.from_json(saved.to_json(), saved.artifact_id, base.base_model_identity))
        ids = WINDOWS[0][2]
        a, b = model.geometry_loss_gradients(ids, [1, len(ids)]), restored.geometry_loss_gradients(ids, [1, len(ids)])
        exact = (a.loss == b.loss and a.parameter_gradients() == b.parameter_gradients()
                 and model.step_geometry_sgd(a, RATE).geometry_artifact().to_json() == restored.step_geometry_sgd(b, RATE).geometry_artifact().to_json())
        if not exact:
            raise ValueError("native artifact continuation changed")
        after = bytes(base.forward(ids, [1, len(ids)], last_only=True).values_buffer())
        if before != after:
            raise ValueError("frozen base probe changed")
        extra = {"base_model_identity": base.base_model_identity, "initial_geometry_artifact_id": initial.artifact_id,
                 "optional_ml_imports_forbidden": True, "artifact_continuation_exact": exact,
                 "base_probe_logits_sha256": hashlib.sha256(before).hexdigest(), "base_probe_unchanged": True}
    else:
        if args.native_report is None:
            parser.error("reference needs --native-report")
        native = json.loads(args.native_report.read_text())
        if native["source_sha256"] != source or native["learning_rate"] != RATE:
            raise ValueError("reference source or learning rate differ")
        import torch
        import transformers
        from tokenizers import Tokenizer
        import spiraltorch as st
        torch.set_num_threads(2)
        reference = transformers.GPT2LMHeadModel.from_pretrained(
            args.model_dir, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
        reference.config.use_cache = False
        tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
        state = st.GeometryAdapterArtifact.from_json((args.native_report.parent / "initial.json").read_text(),
                                                    native["initial_geometry_artifact_id"], native["base_model_identity"])
        save_artifact(args.output / "initial.json", state)
        stack = st.GeometryAdapterStack.from_artifact(state, expected_base_model_identity=state.base_model_identity)
        optimizer = torch.optim.SGD(stack.parameters(), lr=RATE)
        versions = [p._version for p in reference.parameters()]
        for step, (name, text, ids) in enumerate(WINDOWS):
            assert tokenizer.encode(text).ids == ids
            assert native["rows"][step]["ids"] == ids and native["rows"][step]["step"] == step
            tokens = torch.tensor([ids])
            optimizer.zero_grad(set_to_none=True)
            with attach_reference(stack, reference):
                # Identical truncated prefix and labels, rather than evaluating an unused final row.
                logits = reference(tokens[:, :-1], use_cache=False).logits
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
                loss.backward()
            gradients = {path: {name: p.grad.detach().flatten().tolist() for name, p in adapter.named_parameters()}
                         for path, adapter in zip(stack.paths, stack.adapters)}
            parameters = save(args.output / f"gradient-{step}.json", gradients)
            optimizer.step()
            with torch.no_grad(), attach_reference(stack, reference):
                logits = reference(tokens[:, :-1], use_cache=False).logits
                post = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
            saved = stack.export_artifact(base_model_identity=state.base_model_identity)
            receipt = save_artifact(args.output / f"state-{step}.json", saved)
            rows.append({"step": step, "case": name, "text": text, "ids": ids, "pre_nll": float(loss.detach()),
                         "post_nll": float(post), "gradient": parameters, "state": receipt})
            print(json.dumps({"mode": args.mode, "step": step, "pre_nll": rows[-1]["pre_nll"], "post_nll": rows[-1]["post_nll"]}), flush=True)
        if versions != [p._version for p in reference.parameters()] or any(p.grad is not None for p in reference.parameters()):
            raise ValueError("reference base parameters changed or received gradients")
        extra = {"base_model_identity": state.base_model_identity, "initial_geometry_artifact_id": state.artifact_id,
                 "torch": torch.__version__, "transformers": transformers.__version__, "base_parameter_versions_unchanged": True}
    save(args.output / "report.json", {"schema": "spiraltorch.native_gpt2_learning_probe.v1", "mode": args.mode,
                                      "scope": "three_window_gradient_and_update_parity_not_language_quality",
                                      "source_sha256": source, "learning_rate": RATE, "rows": rows, **extra})


if __name__ == "__main__":
    main()

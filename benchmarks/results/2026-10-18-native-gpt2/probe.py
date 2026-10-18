"""Bounded offline native/reference GPT-2 inference probe; no model downloads."""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys

CASES = [
    ("meaning", "The meaning of life is", [464, 3616, 286, 1204, 318]),
    ("models", "Machine learning models can", [37573, 4673, 4981, 460]),
    ("spiraltorch", "SpiralTorch is a", [50, 4063, 282, 15884, 354, 318, 257]),
]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wave(st, model):
    width = json.loads(model.config_json())["width"]
    return st.GeometryAdapterArtifact.create_json(json.dumps({
        "schema": "spiraltorch.geometry_adapter_artifact.v1", "base_model_identity": model.base_model_identity,
        "policy": "full_prefix_f32_no_cache.v1", "training": False, "placements": [{
            "path": "transformer.h.0.mlp", "training": False,
            "recipe": {"schema": "spiraltorch.wave_gate_adapter.v1", "features": width, "strength": .1,
                       "kernel": {"curvature": -1., "saturation": 1., "porosity": .05, "max_values": 65536}},
            "tensors": [{"name": name, "shape": [width], "values": values, "requires_grad": False, "role": "parameter"}
                        for name, values in [("gate", [((i % 7) - 3) * .02 for i in range(width)]),
                                             ("bias", [((i % 5) - 2) * .01 for i in range(width)])]],
        }],
    }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["native", "reference"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-report", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.steps <= 8:
        parser.error("steps must be 0..8")
    args.output.mkdir(parents=True, exist_ok=False)
    source = {name: digest(args.model_dir / name) for name in ("config.json", "model.safetensors", "tokenizer.json")}
    rows = []
    if args.mode == "native":
        class NoForeignML(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"torch", "numpy", "transformers", "accelerate", "safetensors"}:
                    raise AssertionError("native execution attempted optional ML import: " + fullname)
        sys.meta_path.insert(0, NoForeignML())
        import spiraltorch as st
        weights = st.SafeTensorArchive(args.model_dir / "model.safetensors")
        model = st.NativeCausalLm.from_gpt2((args.model_dir / "config.json").read_text(), weights)
        del weights
        artifact = wave(st, model)
        with (args.output / "geometry.json").open("x") as stream:
            stream.write(artifact.to_json())
        for arm, runtime in [("base", model), ("wave", model.with_geometry(artifact))]:
            for name, prompt, ids in CASES:
                result = runtime.forward(ids, [1, len(ids)])
                path = args.output / (name + "-" + arm + ".f32")
                with path.open("xb") as stream:
                    stream.write(result.values_buffer())
                generated = list(ids)
                for _ in range(args.steps):
                    generated.extend(runtime.next_token_ids(generated, [1, len(generated)]))
                rows.append({"case": name, "prompt": prompt, "arm": arm, "ids": ids, "generated_ids": generated,
                             "nll": runtime.next_token_loss(ids, [1, len(ids)]), "shape": result.shape,
                             "logits_file": path.name, "logits_sha256": digest(path)})
        extra = {"base_model_identity": model.base_model_identity, "geometry_artifact_id": artifact.artifact_id,
                 "backend": model.execution_backend, "optional_ml_imports_forbidden": True}
    else:
        if args.native_report is None:
            parser.error("reference needs --native-report")
        native = json.loads(args.native_report.read_text())
        if native["source_sha256"] != source or native["steps"] != args.steps:
            raise ValueError("reference source or steps differ from native probe")
        import torch
        import transformers
        from tokenizers import Tokenizer
        import spiraltorch as st
        torch.set_num_threads(2)
        model = transformers.GPT2LMHeadModel.from_pretrained(
            args.model_dir, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval()
        tokenizer = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
        state = st.GeometryAdapterArtifact.from_json(
            (args.native_report.parent / "geometry.json").read_text(), native["geometry_artifact_id"], native["base_model_identity"])
        placement = json.loads(state.state_json())["placements"][0]
        recipe = placement["recipe"]
        adapter = st.WaveGateAdapter(recipe["features"], strength=recipe["strength"], **recipe["kernel"]).eval()
        with torch.no_grad():
            for parameter in placement["tensors"]:
                getattr(adapter, parameter["name"]).copy_(torch.tensor(parameter["values"], dtype=torch.float32))
        for arm in ("base", "wave"):
            handle = (model.transformer.h[0].mlp.register_forward_hook(lambda _m, _x, output: adapter(output))
                      if arm == "wave" else None)
            try:
                with torch.no_grad():
                    for name, prompt, ids in CASES:
                        assert tokenizer.encode(prompt).ids == ids
                        tokens = torch.tensor([ids])
                        result = model(tokens, labels=tokens, use_cache=False)
                        path = args.output / (name + "-" + arm + ".f32")
                        with path.open("xb") as stream:
                            stream.write(result.logits.contiguous().numpy().tobytes())
                        generated = list(ids)
                        for _ in range(args.steps):
                            last = model(torch.tensor([generated]), use_cache=False).logits[0, -1]
                            generated.append(int(last.argmax()))
                        rows.append({"case": name, "prompt": prompt, "arm": arm, "ids": ids, "generated_ids": generated,
                                     "generated_text": tokenizer.decode(generated), "nll": float(result.loss),
                                     "shape": list(result.logits.shape), "logits_file": path.name, "logits_sha256": digest(path)})
            finally:
                if handle is not None:
                    handle.remove()
        extra = {"torch": torch.__version__, "transformers": transformers.__version__, "backend": "torch_cpu_eager",
                 "base_model_identity": native["base_model_identity"], "geometry_artifact_id": state.artifact_id}
    report = {"schema": "spiraltorch.native_gpt2_probe.v1", "mode": args.mode, "source_sha256": source,
              "steps": args.steps, "byteorder": sys.byteorder, "rows": rows,
              "scope": "offline_forward_compatibility_and_intervention_not_training_or_quality", **extra}
    with (args.output / "report.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"mode": args.mode, "cases": len(rows), "report": str(args.output / "report.json")}))


if __name__ == "__main__":
    main()

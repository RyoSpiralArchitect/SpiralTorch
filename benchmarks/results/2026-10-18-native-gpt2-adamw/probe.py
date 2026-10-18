"""Bounded offline GPT-2 AdamW continuation probe, not a quality/speed benchmark."""
import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import runpy
import sys

FIXTURE = Path(__file__).resolve().parents[1] / "2026-10-18-native-gpt2-learning/probe.py"
POLICY = {"learning_rate": .001, "weight_decay": .01, "max_gradient_norm": .1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    class NoForeignML(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {"torch", "numpy", "transformers", "accelerate", "safetensors"}:
                raise AssertionError("native AdamW attempted foreign ML import: " + fullname)

    sys.meta_path.insert(0, NoForeignML())
    import spiraltorch as st

    helpers = runpy.run_path(str(FIXTURE))
    save, save_artifact, digest = (helpers[key] for key in ("save", "save_artifact", "digest"))
    windows = helpers["WINDOWS"]
    source = {name: digest(args.model_dir / name) for name in ("config.json", "model.safetensors", "tokenizer.json")}
    archive = st.SafeTensorArchive(args.model_dir / "model.safetensors")
    base = st.NativeCausalLm.from_gpt2((args.model_dir / "config.json").read_text(), archive)
    del archive
    initial = helpers["initial_artifact"](st, base)
    save_artifact(args.output / "initial.json", initial)
    model = base.with_geometry(initial)
    optimizer = st.NativeGeometryAdamW(initial, max_gradient_norm=POLICY["max_gradient_norm"])
    ids = windows[0][2]
    frozen = bytes(base.forward(ids, [1, len(ids)], last_only=True).values_buffer())

    def update(model, optimizer, gradients):
        return model.step_geometry_adamw(
            gradients, optimizer, POLICY["learning_rate"], weight_decay=POLICY["weight_decay"]
        )

    rows = []
    for step, (name, text, ids) in enumerate(windows):
        gradients = model.geometry_loss_gradients(ids, [1, len(ids)])
        model, optimizer = update(model, optimizer, gradients)
        geometry = model.geometry_artifact()
        saved = save_artifact(args.output / f"geometry-{step}.json", geometry)
        # Preserve exact optimizer bytes privately; do not publish learned values.
        with (args.output / f"optimizer-{step}.json").open("x") as stream:
            stream.write(optimizer.to_json())
        rows.append({"step": step, "case": name, "text": text, "ids": ids,
                     "pre_nll": gradients.loss, "post_nll": model.next_token_loss(ids, [1, len(ids)]),
                     "geometry": saved, "optimizer_checkpoint_id": optimizer.checkpoint_id,
                     "optimizer_revision": optimizer.revision})
        print(json.dumps(rows[-1]), flush=True)

    geometry = model.geometry_artifact()
    restored_geometry = st.GeometryAdapterArtifact.from_json(geometry.to_json(), geometry.artifact_id, base.base_model_identity)
    restored = base.with_geometry(restored_geometry)
    restored_optimizer = st.NativeGeometryAdamW.from_json(optimizer.to_json(), optimizer.checkpoint_id, restored_geometry)
    ids = windows[0][2]
    a, b = model.geometry_loss_gradients(ids, [1, len(ids)]), restored.geometry_loss_gradients(ids, [1, len(ids)])
    next_a, opt_a = update(model, optimizer, a)
    next_b, opt_b = update(restored, restored_optimizer, b)
    exact = (a.loss == b.loss and a.parameter_gradients() == b.parameter_gradients()
             and next_a.geometry_artifact().to_json() == next_b.geometry_artifact().to_json()
             and opt_a.to_json() == opt_b.to_json())
    if not exact or bytes(base.forward(ids, [1, len(ids)], last_only=True).values_buffer()) != frozen:
        raise ValueError("pair continuation or frozen base probe changed")
    save(args.output / "report.json", {
        "schema": "spiraltorch.native_gpt2_adamw_probe.v1",
        "scope": "three_windows_native_update_and_exact_resume_not_quality_or_speed",
        "source_sha256": source, "fixture_sha256": digest(FIXTURE), "probe_sha256": digest(Path(__file__)),
        "policy": POLICY, "base_model_identity": base.base_model_identity,
        "initial_geometry_artifact_id": initial.artifact_id, "rows": rows,
        "optional_ml_imports_forbidden": True, "pair_continuation_exact": exact,
        "base_probe_unchanged": True, "base_probe_logits_sha256": hashlib.sha256(frozen).hexdigest(),
    })


if __name__ == "__main__":
    main()

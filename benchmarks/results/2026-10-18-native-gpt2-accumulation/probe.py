"""Offline GPT-2 ragged-microbatch updates and boundary resume, not quality/speed."""
import argparse
import hashlib
import importlib.abc
import json
import math
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
                raise AssertionError("native accumulation attempted foreign ML import: " + fullname)

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
    probe_ids = windows[0][2]
    frozen = bytes(base.forward(probe_ids, [1, len(probe_ids)], last_only=True).values_buffer())
    targets = sum(len(ids) - 1 for _, _, ids in windows)

    def accumulate(model):
        accumulator, losses = model.geometry_accumulator(), []
        for name, text, ids in windows:
            loss = accumulator.accumulate_loss(model, ids, [1, len(ids)])
            losses.append({"case": name, "text": text, "ids": ids, "targets": len(ids) - 1, "pre_nll": loss})
        expected = math.fsum(row["pre_nll"] * row["targets"] for row in losses) / targets
        if (accumulator.microbatches != len(windows) or accumulator.target_tokens != targets
                or not math.isclose(accumulator.loss, expected, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("target-weighted native loss accounting changed")
        return accumulator, losses

    def update(model, optimizer, accumulator):
        return accumulator.step_adamw(
            model, optimizer, POLICY["learning_rate"], weight_decay=POLICY["weight_decay"]
        )

    rows = []
    for step in range(3):
        accumulator, cases = accumulate(model)
        gradient = save(args.output / f"gradient-{step}.json", accumulator.parameter_gradients())
        model, optimizer = update(model, optimizer, accumulator)
        if optimizer.revision != step + 1 or {s["steps"] for s in json.loads(optimizer.to_json())["slots"]} != {step + 1}:
            raise ValueError("optimizer must advance once per group, not once per microbatch")
        for row in cases:
            row["post_nll"] = model.next_token_loss(row["ids"], [1, len(row["ids"])])
        geometry = save_artifact(args.output / f"geometry-{step}.json", model.geometry_artifact())
        with (args.output / f"optimizer-{step}.json").open("x") as stream:
            stream.write(optimizer.to_json())
        rows.append({"step": step, "microbatches": accumulator.microbatches, "target_tokens": targets,
                     "pre_nll": accumulator.loss, "post_nll": math.fsum(r["post_nll"] * r["targets"] for r in cases) / targets,
                     "cases": cases, "gradient": gradient, "geometry": geometry,
                     "optimizer_checkpoint_id": optimizer.checkpoint_id, "optimizer_revision": optimizer.revision})
        print(json.dumps(rows[-1]), flush=True)

    geometry = model.geometry_artifact()
    restored_geometry = st.GeometryAdapterArtifact.from_json(geometry.to_json(), geometry.artifact_id, base.base_model_identity)
    restored = base.with_geometry(restored_geometry)
    restored_optimizer = st.NativeGeometryAdamW.from_json(optimizer.to_json(), optimizer.checkpoint_id, restored_geometry)
    a, _ = accumulate(model)
    b, _ = accumulate(restored)
    next_a, opt_a = update(model, optimizer, a)
    next_b, opt_b = update(restored, restored_optimizer, b)
    exact = (a.loss == b.loss and a.parameter_gradients() == b.parameter_gradients()
             and next_a.geometry_artifact().to_json() == next_b.geometry_artifact().to_json()
             and opt_a.to_json() == opt_b.to_json())
    if not exact or bytes(base.forward(probe_ids, [1, len(probe_ids)], last_only=True).values_buffer()) != frozen:
        raise ValueError("next-group continuation or frozen base probe changed")
    save(args.output / "report.json", {
        "schema": "spiraltorch.native_gpt2_accumulation_probe.v1",
        "scope": "three_ragged_groups_native_update_and_boundary_resume_not_quality_or_speed",
        "source_sha256": source, "fixture_sha256": digest(FIXTURE), "probe_sha256": digest(Path(__file__)),
        "policy": POLICY, "base_model_identity": base.base_model_identity,
        "initial_geometry_artifact_id": initial.artifact_id, "rows": rows,
        "optional_ml_imports_forbidden": True, "next_group_continuation_exact": exact,
        "base_probe_unchanged": True, "base_probe_logits_sha256": hashlib.sha256(frozen).hexdigest(),
    })


if __name__ == "__main__":
    main()

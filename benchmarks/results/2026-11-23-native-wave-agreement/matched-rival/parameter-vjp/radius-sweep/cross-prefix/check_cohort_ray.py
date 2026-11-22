"""Two two-member steps per radius on synthetic weights only; independent coordinate oracle."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3] / "radius-transfer"
E = runpy.run_path(str(ROOT / "experiment.py"))
TOY = "e7a486aadc12503cbaa6828aa9ced4ea575900c51de470a1f0abf9c1ae2ab636"
need, f32 = E["need"], E["f32"]


def norm(v):
    return math.sqrt(math.fsum(x * x for x in v))


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def close(actual, expected):
    need(math.isclose(actual, expected, abs_tol=1e-25, rel_tol=1e-12), "receipt mismatch")


def check(plan, out, extension_sha):
    raw = plan.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == TOY, "only the synthetic fixture is allowed")
    p = json.loads(raw)
    need(p["kind"] == "rehearsal" and p["shape"] == [2, 16, 2, 32], "wrong fixture")
    out.mkdir(mode=0o700, exist_ok=False)
    os.environ.update(E["OLD"]["ENV"], SPIRALTORCH_AUTOTUNE_STORE=str(out / "tuning.json"))
    foreign = ("torch", "numpy", "transformers", "safetensors")
    for k in foreign:
        need(k not in sys.modules, "unexpected imported ML runtime")
        sys.modules[k] = None
    import spiraltorch as st
    ext = Path(st._rs.__file__)
    need(hashlib.sha256(ext.read_bytes()).hexdigest() == extension_sha, "unexpected extension")
    base = E["N"]["tiny"](st).with_forward_readout_storage("original_packed")
    base = base.with_readout_accumulator("f32_wide_backward")
    need(base.base_model_identity == p["base"] and base.execution_backend == "rust_f32_cpu_prepacked",
         "unexpected base")
    with (out / "rows.jsonl").open("x", buffering=1) as log:
        for arm in ("r1", "r4"):
            raw = base64.b64decode(p["initial"][arm], validate=True)
            model = base.with_geometry(st.GeometryAdapterArtifact.from_json(raw.decode(), p["pins"][arm], p["base"]))
            for keys, same in [((0, 2), False), ((0, 1), True)]:
                cases = [p["train"][i] for i in keys]
                need((cases[0]["gold"] == cases[1]["gold"]) == same, "wrong cohort")
                handles = [E["FIT"]["gradient"](model, row, "rival")[0] for row in cases]
                gradients = [h.parameter_gradients()[p["site"]] for h in handles]
                vectors = [g["gate"] + g["bias"] for g in gradients]
                mean = [math.fsum(col) / len(vectors) for col in zip(*vectors)]
                length, distance = norm(mean), p["target_l2"]
                before, layout = E["weights"](model.geometry_artifact().to_json().encode(), p, arm)
                nxt, r = model.step_geometry_cohort_gradient_at_distance(handles, distance)
                raw = nxt.geometry_artifact().to_json().encode()
                after, new_layout = E["weights"](raw, p, arm)
                expected = f32([x + distance * (-g / length) for x, g in zip(before, mean)])
                need(len(after) == len(mean) == 32 and after.tobytes() == expected.tobytes(), "wrong f32 coordinates")
                step = r["step"]
                need(layout == new_layout and model.lora_artifact_id is None and nxt.lora_artifact_id is None
                     and nxt.base_model_identity == r["base_model_identity"] == p["base"]
                     and r["members"] == r["total_target_tokens"] == len(vectors) == len(r["member_loss_dots"])
                     and step["method"] == "cohort_mean_gradient_ray_f64.v1" and step["trainable_values"] == len(mean)
                     and step["origin_id"] == model.geometry_artifact_id and step["result_id"] == nxt.geometry_artifact_id
                     and step["result_id"] == "sha256:" + hashlib.sha256(raw).hexdigest(), "bad lineage")
                delta = [b - a for a, b in zip(before, after)]
                close(step["gradient_l2"], length)
                close(step["realized_l2"], norm(delta))
                close(step["requested_l2"], distance)
                close(step["gradient_dot_step"], dot(mean, delta))
                close(r["mean_input_loss"], math.fsum(h.loss for h in handles) / len(handles))
                for v, reported in zip(vectors, r["member_loss_dots"]):
                    close(reported, dot(v, delta))
                stage = arm + ("-same" if same else "-opposite")
                (out / (stage + ".json")).write_bytes(raw)
                log.write(json.dumps(dict(stage=stage, keys=[c["key"] for c in cases], receipt=r,
                                          mean_to_mean_member_norm=length/(math.fsum(map(norm,vectors))/len(vectors))),
                                     sort_keys=True, allow_nan=False) + "\n")
                model = nxt
    need(all(sys.modules.get(k) is None for k in foreign), "unexpected foreign runtime")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("toy_plan", type=Path)
    parser.add_argument("new_output", type=Path)
    parser.add_argument("extension_sha256")
    args = parser.parse_args()
    check(args.toy_plan, args.new_output, args.extension_sha256)

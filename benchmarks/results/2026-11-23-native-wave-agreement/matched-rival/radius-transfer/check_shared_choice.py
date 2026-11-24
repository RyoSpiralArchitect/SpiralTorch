"""Exercise shared full/declared geometry VJPs on the sealed artificial model only."""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
STEP = runpy.run_path(str(HERE / "check_source_step.py"))
E, PLAN = STEP["E"], STEP["PLAN"]
need, sha, f32 = (E[k] for k in ("need", "sha", "f32"))


def scores(model, case):
    xs = case["prefix"]
    row = model.forward(xs, [1, len(xs)], last_only=True).values()
    top = max(row)
    gold = row[case["gold"]]
    return [math.log1p(math.exp(row[case["rival"]] - gold)),
            (top - gold) + math.log(math.fsum(math.exp(x - top) for x in row))]


def compare(toy, ext, output):
    raw = (toy / "plan.json").read_bytes()
    need(sha(raw) == PLAN, "only the pinned artificial model is admitted")
    p = json.loads(raw)
    need(p["kind"] == "rehearsal" and p["model_pins"] is None and p["shape"] == [2, 16, 2, 32]
         and p["sources"] == E["sources"]() and len(p["train"]) == 4, "changed synthetic study")
    output.mkdir(mode=0o700, exist_ok=False)
    os.environ.update(E["OLD"]["ENV"], SPIRALTORCH_AUTOTUNE_STORE=str(output / "tuning.json"))
    for name in ("torch", "numpy", "safetensors", "transformers"):
        need(name not in sys.modules, "not an isolated native process")
        sys.modules[name] = None
    import spiraltorch as st
    need(E["P"]["file_hash"](Path(st._rs.__file__)) == ext, "wrong installed CPU extension")
    base = E["N"]["tiny"](st).with_readout_accumulator("f32_wide_backward").with_forward_readout_storage("original_packed")
    need(base.base_model_identity == p["base"] and base.execution_backend == "rust_f32_cpu_prepacked",
         "unexpected artificial base")
    count = repaired = improved = 0
    journal = output / "checks.jsonl"
    with journal.open("xb", buffering=0) as sink:
        for _, order, arm, _ in E["stages"](p):
            state = base64.b64decode(p["initial"][arm], validate=True)
            model = base.with_geometry(st.GeometryAdapterArtifact.from_json(state.decode(), p["pins"][arm], p["base"]))
            for i in order:
                case = p["train"][i]
                xs = case["prefix"] + [case["gold"]]
                labels = [-100] * (len(xs)-1) + [case["gold"]]
                full, choice = model.geometry_full_and_declared_gradients(xs, [1,len(xs)], labels=labels,
                    contrasts=[(len(xs)-1, case["gold"], [case["rival"]])], mix=1.)
                raw_gradients, vectors, refs = {}, [], []
                for objective, g in (("rival", choice), ("full", full)):
                    ref, report = E["FIT"]["gradient"](model, case, objective)
                    def vector(handle):
                        params = handle.parameter_gradients()[p["site"]]
                        return f32(params["gate"] + params["bias"])
                    v = vector(g)
                    need(v.tobytes() == vector(ref).tobytes() and g.loss == ref.loss and g.declared_rival_report() == report
                         and g.geometry_artifact_id == ref.geometry_artifact_id == model.geometry_artifact_id
                         and g.target_tokens == ref.target_tokens == 1, "independent gradient or receipt differs")
                    raw_gradients[objective] = sha(v.tobytes())
                    vectors.append(v)
                    refs.append(ref)
                def guarded(c, f):
                    proposal, ray = model.step_geometry_gradient_at_distance(c, p["target_l2"])
                    end, guard = model.guard_geometry_step(proposal, [[c], [f]])
                    return end, ray, guard
                end, ray, guard = guarded(choice, full)
                reference, r_ray, r_guard = guarded(*refs)
                new = end.geometry_artifact().to_json().encode()
                need((ray, guard) == (r_ray, r_guard) and reference.geometry_artifact().to_json().encode() == new
                     and end.base_model_identity == model.base_model_identity and end.lora_artifact_id is None,
                     "guarded state or receipt differs")
                before, layout = E["weights"](state, p, arm)
                after, new_layout = E["weights"](new, p, arm)
                need(new_layout == layout, "changed frozen buffers")
                dots = [math.fsum(g * (y-x) for g,x,y in zip(v, before, after)) for v in vectors]
                need(all(a < 0. and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-16)
                         for a, b in zip(dots, guard["applied_loss_dots"])), "rounded displacement is not shared descent")
                initial, final = scores(model, case), scores(end, case)
                need(all(abs(x-y) < 2e-6 for x,y in zip(initial, [choice.loss, full.loss])), "wrong logits/loss")
                delta = [y-x for x,y in zip(initial, final)]
                sink.write(E["enc"](dict(arm=arm, key=case["key"], origin=model.geometry_artifact_id,
                           end=end.geometry_artifact_id, gradients=raw_gradients, ray=ray, guard=guard,
                           dot_choice_full=dots, finite_delta_choice_full=delta)))
                state, model = new, end
                repaired += guard["mix"] > 0.
                improved += all(d < 0. for d in delta)
                count += 1
    need(count == 8, "incomplete artificial trajectories")
    result = dict(reference_plan=PLAN, extension=ext, shared_updates=count, independent_reference_steps=count,
                  exact_gradient_handles=2*count, guarded_repairs=repaired, finite_both_lower=improved,
                  training_rows_per_arm=4, pretrained_calls=0, held_out_calls=0, checks=E["P"]["file_hash"](journal))
    E["save"](output / "result.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("toy", type=Path)
    ap.add_argument("extension")
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    compare(args.toy, args.extension, args.output)

"""Check the source-built gradient step against an existing, artificial training trajectory."""
import argparse
import base64
import json
import os
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent
E = runpy.run_path(str(HERE / "experiment.py"))
need, sha = E["need"], E["sha"]
PLAN = "e7a486aadc12503cbaa6828aa9ced4ea575900c51de470a1f0abf9c1ae2ab636"
RESULT = "53c7bc15999a9eddea11b5329f957c9b9f24d54b11f7491d50f181f36a562c58"


def compare(toy, expected_extension, output):
    need(output.is_dir() and not (output / "checks.jsonl").exists(), "use a fresh output directory")
    raw = (toy / "plan.json").read_bytes()
    need(sha(raw) == PLAN, "only the pinned artificial rehearsal is accepted")
    p = json.loads(raw)
    root = toy / "attempt"
    raw = (root / "result.json").read_bytes()
    need(sha(raw) == RESULT, "wrong saved result")
    result = json.loads(raw)
    need(p["kind"] == "rehearsal" and p["shape"] == [2, 16, 2, 32] and p["updates"] == 8
         and p["model_pins"] is None and result["plan"] == PLAN, "not the artificial decoder")
    need(p["sources"] == E["sources"](), "training protocol changed")
    os.environ.update(E["OLD"]["ENV"], SPIRALTORCH_AUTOTUNE_STORE=str(output / "tuning.json"))
    for name in ("torch", "numpy", "safetensors", "transformers"):
        need(name not in sys.modules, "unexpected ML runtime")
        sys.modules[name] = None
    import spiraltorch as st
    ext = E["P"]["file_hash"](Path(st._rs.__file__))
    need(ext == expected_extension, "wrong installed CPU extension")
    base = E["N"]["tiny"](st).with_readout_accumulator("f32_wide_backward")
    need(base.base_model_identity == p["base"] and base.execution_backend == "rust_f32_cpu_prepacked"
         and base.geometry_artifact_id is None and base.lora_artifact_id is None
         and (base.readout_forward_precision, base.readout_backward_precision) == ("f32", "f64"),
         "unexpected artificial base")
    count = 0
    with (output / "checks.jsonl").open("xb", buffering=0) as sink:
        for _, order, arm, stage in E["stages"](p):
            raw = base64.b64decode(p["initial"][arm], validate=True)
            model = base.with_geometry(st.GeometryAdapterArtifact.from_json(raw.decode(), p["pins"][arm], p["base"]))
            journal = (root / (stage + ".jsonl")).read_bytes()
            tape = (root / (stage + ".g32")).read_bytes()
            refs = [json.loads(r) for r in journal.splitlines()]
            need(sha(journal) == result["paths"][stage]["journal_sha256"] and
                 sha(tape) == result["paths"][stage]["tape_sha256"] and
                 len(refs) == len(order) == 4 and len(tape) == 4 * 128, "changed artificial trajectory")
            for j, (idx, ref) in enumerate(zip(order, refs)):
                case = p["train"][idx]
                need(case["key"] == ref["key"] and ref["step"]["origin_id"] == model.geometry_artifact_id,
                     "wrong artificial row or predecessor")
                g, declared = E["FIT"]["gradient"](model, case, "rival")
                params = g.parameter_gradients()[p["site"]]
                raw_grad = E["f32"](params["gate"] + params["bias"]).tobytes()
                need(raw_grad == tape[128*j:128*(j+1)] and sha(raw_grad) == ref["grad_sha256"]
                     and g.loss == ref["loss"] and declared == ref["declared"], "changed incoming gradient or objective")
                model, trace = model.step_geometry_gradient_at_distance(g, p["target_l2"])
                artifact = model.geometry_artifact()
                state = artifact.to_json().encode()
                expected = ref["step"]["result_id"]
                checks = dict(stage=stage, key=case["key"], grad=sha(raw_grad), expected=expected,
                              actual=artifact.artifact_id, state_exact=state == E["FIT"]["blob"](root, expected).read_bytes(),
                              receipt_exact=trace == ref["step"], receipt=trace)
                sink.write(E["enc"](checks))
                need(checks["state_exact"] and checks["receipt_exact"] and
                     "sha256:" + sha(state) == model.geometry_artifact_id == expected, "source step differs from record")
                count += 1
            need(model.geometry_artifact_id == result["endpoints"][stage], "wrong final synthetic state")
    need(count == 8, "incomplete artificial comparison")
    E["save"](output / "result.json", dict(artificial_updates=count, exact_gradients=count,
              exact_states=count, exact_receipts=count, extension=ext, reference_plan=PLAN, reference_result=RESULT,
              pretrained_updates=0, evaluation_rows=0, checks=E["P"]["file_hash"](output / "checks.jsonl")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("toy", type=Path)
    ap.add_argument("expected_extension")
    ap.add_argument("output", type=Path)
    args = ap.parse_args()
    compare(args.toy, args.expected_extension, args.output)

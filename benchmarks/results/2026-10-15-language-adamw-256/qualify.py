"""Bounded pilot qualification, not a general scheduler or scoring runtime."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess

STOPS = (0, 64, 128, 256)
REVISION = "02f211f72a029ce6455aed2c706ca35989490c03"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    raw = value.encode() if isinstance(value, str) else (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), "wb") as output:
        output.write(raw)
    return digest(raw)


def load(repo, name):
    spec = importlib.util.spec_from_file_location(name, repo / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_sources(repo, request):
    for key, paths in (("train_documents", ("docs/byte_corpus_learning.md", "docs/byte_corpus_calibration.md")),
                       ("validation_documents", ("docs/resident_adamw.md",))):
        expected = [list(subprocess.check_output(["git", "show", REVISION + ":" + path], cwd=repo)) for path in paths]
        require(request[key] == expected, "declared ordered corpus differs: " + key)


def validate_comparison(protocol, raw, reference, actual, stored):
    measured = protocol.compare(raw, reference, actual)
    require(measured == stored, "stored comparison differs from recomputed actual/reference qualification")


def validate_coverage(request, selection):
    names = [c["name"] for c in request["cases"]]
    decisions = selection["decisions"]
    require([d["case_name"] for d in decisions] == names, "selection needs all case decisions in order")
    expected = [(name, stop) for name in names for stop in STOPS]
    items = selection["artifacts"]
    require([(i["case_name"], i["cursor"]) for i in items] == expected, "selection needs all case/cursor artifacts")
    for index, decision in enumerate(decisions):
        selected = decision["selected"]
        candidate = selected["candidate_index"]
        require(type(candidate) is int and 0 <= candidate < len(STOPS)
                and selected["completed_updates"] == STOPS[candidate], "selected candidate cursor differs")
        for item, stop in zip(items[index*len(STOPS):(index+1)*len(STOPS)], STOPS):
            require(item["file"] == f"case{index}-revision{stop}.json", "artifact filename differs")
            require(type(item["selected"]) is bool and item["selected"] == (stop == STOPS[candidate]), "selected artifact flags differ")
            if item["selected"]:
                require(item["artifact_id"] == selected["artifact_id"], "selected identity differs")


def select(repo, root):
    import spiraltorch as st
    protocol = load(repo, "byte_corpus_study")
    verifier = load(repo, "verify_byte_corpus_resume")
    raw = (root / "request.json").read_bytes()
    request = protocol.decode_json(raw)
    require(len(request["cases"]) == 9 and len(request["train_batches"]) == 256, "pilot budget")
    require([c["seed"] for c in request["cases"]] == [11]*3 + [29]*3 + [47]*3, "all fixed seeds")
    require(request["schema"] == protocol.REQUEST_V9 and request["design_schema"] == protocol.REQUEST_V2, "pilot design")
    require(request["config"] == dict(batch=2, steps=32, width=16, hidden=32, heads=2,
                blocks=[False], geometry_cols=4, curvature=-.75), "pilot architecture")
    require(protocol.f32(request["rate"]) == protocol.f32(.001)
            and request["checkpoint_every"] == 64 and len(request["validation_batches"]) == 32, "pilot rate/windows")
    require(request["optimizer"] == dict(kind="adamw_zero_rate_freeze.v1", beta1=.9, beta2=.999,
                epsilon=1e-5, weight_decay=.01, geometry_weight_decay=0.), "pilot optimizer")
    validate_sources(repo, request)
    comparison = json.loads((root / "comparison.json").read_bytes())
    require(comparison["request_sha256"] == digest(raw) and comparison["numerical_checks_passed"] is True, "independent comparison did not pass")
    require((root / "PLAN.md").read_bytes() == Path(__file__).with_name("PLAN.md").read_bytes(), "plan drift")
    completed = json.loads((root / "native/completed.json").read_bytes())
    require(completed["request_sha256"] == digest(raw), "run request drift")
    require([c["stop_after"] for c in completed["segments"]] == list(STOPS), "missing segment")
    study = st.nn.ByteCorpusStudy.from_json(raw)
    reports, checkpoints, bindings = [], [], []
    for stop, receipt in zip(STOPS, completed["segments"]):
        report_raw = (root / f"native/report-{stop}.json").read_bytes()
        checkpoint_raw = (root / f"native/checkpoint-{stop}.json").read_bytes()
        require(digest(report_raw) == receipt["report_sha256"], "segment report hash")
        require(digest(checkpoint_raw) == receipt["checkpoint_sha256"], "segment checkpoint hash")
        reports.append(protocol.decode_json(report_raw))
        checkpoints.append(study.checkpoint_from_json(checkpoint_raw))
    validate_comparison(protocol, raw, protocol.decode_json((root / "torch.json").read_bytes()), reports[-1], comparison)
    for stop, partial in zip(STOPS, reports):
        saved = protocol.decode_json((root / f"native/checkpoint-{stop}.json").read_bytes())
        # This binds weights/moments/histories to the segmented run, not a second
        # uninterrupted execution. Do not label it an exact-resume experiment.
        verifier.verify(request, digest(raw), reports[-1], partial, reports[-1], saved)
        bindings.append(dict(cursor=stop, segment_checkpoint_binding=True))
    output = root / "selection"
    output.mkdir(mode=0o700, exist_ok=False)
    decisions, artifacts = [], []
    for index, case in enumerate(request["cases"]):
        selected, decision_json = study.select_checkpoint(checkpoints, case["name"], policy="fixed_window_frozen.v1")
        decision = json.loads(decision_json)
        # Independent, experiment-only check of the fixed Rust decision.
        means = [sum(protocol.f32(v) for v in report["cases"][index]["validation"][-1]["batch_losses"])
                 / len(request["validation_batches"]) for report in reports]
        expected = min(range(len(STOPS)), key=lambda i: (means[i], STOPS[i]))
        require(decision["selected"]["completed_updates"] == STOPS[expected], "selection oracle")
        study.validate_artifact(checkpoints[expected], selected)
        decisions.append(decision)
        for stop, checkpoint in zip(STOPS, checkpoints):
            artifact = study.case_artifact(checkpoint, case["name"], policy="fixed_window_frozen.v1")
            filename = f"case{index}-revision{stop}.json"
            write(output / filename, artifact.to_json())
            artifacts.append(dict(case_name=case["name"], cursor=stop, file=filename,
                                  artifact_id=artifact.artifact_id,
                                  selected=artifact.artifact_id == selected.artifact_id))
    record = dict(schema="spiraltorch.byte_corpus.pilot_selection.v1", request_sha256=digest(raw),
                  plan_sha256=digest((root / "PLAN.md").read_bytes()),
                  checkpoints=bindings, decisions=decisions, artifacts=artifacts)
    write(root / "selection.json", record)
    print(json.dumps(dict(selected_cursors=[d["selected"]["completed_updates"] for d in decisions])))


def baselines(request, cursor, windows):
    unigram, bigram, previous = Counter(), Counter(), Counter()
    steps = request["config"]["steps"]
    for batch in request["train_batches"][:cursor]:
        for doc, start in batch:
            row = request["train_documents"][doc][start:start + steps + 1]
            unigram.update(row[1:])
            bigram.update(zip(row, row[1:]))
            previous.update(row[:-1])
    losses = [[], []]
    for row in windows:
        for x, y in zip(row, row[1:]):
            losses[0].append(-math.log((unigram[y] + 1) / (sum(unigram.values()) + 256)))
            losses[1].append(-math.log((bigram[x, y] + 1) / (previous[x] + 256)))
    return dict(unigram_ce=sum(losses[0]) / len(losses[0]), bigram_ce=sum(losses[1]) / len(losses[1]))


def generation_oracle(owner, prompts, count, st):
    windows = list(prompts)
    generated = [[] for _ in prompts]
    hashes = []
    steps = len(prompts[0])
    for _ in range(count):
        batch = st.nn.ByteLmBatch.from_windows([row + b"\xff" for row in windows])
        forward = owner.forward(owner.prepare_batch(batch))
        logits = forward.prediction_tensor().snapshot().read_values()
        last = []
        for row in range(len(prompts)):
            scores = logits[((row + 1) * steps - 1) * 256:(row + 1) * steps * 256]
            require(all(math.isfinite(v) for v in scores), "nonfinite generation oracle")
            byte = max(range(256), key=scores.__getitem__)
            generated[row].append(byte)
            windows[row] = windows[row][1:] + bytes([byte])
            last.extend(scores)
        hashes.append(digest(struct.pack("<" + "f" * len(last), *last)))
    return generated, hashes


def evaluate(repo, root):
    import spiraltorch as st
    protocol = load(repo, "byte_corpus_study")
    raw = (root / "request.json").read_bytes()
    request = protocol.decode_json(raw)
    selection_raw = (root / "selection.json").read_bytes()
    selection = json.loads(selection_raw)
    require(selection["request_sha256"] == digest(raw), "selection source drift")
    require(selection["plan_sha256"] == digest((root / "PLAN.md").read_bytes()), "plan drift")
    validate_sources(repo, request)
    validate_coverage(request, selection)
    study = st.nn.ByteCorpusStudy.from_json(raw)
    checkpoints = [study.checkpoint_from_json((root / f"native/checkpoint-{stop}.json").read_bytes()) for stop in STOPS]
    for index, decision in enumerate(selection["decisions"]):
        chosen, actual_decision = study.select_checkpoint(checkpoints, decision["case_name"], policy="fixed_window_frozen.v1")
        require(json.loads(actual_decision) == decision, "stored selection differs from Rust")
        for stop, checkpoint, item in zip(STOPS, checkpoints, selection["artifacts"][index*len(STOPS):(index+1)*len(STOPS)]):
            expected = study.case_artifact(checkpoint, decision["case_name"], policy="fixed_window_frozen.v1")
            require(item["artifact_id"] == expected.artifact_id, "artifact differs from source checkpoint")
            require((root / "selection" / item["file"]).read_bytes() == expected.to_json().encode(), "artifact bytes differ from source")
    output = root / "evaluation"
    output.mkdir(mode=0o700, exist_ok=False)
    write(output / "claim.json", dict(selection_sha256=digest(selection_raw), before_probe_read=True))
    # No probe contents are read until every case has a fixed selected artifact.
    probe_path = "docs/byte_corpus_query_controls.md"
    probe_raw = subprocess.check_output(["git", "show", REVISION + ":" + probe_path], cwd=repo)
    require(list(probe_raw) not in request["train_documents"] + request["validation_documents"], "probe document overlap")
    b, t = request["config"]["batch"], request["config"]["steps"]
    require((b, t) == (2, 32), "pilot shape")
    available = protocol.windows([probe_raw], t)
    require(len(available) >= 64, "probe windows")
    starts = [available[i * len(available) // 64][1] for i in range(64)]
    probe_windows = [probe_raw[start:start + t + 1] for start in starts]
    valid = [bytes(request["validation_documents"][d][s:s+t+1]) for batch in request["validation_batches"] for d,s in batch]
    prompts = [row[:t] for row in valid[:b]]
    reports = {stop: protocol.decode_json((root / f"native/report-{stop}.json").read_bytes()) for stop in STOPS}
    rows = []
    for index, item in enumerate(selection["artifacts"]):
        artifact_raw = (root / "selection" / item["file"]).read_bytes()
        artifact = st.nn.ByteCorpusModelArtifact.from_json(artifact_raw, expected_id=item["artifact_id"])
        require(artifact.case_name == item["case_name"], "artifact case drift")
        owner = artifact.restore_wgpu()
        require(owner.tensor_device().adapter_info()["device_type"] != "Cpu", "real GPU required")
        batch = st.nn.ByteLmBatch.from_windows(valid[:b])
        forward = owner.forward(owner.prepare_batch(batch))
        actual_ce = forward.next_byte_loss().loss_tensor().snapshot().read_values()[0]
        case = next(c for c in reports[item["cursor"]]["cases"] if c["name"] == item["case_name"])
        require(struct.pack("<f", actual_ce) == struct.pack("<f", case["validation"][-1]["batch_losses"][0]), "fresh artifact loss binding")
        generation = json.loads(owner.generate_greedy(prompts, max_new_bytes=32))
        expected, hashes = generation_oracle(artifact.restore_wgpu(), prompts, 32, st)
        require(generation["generated_bytes"] == expected and generation["next_logits_f32le_sha256"] == hashes, "generation full-forward oracle")
        flattened = list(itertools.chain.from_iterable(expected))
        row = dict(case_name=item["case_name"], cursor=item["cursor"], artifact_id=artifact.artifact_id,
                   selected=item["selected"], development=case["validation"][-1],
                   generation=generation, generated_latin1=[bytes(r).decode("latin1") for r in expected],
                   space_fraction=flattened.count(32) / len(flattened),
                   distinct_bytes=len(set(flattened)), fresh_loss_exact_f32=True,
                   generation_full_forward_oracle=True)
        if item["selected"] or item["cursor"] == STOPS[-1]:
            losses = []
            for start in range(0, len(probe_windows), b):
                batch = st.nn.ByteLmBatch.from_windows(probe_windows[start:start+b])
                loss = owner.forward(owner.prepare_batch(batch)).next_byte_loss().loss_tensor().snapshot().read_values()[0]
                require(math.isfinite(loss), "nonfinite probe loss")
                losses.append(loss)
            row["probe"] = dict(batch_losses=losses, mean_ce=sum(losses) / len(losses), target_bytes=len(probe_windows) * t)
        write(output / f"case-{index}.json", row)
        rows.append(row)
        print(json.dumps(dict(evaluated=item["case_name"], cursor=item["cursor"], selected=item["selected"], spaces=row["space_fraction"])), flush=True)
    result = dict(schema="spiraltorch.byte_corpus.language_budget.v1", request_sha256=digest(raw),
                  selection_sha256=digest(selection_raw), prompt_sha256=digest(b"".join(prompts)),
                  probe=dict(path=probe_path, source_revision=REVISION, sha256=digest(probe_raw), starts=starts),
                  baselines=[dict(cursor=stop, development=baselines(request, stop, valid),
                                  probe=baselines(request, stop, probe_windows)) for stop in STOPS],
                  cases=rows)
    write(root / "qualification.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "evaluate"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    {"select": select, "evaluate": evaluate}[args.command](args.repo, args.root)


if __name__ == "__main__":
    main()
